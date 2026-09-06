"""The plugin registry.

Every extensible axis of sdmbench -- datasets, modalities, representations,
fusion strategies, graph builders, objectives, models, metrics, explainers --
is a :class:`Registry`. A researcher adds one new component by writing a class
and decorating it; the benchmark engine is never edited.

    @REPRESENTATIONS.register("my_encoder")
    class MyEncoder(BaseEncoder):
        ...

Third-party packages can extend sdmbench without being imported by it, via
Python entry points declared in their own ``pyproject.toml``::

    [project.entry-points."sdmbench.representations"]
    my_encoder = "mypkg.encoders:MyEncoder"

:meth:`Registry.load_entry_points` discovers those at first use.

Design notes
------------
* Registration is explicit and eager; lookup is lazy. Importing ``sdmbench``
  must not import torch.
* A registry stores a :class:`Spec` alongside each component -- what it accepts,
  what it produces, what it depends on. That metadata is what lets the
  methodology planner decide whether a combination is even *valid* before
  spending compute on it.
* Duplicate names raise. Silently shadowing a component would make a benchmark
  irreproducible in the worst way: the config would look right.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, TypeVar

from sdmbench.exceptions import ConfigurationError

__all__ = ["Registry", "Spec", "ComponentEntry", "REGISTRIES", "all_registries"]

T = TypeVar("T")


@dataclass
class Spec:
    """Declarative metadata about a component.

    This is what makes a methodology *checkable*. A representation encoder that
    accepts only raster modalities cannot be applied to a tabular one, and the
    planner should say so before a GPU is allocated rather than after.
    """

    #: Modality kinds this component accepts (``"tabular"``, ``"raster"``,
    #: ``"sequence"``, ``"graph"``, ``"any"``).
    accepts: tuple[str, ...] = ("any",)
    #: What it produces.
    produces: str = "vector"
    #: Importable modules required at run time.
    requires: tuple[str, ...] = ()
    #: ``pip install "sdmbench[extra]"`` that provides them.
    extra: str | None = None
    #: Fixed output dimensionality, when known ahead of fitting.
    output_dim: int | None = None
    #: Whether parameters are learned (and therefore must be fitted on training
    #: data only -- a leakage-relevant property).
    trainable: bool = False
    #: Where pretrained weights come from, for provenance.
    pretraining_source: str = ""
    #: Free-text description used by ``sdmbench <registry> list``.
    description: str = ""
    #: Descriptive research metadata. Explicitly NOT authoritative scoring.
    tags: tuple[str, ...] = ()
    maturity: str = "stable"
    #: Anything else a component wants to advertise.
    extras: dict[str, Any] = field(default_factory=dict)

    def accepts_kind(self, kind: str) -> bool:
        return "any" in self.accepts or kind in self.accepts

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepts": list(self.accepts),
            "produces": self.produces,
            "requires": list(self.requires),
            "extra": self.extra,
            "output_dim": self.output_dim,
            "trainable": self.trainable,
            "pretraining_source": self.pretraining_source,
            "description": self.description,
            "tags": list(self.tags),
            "maturity": self.maturity,
            **self.extras,
        }


@dataclass
class ComponentEntry(Generic[T]):
    """A registered component and its metadata."""

    name: str
    component: type[T]
    spec: Spec
    aliases: tuple[str, ...] = ()
    source: str = "builtin"

    @property
    def qualified_name(self) -> str:
        return f"{self.component.__module__}.{self.component.__qualname__}"

    def available(self) -> tuple[bool, str]:
        """Whether this component's dependencies are importable here."""
        from sdmbench.optional import have

        missing = [m for m in self.spec.requires if not have(m)]
        if not missing:
            return True, ""
        hint = f'pip install "sdmbench[{self.spec.extra}]"' if self.spec.extra else ""
        return False, f"missing: {', '.join(missing)}" + (f" -- {hint}" if hint else "")

    def to_dict(self) -> dict[str, Any]:
        available, detail = self.available()
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "class": self.qualified_name,
            "source": self.source,
            "available": available,
            "availability_detail": detail,
            "spec": self.spec.to_dict(),
        }


class Registry(Generic[T]):
    """A named collection of pluggable components.

    Parameters
    ----------
    kind:
        What this registry holds, e.g. ``"representation"``. Used in error
        messages and as the entry-point group suffix.
    """

    def __init__(self, kind: str, *, entry_point_group: str | None = None) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group or f"sdmbench.{kind}s"
        self._entries: dict[str, ComponentEntry[T]] = {}
        self._aliases: dict[str, str] = {}
        self._entry_points_loaded = False
        REGISTRIES[kind] = self

    # ------------------------------------------------------------- register --
    def register(
        self,
        name: str,
        *aliases: str,
        spec: Spec | None = None,
        **spec_kwargs: Any,
    ) -> Callable[[type[T]], type[T]]:
        """Class decorator registering a component under ``name``.

        ``spec`` may be passed directly, or its fields as keyword arguments.
        """
        resolved_spec = spec or Spec(**spec_kwargs)

        def decorator(component: type[T]) -> type[T]:
            self.add(name, component, spec=resolved_spec, aliases=aliases)
            return component

        return decorator

    def add(
        self,
        name: str,
        component: type[T],
        *,
        spec: Spec | None = None,
        aliases: tuple[str, ...] = (),
        source: str = "builtin",
        replace: bool = False,
    ) -> ComponentEntry[T]:
        """Register a component imperatively."""
        key = _normalise(name)
        if key in self._entries and not replace:
            existing = self._entries[key].qualified_name
            raise ConfigurationError(
                f"{self.kind} {name!r} is already registered by {existing}. "
                "Choose a different name, or pass replace=True deliberately."
            )
        entry = ComponentEntry(
            name=key,
            component=component,
            spec=spec or Spec(),
            aliases=tuple(_normalise(a) for a in aliases),
            source=source,
        )
        self._entries[key] = entry
        for alias in entry.aliases:
            if alias in self._aliases and not replace:
                raise ConfigurationError(
                    f"{self.kind} alias {alias!r} already points to {self._aliases[alias]!r}"
                )
            self._aliases[alias] = key
        # Let the component know its registered name, the way model adapters
        # expect, without requiring every base class to define it.
        if getattr(component, "name", None) in (None, "", "model", "component"):
            try:
                component.name = key  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass
        return entry

    # ----------------------------------------------------------------- read --
    def load_entry_points(self) -> int:
        """Discover third-party components declared via entry points."""
        if self._entry_points_loaded:
            return 0
        self._entry_points_loaded = True
        loaded = 0
        try:
            points = importlib_metadata.entry_points(group=self.entry_point_group)
        except Exception:  # noqa: BLE001 - entry point discovery must never be fatal
            return 0
        for point in points:
            try:
                component = point.load()
            except Exception:  # noqa: BLE001 - a broken plugin skips, it does not crash
                continue
            spec = getattr(component, "spec", None)
            self.add(
                point.name,
                component,
                spec=spec if isinstance(spec, Spec) else None,
                source=f"entry_point:{getattr(point, 'value', point.name)}",
                replace=True,
            )
            loaded += 1
        return loaded

    def resolve(self, name: str) -> ComponentEntry[T]:
        """Look up an entry by name or alias."""
        self.load_entry_points()
        key = _normalise(name)
        key = self._aliases.get(key, key)
        entry = self._entries.get(key)
        if entry is None:
            raise ConfigurationError(
                f"unknown {self.kind} {name!r}.\n"
                f"  Available: {', '.join(self.names()) or '(none registered)'}"
            )
        return entry

    def get(self, name: str) -> type[T]:
        """The component class registered under ``name``."""
        return self.resolve(name).component

    def create(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered component."""
        return self.resolve(name).component(**kwargs)

    def spec(self, name: str) -> Spec:
        return self.resolve(name).spec

    def names(self) -> list[str]:
        self.load_entry_points()
        return sorted(self._entries)

    def entries(self) -> list[ComponentEntry[T]]:
        self.load_entry_points()
        return [self._entries[k] for k in sorted(self._entries)]

    def available_names(self) -> list[str]:
        """Names whose dependencies are importable in this environment."""
        return [e.name for e in self.entries() if e.available()[0]]

    def describe(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries()]

    def accepting(self, kind: str) -> list[str]:
        """Components that accept a given modality kind."""
        return [e.name for e in self.entries() if e.spec.accepts_kind(kind)]

    def __contains__(self, name: str) -> bool:
        self.load_entry_points()
        key = _normalise(name)
        return key in self._entries or key in self._aliases

    def __iter__(self) -> Iterator[ComponentEntry[T]]:
        return iter(self.entries())

    def __len__(self) -> int:
        self.load_entry_points()
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Registry {self.kind!r}: {len(self._entries)} component(s)>"


def _normalise(name: str) -> str:
    """Registry keys are lowercase with hyphens, so ``my_encoder`` == ``my-encoder``."""
    return name.strip().lower().replace("_", "-")


#: Every registry, by kind. Populated as each module is imported.
REGISTRIES: dict[str, Registry[Any]] = {}


def all_registries() -> dict[str, Registry[Any]]:
    """Every registry, after importing the modules that populate them."""
    import importlib

    for module in (
        "sdmbench.core.modality",
        "sdmbench.representations",
        "sdmbench.fusion",
        "sdmbench.graphs",
        "sdmbench.objectives",
        "sdmbench.temporal",
        "sdmbench.explain",
    ):
        try:
            importlib.import_module(module)
        except ImportError:  # pragma: no cover - optional subsystems
            continue
    return dict(REGISTRIES)
