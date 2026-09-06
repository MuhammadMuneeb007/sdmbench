"""Methodology -- the full specification of one modelling approach.

A **methodology** is not an algorithm. It is the whole chain::

    modalities + scales + representations + fusion
        + spatial structure + temporal structure
        + objective + predictor

"Random Forest" is an algorithm. *"Encode climate as raw tabular values and
terrain with a multi-scale CNN at 1/5/25 km, fuse by cross-attention, build a
geographic k-NN graph, and train a Graph Transformer under a Poisson
point-process objective"* is a methodology. The benchmark compares
methodologies; algorithms are one field inside them.

Making this a first-class object has three practical payoffs:

1. **Validity checking before compute.** :meth:`Methodology.validate` catches
   "you asked for a CNN on a tabular modality" in milliseconds rather than
   after an hour of data preparation.
2. **Hashing.** A methodology hashes to a stable id, so results are keyed by
   *what was actually done*, and a resumed run recognises completed work.
3. **Ablation.** :meth:`Methodology.ablate_modality` and friends produce the
   variant methodologies that the staged planner needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from sdmbench.exceptions import ConfigurationError
from sdmbench.reproducibility.hashes import hash_json

__all__ = ["ModalitySpec", "Methodology", "ValidationIssue", "MethodologyValidation"]


@dataclass(frozen=True)
class ModalitySpec:
    """How one modality participates in a methodology."""

    name: str
    representation: str = "raw-tabular"
    #: Spatial scales, for raster modalities: ``["1km", "5km", "25km"]``.
    scales: tuple[str, ...] = ()
    #: Data provider, when the modality must be fetched.
    provider: str = ""
    #: Encoder keyword arguments.
    options: Mapping[str, Any] = field(default_factory=dict)
    #: Explicit override of the modality's structural kind.
    kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "representation": self.representation,
            "scales": list(self.scales),
            "provider": self.provider,
            "options": dict(self.options),
            "kind": self.kind,
        }

    @property
    def is_multiscale(self) -> bool:
        return len(self.scales) > 1


@dataclass
class ValidationIssue:
    """One problem found while validating a methodology."""

    severity: str  # "error" or "warning"
    component: str
    message: str
    suggestion: str = ""

    def render(self) -> str:
        prefix = "ERROR" if self.severity == "error" else "WARN "
        line = f"[{prefix}] {self.component}: {self.message}"
        return line + (f"\n         -> {self.suggestion}" if self.suggestion else "")


@dataclass
class MethodologyValidation:
    """The result of checking a methodology."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def is_runnable(self) -> bool:
        """Valid *and* every dependency importable here."""
        return self.is_valid and not any(
            "not installed" in i.message or "missing" in i.message for i in self.warnings
        )

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ConfigurationError(
                "invalid methodology:\n" + "\n".join(i.render() for i in self.errors)
            )

    def render(self) -> str:
        if not self.issues:
            return "Methodology is valid; all components are available."
        return "\n".join(i.render() for i in self.issues)


@dataclass
class Methodology:
    """A complete modelling approach.

    Examples
    --------
    >>> m = Methodology(
    ...     name="climate_only_rf",
    ...     modalities=[ModalitySpec("climate", "raw-tabular")],
    ...     model="random-forest-sklearn",
    ... )
    >>> m.modality_names
    ['climate']
    """

    name: str
    modalities: list[ModalitySpec] = field(default_factory=list)
    fusion: str = "concat"
    fusion_options: dict[str, Any] = field(default_factory=dict)
    #: Graph construction strategy, or ``None`` for no graph.
    graph: str | None = None
    graph_options: dict[str, Any] = field(default_factory=dict)
    #: Temporal encoder applied to sequence modalities.
    temporal: str | None = None
    temporal_options: dict[str, Any] = field(default_factory=dict)
    objective: str = "weighted-bce"
    objective_options: dict[str, Any] = field(default_factory=dict)
    model: str = "random-forest-sklearn"
    model_options: dict[str, Any] = field(default_factory=dict)
    #: Whether coordinates may be used as predictive features. Off by default.
    include_coordinates: bool = False
    #: Free-text description used in reports.
    description: str = ""
    #: Descriptive metadata only -- explicitly not authoritative scores.
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- access --
    @property
    def modality_names(self) -> list[str]:
        return [m.name for m in self.modalities]

    def modality(self, name: str) -> ModalitySpec | None:
        for spec in self.modalities:
            if spec.name == name:
                return spec
        return None

    @property
    def uses_graph(self) -> bool:
        return self.graph is not None

    @property
    def scales(self) -> dict[str, list[str]]:
        return {m.name: list(m.scales) for m in self.modalities if m.scales}

    # -------------------------------------------------------------- identity --
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modalities": [m.to_dict() for m in self.modalities],
            "fusion": self.fusion,
            "fusion_options": self.fusion_options,
            "graph": self.graph,
            "graph_options": self.graph_options,
            "temporal": self.temporal,
            "temporal_options": self.temporal_options,
            "objective": self.objective,
            "objective_options": self.objective_options,
            "model": self.model,
            "model_options": self.model_options,
            "include_coordinates": self.include_coordinates,
            "description": self.description,
        }

    @property
    def methodology_hash(self) -> str:
        """Stable hash of everything that affects the result.

        ``name`` and ``description`` are excluded: renaming a methodology must
        not invalidate cached results computed under it.
        """
        payload = self.to_dict()
        payload.pop("name", None)
        payload.pop("description", None)
        return hash_json(payload)

    @property
    def short_id(self) -> str:
        return f"{self.name}@{self.methodology_hash[:8]}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Methodology:
        modalities: list[ModalitySpec] = []
        raw = data.get("modalities", [])
        if isinstance(raw, Mapping):
            # YAML mapping form: {climate: {representation: raw}, ...}
            for name, spec in raw.items():
                spec = spec or {}
                modalities.append(
                    ModalitySpec(
                        name=str(name),
                        representation=str(spec.get("representation", "raw-tabular")),
                        scales=tuple(spec.get("scales", ()) or ()),
                        provider=str(spec.get("provider", "")),
                        options=dict(spec.get("options", {}) or {}),
                        kind=str(spec.get("kind", "")),
                    )
                )
        else:
            for spec in raw:
                if isinstance(spec, str):
                    modalities.append(ModalitySpec(name=spec))
                else:
                    modalities.append(
                        ModalitySpec(
                            name=str(spec["name"]),
                            representation=str(spec.get("representation", "raw-tabular")),
                            scales=tuple(spec.get("scales", ()) or ()),
                            provider=str(spec.get("provider", "")),
                            options=dict(spec.get("options", {}) or {}),
                            kind=str(spec.get("kind", "")),
                        )
                    )

        fusion = data.get("fusion", "concat")
        fusion_options: dict[str, Any] = {}
        if isinstance(fusion, Mapping):
            fusion_options = {k: v for k, v in fusion.items() if k != "method"}
            fusion = fusion.get("method", "concat")

        graph = data.get("graph") or data.get("spatial")
        graph_options: dict[str, Any] = {}
        if isinstance(graph, Mapping):
            graph_options = {
                k: v for k, v in graph.items() if k not in {"method", "representation"}
            }
            graph = graph.get("method") or graph.get("representation")

        objective = data.get("objective", "weighted-bce")
        objective_options: dict[str, Any] = {}
        if isinstance(objective, Mapping):
            objective_options = {k: v for k, v in objective.items() if k != "method"}
            objective = objective.get("method", "weighted-bce")

        model = data.get("model") or data.get("graph_model") or "random-forest-sklearn"
        model_options: dict[str, Any] = {}
        if isinstance(model, Mapping):
            model_options = {k: v for k, v in model.items() if k != "model"}
            model = model.get("model", "random-forest-sklearn")

        return cls(
            name=str(data.get("name", "methodology")),
            modalities=modalities,
            fusion=str(fusion),
            fusion_options=fusion_options or dict(data.get("fusion_options", {})),
            graph=str(graph) if graph else None,
            graph_options=graph_options or dict(data.get("graph_options", {})),
            temporal=data.get("temporal"),
            temporal_options=dict(data.get("temporal_options", {})),
            objective=str(objective),
            objective_options=objective_options or dict(data.get("objective_options", {})),
            model=str(model),
            model_options=model_options or dict(data.get("model_options", {})),
            include_coordinates=bool(data.get("include_coordinates", False)),
            description=str(data.get("description", "")),
            metadata=dict(data.get("metadata", {})),
        )

    # -------------------------------------------------------------- ablation --
    def ablate_modality(self, name: str) -> Methodology:
        """A copy with one modality removed."""
        remaining = [m for m in self.modalities if m.name != name]
        if len(remaining) == len(self.modalities):
            raise ConfigurationError(f"methodology {self.name!r} has no modality {name!r}")
        if not remaining:
            raise ConfigurationError(
                f"removing {name!r} would leave no modalities; a model needs some input"
            )
        return replace(self, name=f"{self.name}-minus-{name}", modalities=remaining)

    def with_modalities(self, names: Iterable[str]) -> Methodology:
        """A copy keeping only the named modalities."""
        keep = list(names)
        chosen = [m for m in self.modalities if m.name in keep]
        if not chosen:
            raise ConfigurationError(f"none of {keep} are present in {self.name!r}")
        return replace(self, name=f"{self.name}-{'+'.join(keep)}", modalities=chosen)

    def with_representation(self, modality: str, representation: str) -> Methodology:
        """A copy using a different encoder for one modality."""
        updated = [
            replace(m, representation=representation) if m.name == modality else m
            for m in self.modalities
        ]
        return replace(self, name=f"{self.name}-{modality}={representation}", modalities=updated)

    def with_scales(self, modality: str, scales: Iterable[str]) -> Methodology:
        """A copy using different spatial scales for one modality."""
        scale_tuple = tuple(scales)
        updated = [
            replace(m, scales=scale_tuple) if m.name == modality else m
            for m in self.modalities
        ]
        label = "+".join(scale_tuple) if scale_tuple else "point"
        return replace(self, name=f"{self.name}-{modality}@{label}", modalities=updated)

    def with_fusion(self, fusion: str, **options: Any) -> Methodology:
        return replace(
            self, name=f"{self.name}-fuse={fusion}", fusion=fusion, fusion_options=options
        )

    def with_graph(self, graph: str | None, **options: Any) -> Methodology:
        label = graph or "none"
        return replace(
            self, name=f"{self.name}-graph={label}", graph=graph, graph_options=options
        )

    def with_objective(self, objective: str, **options: Any) -> Methodology:
        return replace(
            self,
            name=f"{self.name}-obj={objective}",
            objective=objective,
            objective_options=options,
        )

    def with_model(self, model: str, **options: Any) -> Methodology:
        return replace(
            self, name=f"{self.name}-{model}", model=model, model_options=options
        )

    # ------------------------------------------------------------- validation --
    def validate(self, *, available_modalities: Iterable[str] | None = None) -> MethodologyValidation:
        """Check the methodology is coherent before anything is computed."""
        from sdmbench.core.modality import STANDARD_MODALITIES
        from sdmbench.fusion import FUSION_STRATEGIES
        from sdmbench.graphs import GRAPH_BUILDERS
        from sdmbench.models.base import MODEL_REGISTRY, _load_all_adapters
        from sdmbench.objectives import OBJECTIVES
        from sdmbench.representations import REPRESENTATIONS

        validation = MethodologyValidation()
        available = set(available_modalities) if available_modalities is not None else None

        if not self.modalities:
            validation.issues.append(
                ValidationIssue("error", "modalities", "no modalities specified")
            )

        for spec in self.modalities:
            if available is not None and spec.name not in available:
                validation.issues.append(
                    ValidationIssue(
                        "error",
                        f"modality:{spec.name}",
                        "not available in this dataset",
                        f"available: {sorted(available)}",
                    )
                )
                continue

            if spec.representation not in REPRESENTATIONS:
                validation.issues.append(
                    ValidationIssue(
                        "error",
                        f"modality:{spec.name}",
                        f"unknown representation {spec.representation!r}",
                        f"available: {', '.join(REPRESENTATIONS.names())}",
                    )
                )
                continue

            entry = REPRESENTATIONS.resolve(spec.representation)
            ok, detail = entry.available()
            if not ok:
                validation.issues.append(
                    ValidationIssue(
                        "warning",
                        f"modality:{spec.name}",
                        f"encoder {spec.representation!r} is not installed ({detail})",
                        "this component will be SKIPPED_DEPENDENCY at run time",
                    )
                )

            # Encoder/modality kind compatibility -- the most common config error.
            definition = STANDARD_MODALITIES.get(spec.name)
            kind = spec.kind or (definition.kind.value if definition else None)
            if kind and not entry.spec.accepts_kind(kind):
                validation.issues.append(
                    ValidationIssue(
                        "error",
                        f"modality:{spec.name}",
                        f"encoder {spec.representation!r} accepts "
                        f"{list(entry.spec.accepts)} but this modality is {kind!r}",
                        f"try one of: {', '.join(REPRESENTATIONS.accepting(kind))}",
                    )
                )

            if spec.scales and kind not in {"raster", None}:
                validation.issues.append(
                    ValidationIssue(
                        "warning",
                        f"modality:{spec.name}",
                        f"scales {list(spec.scales)} were given for a {kind!r} modality",
                        "scales only apply to raster modalities and will be ignored",
                    )
                )

        if self.fusion not in FUSION_STRATEGIES:
            validation.issues.append(
                ValidationIssue(
                    "error", "fusion", f"unknown fusion strategy {self.fusion!r}",
                    f"available: {', '.join(FUSION_STRATEGIES.names())}",
                )
            )
        elif len(self.modalities) == 1 and self.fusion != "concat":
            validation.issues.append(
                ValidationIssue(
                    "warning", "fusion",
                    f"fusion {self.fusion!r} with a single modality has nothing to fuse",
                    "use 'concat', or add modalities",
                )
            )

        if self.graph is not None and self.graph not in GRAPH_BUILDERS:
            validation.issues.append(
                ValidationIssue(
                    "error", "graph", f"unknown graph builder {self.graph!r}",
                    f"available: {', '.join(GRAPH_BUILDERS.names())}",
                )
            )

        if self.objective not in OBJECTIVES:
            validation.issues.append(
                ValidationIssue(
                    "error", "objective", f"unknown objective {self.objective!r}",
                    f"available: {', '.join(OBJECTIVES.names())}",
                )
            )

        if not MODEL_REGISTRY:
            _load_all_adapters()
        if self.model not in MODEL_REGISTRY:
            validation.issues.append(
                ValidationIssue(
                    "error", "model", f"unknown model {self.model!r}",
                    f"available: {', '.join(sorted(MODEL_REGISTRY))}",
                )
            )
        else:
            model_class = MODEL_REGISTRY[self.model]
            family = getattr(model_class, "family", "")
            # A non-neural model cannot optimise a custom differentiable
            # objective; saying so up front avoids a silently ignored setting.
            if self.objective not in {"bce", "weighted-bce"} and family not in {
                "neural", "graph"
            }:
                validation.issues.append(
                    ValidationIssue(
                        "warning", "objective",
                        f"model {self.model!r} ({family}) has a fixed internal objective; "
                        f"{self.objective!r} cannot be applied to it",
                        "use a neural or graph model to vary the objective",
                    )
                )
            if self.graph is not None and family != "graph":
                validation.issues.append(
                    ValidationIssue(
                        "warning", "graph",
                        f"a graph was specified but {self.model!r} is not a graph model",
                        "the graph will be ignored; use gcn/graphsage/gatv2/graph-transformer",
                    )
                )

        if self.include_coordinates:
            validation.issues.append(
                ValidationIssue(
                    "warning", "coordinates",
                    "coordinates are enabled as predictive features",
                    "this inflates interpolation performance and harms transferability; "
                    "the results will record it",
                )
            )
        return validation

    def render(self) -> str:
        lines = [f"Methodology: {self.name}  [{self.methodology_hash[:8]}]"]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append("  modalities:")
        for spec in self.modalities:
            scales = f" @ {', '.join(spec.scales)}" if spec.scales else ""
            lines.append(f"    - {spec.name:<20} {spec.representation}{scales}")
        lines.append(f"  fusion:     {self.fusion}")
        lines.append(f"  graph:      {self.graph or '(none)'}")
        lines.append(f"  temporal:   {self.temporal or '(none)'}")
        lines.append(f"  objective:  {self.objective}")
        lines.append(f"  model:      {self.model}")
        return "\n".join(lines)
