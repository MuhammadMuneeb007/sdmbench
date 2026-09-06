"""TabPFN adapters, including the released finetuned SDM checkpoints.

The variants below are the ones Dinnage & Warren (2026) sec. 2.2.2 actually
tested -- the names and settings are taken from the paper, not invented:

``tabpfn-default``
    "the standard TabPFN 2.5 classifier with default parameters:
    n_estimators = 8, softmax_temperature = 0.9, balance_probabilities = FALSE,
    average_before_softmax = FALSE."
``tabpfn-real``
    "uses the ``v2_5_real`` checkpoint, which was additionally finetuned on
    real-world tabular datasets."
``tabpfn-balanced``
    "adds two adjustments to the base configurations. The
    balance_probabilities setting rescales predicted class probabilities to
    account for observed class frequencies in the training data. The
    average_before_softmax setting averages logits across ensemble members
    before applying the softmax transformation."
``tabpfn-ss``
    Subsample ensemble -- the class-balancing scheme of sec. 2.3, with K = 16,
    ``n_estimators = 16``, ``average_before_softmax = True`` and
    ``balance_probabilities = True``.
``tabpfn-sdm`` / ``tabpfn-sdm-spatial``
    The released finetuned checkpoints combined with the subsample ensemble.

Licensing
---------
TabPFN weights are distributed under the Prior Labs License v1.1. sdmbench
downloads them through ``huggingface_hub`` and never attempts to bypass
authentication or licence acceptance. If a download is gated, the job becomes
``SKIPPED_LICENSE`` with instructions.

Version caveat
--------------
The paper used the ``tabpfn`` Python package **version 2.5**, while the
finetuned checkpoints declare ``Prior-Labs/TabPFN-v2-clf`` as their base model
and the model card's loading snippet targets the v2 API
(``clf.models_[0].load_state_dict(...)``). :meth:`FinetunedTabPFNAdapter._load_checkpoint`
follows the model card's documented procedure and reports clearly if the
installed package exposes a different internal layout, rather than silently
loading partial weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sdmbench.exceptions import LicenseError, SdmbenchError
from sdmbench.models.base import ModelAdapter, register_model
from sdmbench.models.ensemble import PUBLISHED_K, PresenceBackgroundEnsemble
from sdmbench.optional import package_version, require, resolve_device
from sdmbench.paths import ensure_dir, models_dir
from sdmbench.reproducibility.hashes import hash_file

__all__ = [
    "TabPFNAdapter",
    "TabPFNBalancedAdapter",
    "TabPFNSubsampleEnsembleAdapter",
    "FinetunedTabPFNAdapter",
    "CheckpointInfo",
    "download_checkpoint",
    "HF_REPO",
    "CHECKPOINTS",
]

#: Hugging Face repository holding the released finetuned weights.
HF_REPO = "rdinnager/tabpfn-sdm-finetuned"

#: Base model the checkpoints were finetuned from (model card ``config.json``).
BASE_MODEL = "Prior-Labs/TabPFN-v2-clf"

#: Checkpoint filenames, with the validation metrics reported in the model
#: card's ``config.json``. These are CHECKPOINT-VALIDATION metrics measured on
#: held-out *species* during finetuning -- they are NOT the benchmark aggregate
#: results reported in the paper, and must never be presented as such. The
#: benchmark figures live in ``benchmarks/tabpfn_sdm_2026/references.yaml``.
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "nonspatial": {
        "filename": "tabpfn-sdm-nonspatial.pt",
        "step1_epochs": 40,
        "step2_epochs": 100,
        "checkpoint_val_roc_auc": 0.747,
        "checkpoint_val_pr_auc": 0.261,
        "sha256": "e08fc4aa91cc4364490530c90418bc3152e60212a059852694b6190ba6f41f90",
    },
    "spatial": {
        "filename": "tabpfn-sdm-spatial.pt",
        "step1_epochs": 25,
        "step2_epochs": 50,
        "checkpoint_val_roc_auc": 0.653,
        "checkpoint_val_pr_auc": 0.144,
        "sha256": "eafaf07012dd3c18385a7a6b2e7303b775ba8e891d96e2a3c45bd722a63cb295",
    },
}

#: Seed used throughout the paper and the finetuning runs.
PUBLISHED_SEED = 32639


@dataclass
class CheckpointInfo:
    """Provenance of a downloaded checkpoint, recorded in the run manifest."""

    repo_id: str
    filename: str
    local_path: str
    revision: str = ""
    sha256: str = ""
    expected_sha256: str = ""
    base_model: str = BASE_MODEL
    tabpfn_version: str = ""

    @property
    def hash_matches(self) -> bool | None:
        """Whether the file matches the published hash (``None`` if unknown)."""
        if not self.expected_sha256 or not self.sha256:
            return None
        return self.sha256 == self.expected_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "local_path": self.local_path,
            "revision": self.revision,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "hash_matches": self.hash_matches,
            "base_model": self.base_model,
            "tabpfn_version": self.tabpfn_version,
            "license": "Prior Labs License v1.1",
        }


def download_checkpoint(
    variant: str = "nonspatial",
    *,
    repo_id: str = HF_REPO,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> CheckpointInfo:
    """Download a finetuned checkpoint from Hugging Face and record its hash.

    Raises :class:`~sdmbench.exceptions.LicenseError` -- which becomes a
    ``SKIPPED_LICENSE`` result -- when the repository is gated or requires
    authentication. sdmbench never works around such a gate.
    """
    if variant not in CHECKPOINTS:
        raise SdmbenchError(
            f"unknown checkpoint variant {variant!r}; expected one of {sorted(CHECKPOINTS)}"
        )
    spec = CHECKPOINTS[variant]
    hub = require("huggingface_hub")
    target_dir = ensure_dir(Path(cache_dir) if cache_dir else models_dir("tabpfn_sdm"))

    try:
        local_path = hub.hf_hub_download(
            repo_id=repo_id,
            filename=spec["filename"],
            revision=revision,
            cache_dir=str(target_dir),
        )
    except Exception as exc:  # noqa: BLE001 - hub raises a wide variety of types
        message = str(exc)
        lowered = message.lower()
        if any(k in lowered for k in ("gated", "401", "403", "authentication", "unauthorized")):
            raise LicenseError(
                f"Access to {repo_id}/{spec['filename']} requires authentication or licence "
                "acceptance.\n"
                f"  1. Open https://huggingface.co/{repo_id} and accept the Prior Labs "
                "License v1.1\n"
                "  2. Authenticate locally:  huggingface-cli login\n"
                "  sdmbench will not bypass this gate.\n"
                f"  Underlying error: {message}"
            ) from exc
        raise SdmbenchError(f"failed to download {spec['filename']} from {repo_id}: {message}") from exc

    resolved_revision = revision or ""
    # The hub stores blobs under <cache>/models--<org>--<name>/snapshots/<sha>/
    parts = Path(local_path).parts
    if "snapshots" in parts:
        resolved_revision = parts[parts.index("snapshots") + 1]

    return CheckpointInfo(
        repo_id=repo_id,
        filename=spec["filename"],
        local_path=str(local_path),
        revision=resolved_revision,
        sha256=hash_file(local_path),
        expected_sha256=str(spec.get("sha256", "")),
        tabpfn_version=package_version("tabpfn"),
    )


class _BaseTabPFNAdapter(ModelAdapter):
    """Shared plumbing for every TabPFN variant."""

    family = "foundation"
    supports_categorical = False
    requires_gpu = False  # runs on CPU, just slowly

    #: Defaults from the paper's "TabPFN Default" configuration (sec. 2.2.2).
    default_params: dict[str, Any] = {
        "n_estimators": 8,
        "softmax_temperature": 0.9,
        "balance_probabilities": False,
        "average_before_softmax": False,
    }

    def check_available(self) -> None:
        require("tabpfn", hint="TabPFN is licensed under the Prior Labs License v1.1.")
        super().check_available()

    def device_used(self) -> str:
        return resolve_device(self.hyperparameters.get("device", "auto"))

    def classifier_kwargs(self) -> dict[str, Any]:
        params = dict(self.default_params)
        params.update(
            {
                k: v
                for k, v in self.hyperparameters.items()
                if k not in {"seed", "device", "n_members", "scheme", "max_train_size", "combine",
                             "checkpoint", "variant", "revision"}
            }
        )
        params.setdefault("random_state", self.seed)
        params.setdefault("ignore_pretraining_limits", True)
        params["device"] = self.device_used()
        return params

    def build_classifier(self) -> Any:
        tabpfn = require("tabpfn")
        return tabpfn.TabPFNClassifier(**self.classifier_kwargs())

    def prepare_features(self, split):
        # TabPFN needs explicit categorical feature indices (paper sec. 2.1.2),
        # so keep them on the instance and hand over numeric arrays.
        self._categorical_indices = split.categorical_indices
        X_train, _, X_test, _ = split.as_arrays()
        return X_train, X_test

    def _fit_one(self, clf: Any, X: np.ndarray, y: np.ndarray) -> Any:
        cat = getattr(self, "_categorical_indices", None)
        if cat:
            # Older/newer tabpfn releases expose this differently; set the
            # attribute when the constructor did not accept it.
            try:
                clf.set_params(categorical_features_indices=list(cat))
            except (AttributeError, ValueError):
                setattr(clf, "categorical_features_indices", list(cat))
        clf.fit(X, y)
        return clf

    def version(self) -> str:
        return package_version("tabpfn")

    def effective_hyperparameters(self) -> dict[str, Any]:
        params = self.classifier_kwargs()
        params["seed"] = self.seed
        return params


@register_model("tabpfn-default", "tabpfn")
class TabPFNAdapter(_BaseTabPFNAdapter):
    """TabPFN with the paper's default configuration."""

    def fit(self, X, y, **kwargs: Any) -> TabPFNAdapter:
        self._fitted = self._fit_one(self.build_classifier(), np.asarray(X), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._fitted is None:
            raise SdmbenchError("TabPFN adapter was not fitted")
        return self._fitted.predict_proba(np.asarray(X))[:, 1]


@register_model("tabpfn-real")
class TabPFNRealAdapter(TabPFNAdapter):
    """TabPFN using the ``v2_5_real`` checkpoint (paper sec. 2.2.2)."""

    default_params = {
        **_BaseTabPFNAdapter.default_params,
        "model_path": "v2_5_real",
    }


@register_model("tabpfn-balanced")
class TabPFNBalancedAdapter(TabPFNAdapter):
    """TabPFN with balanced probabilities and logit averaging."""

    default_params = {
        "n_estimators": 8,
        "softmax_temperature": 0.9,
        "balance_probabilities": True,
        "average_before_softmax": True,
    }


@register_model("tabpfn-ss", "tabpfn-subsample-ensemble")
class TabPFNSubsampleEnsembleAdapter(_BaseTabPFNAdapter):
    """TabPFN-SS: the paper's ensemble class balancing (sec. 2.3).

    K balanced subsets, all presences retained in each, logits averaged across
    members. The paper's settings are K = 16, ``n_estimators = 16``,
    ``average_before_softmax = True``, ``balance_probabilities = True``.
    """

    default_params = {
        "n_estimators": PUBLISHED_K,
        "softmax_temperature": 0.9,
        "balance_probabilities": True,
        "average_before_softmax": True,
    }

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.ensemble = PresenceBackgroundEnsemble(
            n_members=int(hyperparameters.get("n_members", PUBLISHED_K)),
            scheme=hyperparameters.get("scheme", "balanced"),
            max_train_size=hyperparameters.get("max_train_size", 1500),
            combine=hyperparameters.get("combine", "logit"),
            seed=self.seed,
        )

    def run(self, split):
        from sdmbench.models.base import FitResult
        import time

        X_train, X_test = self.prepare_features(split)
        y_train = np.asarray(split.y_train, dtype=int)

        started = time.perf_counter()

        def fit_predict(X_member, y_member, X_eval):
            clf = self._fit_one(self.build_classifier(), X_member, y_member)
            return clf.predict_proba(X_eval)[:, 1]

        probabilities = self.ensemble.fit_predict(fit_predict, X_train, y_train, X_test)
        elapsed = time.perf_counter() - started

        return FitResult(
            probabilities=np.asarray(probabilities, dtype=float),
            # TabPFN's in-context learning does not separate fitting from
            # prediction: one forward pass does both. Splitting the total
            # arbitrarily would be a fabricated number, so it is attributed to
            # fit time and predict time is reported as 0.
            fit_seconds=elapsed,
            predict_seconds=0.0,
            device=self.device_used(),
            extra={"ensemble": self.ensemble.describe()},
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover - run() is the entry point
        raise NotImplementedError("TabPFN-SS fits per ensemble member inside run()")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError("TabPFN-SS predicts inside run()")

    def effective_hyperparameters(self) -> dict[str, Any]:
        return {**super().effective_hyperparameters(), "ensemble": self.ensemble.describe()}


@register_model("tabpfn-sdm", "tabpfn-finetuned")
class FinetunedTabPFNAdapter(TabPFNSubsampleEnsembleAdapter):
    """The released finetuned TabPFN-SDM checkpoint plus the subsample ensemble.

    This is the paper's best-performing configuration ("TabPFN Finetuned
    combines domain-specific finetuning with the subsample ensemble approach").
    """

    #: Which checkpoint to load; overridden by ``tabpfn-sdm-spatial``.
    checkpoint_variant = "nonspatial"

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.variant = hyperparameters.get("variant", self.checkpoint_variant)
        self.checkpoint_info: CheckpointInfo | None = None
        self._state_dict: Any = None

    def check_available(self) -> None:
        super().check_available()
        require("torch")
        require("huggingface_hub")

    def ensure_checkpoint(self) -> CheckpointInfo:
        """Download (once) and hash the checkpoint."""
        if self.checkpoint_info is None:
            local = self.hyperparameters.get("checkpoint")
            if local:
                path = Path(local)
                if not path.is_file():
                    raise SdmbenchError(f"checkpoint file not found: {path}")
                self.checkpoint_info = CheckpointInfo(
                    repo_id="local",
                    filename=path.name,
                    local_path=str(path),
                    sha256=hash_file(path),
                    expected_sha256=str(CHECKPOINTS[self.variant].get("sha256", "")),
                    tabpfn_version=package_version("tabpfn"),
                )
            else:
                self.checkpoint_info = download_checkpoint(
                    self.variant, revision=self.hyperparameters.get("revision")
                )
        return self.checkpoint_info

    def _load_state_dict(self) -> Any:
        """Read ``model_state_dict`` out of the checkpoint archive."""
        if self._state_dict is None:
            torch = require("torch")
            info = self.ensure_checkpoint()
            # weights_only=False is required: the archive also holds the config
            # and training history (model card, "Checkpoint Contents").
            payload = torch.load(info.local_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or "model_state_dict" not in payload:
                raise SdmbenchError(
                    f"{info.filename} does not contain a 'model_state_dict' key; "
                    f"found: {list(payload)[:10] if isinstance(payload, dict) else type(payload)}"
                )
            self._state_dict = payload["model_state_dict"]
            self._checkpoint_config = payload.get("config", {})
        return self._state_dict

    def build_classifier(self) -> Any:
        """Build a TabPFN classifier with the finetuned weights loaded.

        Follows the loading procedure documented in the model card:
        instantiate, ``_initialize_model_variables()``, then
        ``models_[0].load_state_dict(...)``.
        """
        clf = super().build_classifier()
        state_dict = self._load_state_dict()

        initialise = getattr(clf, "_initialize_model_variables", None)
        if initialise is None:
            raise SdmbenchError(
                "the installed tabpfn package does not expose "
                "TabPFNClassifier._initialize_model_variables(), which the TabPFN-SDM model "
                "card's loading procedure requires.\n"
                f"  installed tabpfn version: {package_version('tabpfn')}\n"
                "  sdmbench will not guess at an alternative loading path, because partially "
                "loaded weights would produce plausible but wrong results."
            )
        initialise()
        models = getattr(clf, "models_", None)
        if not models:
            raise SdmbenchError(
                "TabPFNClassifier.models_ is empty after initialisation; cannot load the "
                "finetuned weights."
            )
        models[0].load_state_dict(state_dict)
        models[0].eval()
        return clf

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        if self.checkpoint_info is not None:
            meta["checkpoint"] = self.checkpoint_info.to_dict()
        meta["checkpoint_variant"] = self.variant
        # Surface the model card's own validation numbers, clearly labelled so
        # they can never be mistaken for a reproduced benchmark result.
        meta["checkpoint_validation_metrics"] = {
            k: v for k, v in CHECKPOINTS[self.variant].items() if k.startswith("checkpoint_val")
        }
        return meta

    def version(self) -> str:
        info = self.checkpoint_info
        base = package_version("tabpfn")
        return f"{base}+{self.variant}@{info.revision[:8]}" if info and info.revision else base


@register_model("tabpfn-sdm-spatial", "tabpfn-finetuned-spatial")
class FinetunedTabPFNSpatialAdapter(FinetunedTabPFNAdapter):
    """The spatially finetuned checkpoint.

    The paper trained "separate finetuned models ... for non-spatial and
    spatial evaluation protocols" (sec. 2.4.2), so the spatial scenario must
    use this checkpoint rather than the non-spatial one.
    """

    checkpoint_variant = "spatial"
