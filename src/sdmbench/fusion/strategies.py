"""Concrete fusion strategies.

Ordered from the cheap baseline to the expressive-but-hungry. The scientific
point of running all of them is that in ecological data the baseline often
wins, and a framework that only offers the sophisticated option cannot discover
that.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sdmbench.exceptions import DataError
from sdmbench.fusion.base import (
    FUSION_STRATEGIES,
    FusedFeatures,
    FusionStrategy,
    fusion_spec,
)
from sdmbench.optional import require, resolve_device
from sdmbench.representations.base import EncodedModality

__all__ = [
    "ConcatFusion",
    "EarlyFusion",
    "LateFusion",
    "WeightedFusion",
    "GatedFusion",
    "CrossAttentionFusion",
    "MixtureOfExpertsFusion",
]


def _stack(encoded: Mapping[str, EncodedModality], order: list[str]) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for name in order:
        modality = encoded[name]
        array = np.asarray(modality.values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        blocks.append(np.nan_to_num(array))
        names.extend(
            modality.feature_names
            if len(modality.feature_names) == array.shape[1]
            else [f"{name}_{i}" for i in range(array.shape[1])]
        )
    return np.hstack(blocks), names


@FUSION_STRATEGIES.register(
    "concat",
    "concatenation",
    spec=fusion_spec("Concatenate encoded modality vectors. The baseline."),
)
class ConcatFusion(FusionStrategy):
    """Plain concatenation.

    Keeps every feature separable, which is what makes downstream SHAP and
    permutation importance attributable back to a modality.
    """

    trainable = False

    def _transform(self, encoded: Mapping[str, EncodedModality]) -> FusedFeatures:
        order = self._ordered(encoded)
        values, names = _stack(encoded, order)
        return FusedFeatures(
            values=values,
            feature_names=names,
            metadata={"modalities": order, "dims": {n: encoded[n].dim for n in order}},
        )


@FUSION_STRATEGIES.register(
    "early",
    spec=fusion_spec(
        "Standardise per modality, then concatenate before a shared encoder.",
        trainable=True,
    ),
)
class EarlyFusion(FusionStrategy):
    """Per-modality standardisation, then concatenation.

    Without this, a 64-dimensional embedding with values in the hundreds
    dominates a 5-dimensional standardised climate block purely by scale.
    Statistics come from training rows only.
    """

    trainable = True

    def _fit(self, encoded: Mapping[str, EncodedModality], y: np.ndarray | None) -> None:
        self._stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self._ordered(encoded):
            array = np.nan_to_num(np.asarray(encoded[name].values, dtype=float))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            mean = array.mean(axis=0)
            std = array.std(axis=0, ddof=1) if len(array) > 1 else np.ones(array.shape[1])
            self._stats[name] = (mean, np.where(std > 0, std, 1.0))

    def _transform(self, encoded: Mapping[str, EncodedModality]) -> FusedFeatures:
        order = self._ordered(encoded)
        blocks, names = [], []
        for name in order:
            array = np.nan_to_num(np.asarray(encoded[name].values, dtype=float))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            mean, std = self._stats.get(
                name, (np.zeros(array.shape[1]), np.ones(array.shape[1]))
            )
            blocks.append((array - mean) / std)
            names.extend(f"{name}_z{i}" for i in range(array.shape[1]))
        return FusedFeatures(
            values=np.hstack(blocks), feature_names=names, metadata={"modalities": order}
        )


@FUSION_STRATEGIES.register(
    "late",
    "late-fusion",
    spec=fusion_spec(
        "One classifier per modality; average their predicted probabilities.",
        trainable=True,
    ),
)
class LateFusion(FusionStrategy):
    """Independent per-modality predictors, combined at the probability level.

    Produces probabilities directly rather than a feature vector, so
    :attr:`FusedFeatures.probabilities` is set and the downstream model is
    bypassed. Robust when one modality is uninformative, but it cannot capture
    interactions *between* modalities by construction -- which is exactly what
    comparing it against cross-attention measures.
    """

    trainable = True
    supervised = True

    def _fit(self, encoded: Mapping[str, EncodedModality], y: np.ndarray | None) -> None:
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y, dtype=int)
        self._models: dict[str, Any] = {}
        self._weights: dict[str, float] = {}
        base = self.params.get("base_estimator")

        for name in self._ordered(encoded):
            array = np.nan_to_num(np.asarray(encoded[name].values, dtype=float))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            model = (
                base()
                if callable(base)
                else LogisticRegression(max_iter=2000, class_weight="balanced")
            )
            try:
                model.fit(array, y)
            except Exception:  # noqa: BLE001 - a degenerate modality is dropped, not fatal
                continue
            self._models[name] = model
            # Weight each modality by its own training AUC, so an uninformative
            # modality contributes little. Training-only, so no leakage.
            try:
                from sklearn.metrics import roc_auc_score

                score = float(roc_auc_score(y, model.predict_proba(array)[:, 1]))
            except Exception:  # noqa: BLE001
                score = 0.5
            self._weights[name] = max(score - 0.5, 0.0)

        if not self._models:
            raise DataError("late fusion could not fit any modality-level model")
        total = sum(self._weights.values())
        if total <= 0:
            self._weights = {k: 1.0 / len(self._models) for k in self._models}
        else:
            self._weights = {k: v / total for k, v in self._weights.items()}

    def _transform(self, encoded: Mapping[str, EncodedModality]) -> FusedFeatures:
        columns, probabilities, names = [], np.zeros(0), []
        weighted_sum = None
        for name, model in self._models.items():
            if name not in encoded:
                continue
            array = np.nan_to_num(np.asarray(encoded[name].values, dtype=float))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            probability = model.predict_proba(array)[:, 1]
            columns.append(probability)
            names.append(f"{name}_p")
            contribution = self._weights.get(name, 0.0) * probability
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        if weighted_sum is None:
            raise DataError("late fusion has no usable modality at transform time")
        probabilities = weighted_sum / max(
            sum(self._weights.get(n, 0.0) for n in self._models if n in encoded), 1e-12
        )
        return FusedFeatures(
            values=np.column_stack(columns),
            feature_names=names,
            probabilities=probabilities,
            modality_importance=dict(self._weights),
            metadata={"combination": "auc_weighted_probability_average"},
        )

    def modality_importance(self) -> dict[str, float]:
        return dict(getattr(self, "_weights", {}))


@FUSION_STRATEGIES.register(
    "weighted",
    spec=fusion_spec(
        "Learn one scalar weight per modality from its univariate signal.",
        trainable=True,
    ),
)
class WeightedFusion(FusionStrategy):
    """Scale each modality block by a learned scalar.

    The weights are directly readable as "how much this modality mattered",
    which is the cheapest interpretable answer to the information-ablation
    question.
    """

    trainable = True
    supervised = True

    def _fit(self, encoded: Mapping[str, EncodedModality], y: np.ndarray | None) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        y = np.asarray(y, dtype=int)
        self._weights = {}
        for name in self._ordered(encoded):
            array = np.nan_to_num(np.asarray(encoded[name].values, dtype=float))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            try:
                model = LogisticRegression(max_iter=1000, class_weight="balanced")
                model.fit(array, y)
                auc = float(roc_auc_score(y, model.predict_proba(array)[:, 1]))
            except Exception:  # noqa: BLE001
                auc = 0.5
            self._weights[name] = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))

    def _transform(self, encoded: Mapping[str, EncodedModality]) -> FusedFeatures:
        order = self._ordered(encoded)
        blocks, names = [], []
        for name in order:
            array = np.nan_to_num(np.asarray(encoded[name].values, dtype=float))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            blocks.append(array * self._weights.get(name, 1.0))
            names.extend(f"{name}_w{i}" for i in range(array.shape[1]))
        return FusedFeatures(
            values=np.hstack(blocks),
            feature_names=names,
            modality_importance=dict(self._weights),
        )

    def modality_importance(self) -> dict[str, float]:
        return dict(getattr(self, "_weights", {}))


class _TorchFusion(FusionStrategy):
    """Shared training loop for the neural fusion strategies."""

    trainable = True
    supervised = True

    defaults: dict[str, Any] = {
        "hidden_dim": 64,
        "epochs": 150,
        "lr": 1e-3,
        "batch_size": 256,
        "device": "auto",
        "dropout": 0.2,
    }

    def settings(self) -> dict[str, Any]:
        return {**self.defaults, **self.params}

    def _build(self, dims: dict[str, int]) -> Any:
        raise NotImplementedError

    def _fit(self, encoded: Mapping[str, EncodedModality], y: np.ndarray | None) -> None:
        torch = require("torch")
        settings = self.settings()
        device = torch.device(resolve_device(str(settings["device"])))
        torch.manual_seed(self.seed)

        order = self._ordered(encoded)
        blocks = {n: _block(encoded[n]) for n in order}
        self._dims = {n: blocks[n].shape[1] for n in order}
        self._device = device

        model = self._build(self._dims).to(device)
        tensors = [torch.tensor(blocks[n], dtype=torch.float32, device=device) for n in order]
        labels = torch.tensor(np.asarray(y, dtype=int), dtype=torch.long, device=device)

        counts = np.bincount(np.asarray(y, dtype=int), minlength=2).astype(float)
        weights = torch.tensor(
            (counts.sum() / np.maximum(counts, 1.0)) / 2.0, dtype=torch.float32, device=device
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]))

        model.train()
        n = len(labels)
        batch_size = int(settings["batch_size"])
        for _epoch in range(int(settings["epochs"])):
            permutation = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                index = permutation[start : start + batch_size]
                optimizer.zero_grad()
                logits, _ = model([t[index] for t in tensors])
                loss = criterion(logits, labels[index])
                loss.backward()
                optimizer.step()
        model.eval()
        self._model = model
        self._order = order

    def _transform(self, encoded: Mapping[str, EncodedModality]) -> FusedFeatures:
        torch = require("torch")
        order = [n for n in self._order if n in encoded]
        if len(order) != len(self._order):
            missing = set(self._order) - set(order)
            raise DataError(
                f"fusion was fitted with modalities {self._order} but {sorted(missing)} "
                "are absent at transform time. Re-fit for an ablation rather than "
                "silently dropping inputs."
            )
        tensors = [
            torch.tensor(_block(encoded[n]), dtype=torch.float32, device=self._device)
            for n in order
        ]
        with torch.no_grad():
            logits, extras = self._model(tensors)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            representation = extras["representation"].cpu().numpy()
            importance = extras.get("importance")
            importance_map = (
                {
                    n: float(v)
                    for n, v in zip(order, importance.mean(dim=0).cpu().numpy())
                }
                if importance is not None
                else {}
            )
        self._importance = importance_map
        return FusedFeatures(
            values=representation,
            feature_names=[f"fused_{i}" for i in range(representation.shape[1])],
            probabilities=probabilities,
            modality_importance=importance_map,
            metadata={"modalities": order},
        )

    def modality_importance(self) -> dict[str, float]:
        return dict(getattr(self, "_importance", {}))


def _block(modality: EncodedModality) -> np.ndarray:
    array = np.nan_to_num(np.asarray(modality.values, dtype=float))
    return array.reshape(-1, 1) if array.ndim == 1 else array


@FUSION_STRATEGIES.register(
    "gated",
    spec=fusion_spec(
        "Per-observation modality gates: the network decides what to rely on where.",
        requires=("torch",),
        extra="deep",
        trainable=True,
        maturity="emerging",
    ),
)
class GatedFusion(_TorchFusion):
    """Learn a gate per modality **per observation**.

    Ecologically motivated: the informative modality is not the same everywhere.
    The mean gate value per modality is reported as its importance.
    """

    def _build(self, dims: dict[str, int]) -> Any:
        torch = require("torch")
        settings = self.settings()
        hidden = int(settings["hidden_dim"])
        names = list(dims)

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projections = torch.nn.ModuleList(
                    [torch.nn.Linear(dims[n], hidden) for n in names]
                )
                self.gate = torch.nn.Sequential(
                    torch.nn.Linear(sum(dims.values()), hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, len(names)), torch.nn.Sigmoid(),
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Dropout(float(settings["dropout"])),
                    torch.nn.Linear(hidden, 2),
                )

            def forward(self, blocks):
                projected = torch.stack(
                    [torch.relu(p(b)) for p, b in zip(self.projections, blocks)], dim=1
                )
                gates = self.gate(torch.cat(blocks, dim=1))
                representation = (projected * gates.unsqueeze(-1)).sum(dim=1)
                return self.head(representation), {
                    "representation": representation,
                    "importance": gates,
                }

        return Net()


@FUSION_STRATEGIES.register(
    "cross-attention",
    "cross_attention",
    "attention",
    spec=fusion_spec(
        "Treat each modality as a token and attend across them.",
        requires=("torch",),
        extra="deep",
        trainable=True,
        maturity="emerging",
    ),
)
class CrossAttentionFusion(_TorchFusion):
    """Multi-head attention over modality tokens.

    Each modality is projected to a shared token, a ``[CLS]`` token attends
    over them, and its output is the fused representation. Attention weights
    are reported as importance -- with the standard caveat that attention is
    suggestive of what the model used, not proof of it.
    """

    defaults = {**_TorchFusion.defaults, "n_heads": 4}

    def _build(self, dims: dict[str, int]) -> Any:
        torch = require("torch")
        settings = self.settings()
        names = list(dims)
        d_model = int(settings["hidden_dim"])
        n_heads = int(settings["n_heads"])
        if d_model % n_heads:
            d_model = max(n_heads, (d_model // n_heads) * n_heads)

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projections = torch.nn.ModuleList(
                    [torch.nn.Linear(dims[n], d_model) for n in names]
                )
                self.cls = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                self.attention = torch.nn.MultiheadAttention(
                    d_model, n_heads, dropout=float(settings["dropout"]), batch_first=True
                )
                self.norm = torch.nn.LayerNorm(d_model)
                self.head = torch.nn.Linear(d_model, 2)

            def forward(self, blocks):
                tokens = torch.stack(
                    [p(b) for p, b in zip(self.projections, blocks)], dim=1
                )
                cls = self.cls.expand(tokens.shape[0], -1, -1)
                attended, weights = self.attention(cls, tokens, tokens, need_weights=True)
                representation = self.norm(attended.squeeze(1))
                return self.head(representation), {
                    "representation": representation,
                    # (batch, 1, n_modalities) -> (batch, n_modalities)
                    "importance": weights.squeeze(1),
                }

        return Net()


@FUSION_STRATEGIES.register(
    "moe",
    "mixture-of-experts",
    spec=fusion_spec(
        "Mixture of experts with a learned router over modalities.",
        requires=("torch",),
        extra="deep",
        trainable=True,
        maturity="experimental",
    ),
)
class MixtureOfExpertsFusion(_TorchFusion):
    """One expert per modality plus a learned router.

    Differs from gated fusion in that each expert produces its own
    representation and the router mixes *outputs* rather than scaling inputs.
    The router weights say which expert handled which observation.
    """

    defaults = {**_TorchFusion.defaults, "n_experts_per_modality": 1}

    def _build(self, dims: dict[str, int]) -> Any:
        torch = require("torch")
        settings = self.settings()
        names = list(dims)
        hidden = int(settings["hidden_dim"])

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(dims[n], hidden), torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden),
                        )
                        for n in names
                    ]
                )
                self.router = torch.nn.Sequential(
                    torch.nn.Linear(sum(dims.values()), len(names)),
                    torch.nn.Softmax(dim=1),
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Dropout(float(settings["dropout"])),
                    torch.nn.Linear(hidden, 2),
                )

            def forward(self, blocks):
                outputs = torch.stack(
                    [e(b) for e, b in zip(self.experts, blocks)], dim=1
                )
                weights = self.router(torch.cat(blocks, dim=1))
                representation = (outputs * weights.unsqueeze(-1)).sum(dim=1)
                return self.head(representation), {
                    "representation": representation,
                    "importance": weights,
                }

        return Net()
