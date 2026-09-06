"""Raster-patch representations, including multi-scale encoding.

The scale question
------------------
A species may respond to topography at 1 km (does this slope drain?), at 5 km
(is this a valley system?) and at 25 km (is this a mountain range?) at once. A
point sample of elevation answers none of those. So terrain, land cover and
imagery are extracted as **patches**, and a patch has a scale.

The multi-scale encoder makes scale a benchmark dimension rather than an
implicit choice::

    location
       |
    1 km patch  -> encoder --.
    5 km patch  -> encoder --+-> fusion -> vector
    25 km patch -> encoder --'

with the per-scale encoder either shared (fewer parameters, scale-invariant
features) or separate (each scale free to specialise), and the combination by
concatenation, pooling, or attention over scales. Attention is the interesting
one: the learned weights say *which scale this species actually responds to*,
which is an ecological result, not just an implementation detail.

Input shape: ``(n, c, h, w)`` single-scale, or ``(n, s, c, h, w)`` multi-scale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sdmbench.core.modality import ModalityData, ModalityKind
from sdmbench.optional import package_version, require, resolve_device
from sdmbench.representations.base import (
    REPRESENTATIONS,
    BaseEncoder,
    EncodedModality,
    encoder_spec,
)

__all__ = [
    "PatchStatisticsEncoder",
    "CNNPatchEncoder",
    "MultiScaleCNNEncoder",
    "VisionBackboneEncoder",
]


def _as_patches(modality: ModalityData) -> np.ndarray:
    """Normalise a raster modality to ``(n, s, c, h, w)``."""
    array = np.asarray(modality.values, dtype=float)
    if array.ndim == 4:  # (n, c, h, w) -> single scale
        array = array[:, None, ...]
    if array.ndim != 5:
        raise ValueError(
            f"raster modality {modality.name!r} must be (n, c, h, w) or (n, s, c, h, w); "
            f"got shape {array.shape}"
        )
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


@REPRESENTATIONS.register(
    "patch-statistics",
    "patch-stats",
    spec=encoder_spec(
        accepts=("raster",),
        description="Summarise each patch by mean/SD/min/max/median per channel.",
    ),
)
class PatchStatisticsEncoder(BaseEncoder):
    """Hand-crafted patch summaries -- the baseline a CNN must beat.

    Cheap, interpretable, and frequently competitive. Including it keeps the
    comparison honest: if a ResNet does not beat five summary statistics, that
    is the finding.
    """

    accepts = (ModalityKind.RASTER,)
    trainable = False

    def _transform(self, modality: ModalityData) -> EncodedModality:
        patches = _as_patches(modality)
        n, n_scales, n_channels = patches.shape[:3]
        flat = patches.reshape(n, n_scales, n_channels, -1)

        stats = np.concatenate(
            [
                flat.mean(axis=3),
                flat.std(axis=3),
                flat.min(axis=3),
                flat.max(axis=3),
                np.median(flat, axis=3),
            ],
            axis=2,
        ).reshape(n, -1)

        scale_labels = modality.scales or [f"s{i}" for i in range(n_scales)]
        names = [
            f"{modality.name}_{scale}_c{channel}_{stat}"
            for scale in scale_labels
            for stat in ("mean", "sd", "min", "max", "median")
            for channel in range(n_channels)
        ]
        return EncodedModality(
            name=modality.name,
            values=stats,
            feature_names=names[: stats.shape[1]],
            metadata={"n_scales": n_scales, "n_channels": n_channels},
        )


class _TorchPatchEncoder(BaseEncoder):
    """Shared machinery for the torch-based patch encoders."""

    accepts = (ModalityKind.RASTER,)
    trainable = True

    defaults: dict[str, Any] = {
        "embedding_dim": 64,
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 64,
        "seed": 32639,
        "device": "auto",
        "shared_encoder": True,
        "scale_fusion": "concat",
    }

    def settings(self) -> dict[str, Any]:
        return {**self.defaults, **self.params}

    def _device(self) -> Any:
        torch = require("torch")
        return torch.device(resolve_device(str(self.settings()["device"])))


@REPRESENTATIONS.register(
    "cnn",
    "raster-cnn",
    spec=encoder_spec(
        accepts=("raster",),
        description="Small convolutional encoder trained on patches (self-supervised).",
        requires=("torch",),
        extra="deep",
        trainable=True,
    ),
)
class CNNPatchEncoder(_TorchPatchEncoder):
    """A small CNN trained as a patch autoencoder, used as a feature extractor.

    Trained self-supervised (reconstruction) rather than supervised, because
    with 30 presences a supervised CNN memorises. The encoder is fitted on
    training patches only.
    """

    def _build(self, n_channels: int, size: int) -> Any:
        torch = require("torch")
        dim = int(self.settings()["embedding_dim"])

        class ConvEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = torch.nn.Sequential(
                    torch.nn.Conv2d(n_channels, 32, 3, padding=1), torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d(1),
                )
                self.head = torch.nn.Linear(64, dim)
                self.decoder = torch.nn.Linear(dim, n_channels * size * size)
                self.shape = (n_channels, size, size)

            def encode(self, x):
                return self.head(self.body(x).flatten(1))

            def forward(self, x):
                return self.decoder(self.encode(x)).view(-1, *self.shape)

        return ConvEncoder()

    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        torch = require("torch")
        settings = self.settings()
        patches = _as_patches(modality)
        n, n_scales, n_channels, height, _ = patches.shape
        device = self._device()
        torch.manual_seed(int(settings["seed"]))

        shared = bool(settings["shared_encoder"])
        n_models = 1 if shared else n_scales
        self._models = [self._build(n_channels, height).to(device) for _ in range(n_models)]
        self._shared = shared
        self._n_scales = n_scales
        self._device_obj = device

        for index, model in enumerate(self._models):
            # A shared encoder sees every scale; separate encoders see one each.
            source = patches.reshape(-1, n_channels, height, height) if shared else patches[:, index]
            tensor = torch.tensor(source, dtype=torch.float32, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]))
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(tensor),
                batch_size=int(settings["batch_size"]),
                shuffle=True,
            )
            model.train()
            for _epoch in range(int(settings["epochs"])):
                for (batch,) in loader:
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(batch), batch)
                    loss.backward()
                    optimizer.step()
            model.eval()

    def _transform(self, modality: ModalityData) -> EncodedModality:
        torch = require("torch")
        patches = _as_patches(modality)
        n, n_scales, n_channels, height, _ = patches.shape
        per_scale: list[np.ndarray] = []
        with torch.no_grad():
            for index in range(n_scales):
                model = self._models[0] if self._shared else self._models[index]
                tensor = torch.tensor(
                    patches[:, index], dtype=torch.float32, device=self._device_obj
                )
                per_scale.append(model.encode(tensor).cpu().numpy())

        values, names = _combine_scales(
            per_scale,
            mode=str(self.settings()["scale_fusion"]),
            labels=modality.scales or [f"s{i}" for i in range(n_scales)],
            prefix=modality.name,
        )
        return EncodedModality(
            name=modality.name,
            values=values,
            feature_names=names,
            metadata={
                "n_scales": n_scales,
                "shared_encoder": self._shared,
                "scale_fusion": self.settings()["scale_fusion"],
            },
        )


@REPRESENTATIONS.register(
    "multiscale-cnn",
    "multiscale",
    spec=encoder_spec(
        accepts=("raster",),
        description="Multi-scale CNN with concat/pool/attention fusion across scales.",
        requires=("torch",),
        extra="deep",
        trainable=True,
        maturity="emerging",
    ),
)
class MultiScaleCNNEncoder(CNNPatchEncoder):
    """Multi-scale patch encoding with attention over scales by default.

    The attention weights are recorded in the encoded metadata, because
    "terrain matters at 5 km for this species" is an ecological finding worth
    reporting, not just an internal activation.
    """

    defaults = {**CNNPatchEncoder.defaults, "scale_fusion": "attention"}

    def _transform(self, modality: ModalityData) -> EncodedModality:
        encoded = super()._transform(modality)
        encoded.metadata["scale_labels"] = list(modality.scales)
        return encoded


@REPRESENTATIONS.register(
    "vision-backbone",
    "resnet18",
    "vit",
    spec=encoder_spec(
        accepts=("raster",),
        description="Frozen pretrained vision backbone (torchvision) as a patch encoder.",
        requires=("torch", "torchvision"),
        extra="deep",
        pretraining_source="torchvision ImageNet weights",
        maturity="emerging",
    ),
)
class VisionBackboneEncoder(_TorchPatchEncoder):
    """A frozen torchvision backbone applied to raster patches.

    Two caveats, stated because they affect interpretation rather than
    execution:

    1. ImageNet weights were learned on 3-channel photographs. Environmental
       rasters are neither. The adapter maps channels to 3 by selection or
       repetition, and this is a real domain gap -- the encoder is included so
       the gap can be *measured* against a purpose-built alternative, not
       because it is expected to win.
    2. The backbone is frozen, so nothing is fitted here and no leakage is
       possible; ``fitted_on`` is ``"pretrained"``.
    """

    trainable = False
    pretraining_source = "torchvision ImageNet weights"

    defaults = {**_TorchPatchEncoder.defaults, "architecture": "resnet18", "scale_fusion": "concat"}

    def _backbone(self) -> Any:
        torch = require("torch")
        torchvision = require("torchvision")
        name = str(self.settings()["architecture"])
        factory = getattr(torchvision.models, name, None)
        if factory is None:
            raise ValueError(
                f"torchvision has no model {name!r} "
                f"(torchvision {package_version('torchvision')})"
            )
        model = factory(weights="DEFAULT")
        # Strip the classifier: we want features, not ImageNet classes.
        if hasattr(model, "fc"):
            model.fc = torch.nn.Identity()
        elif hasattr(model, "heads"):
            model.heads = torch.nn.Identity()
        elif hasattr(model, "classifier"):
            model.classifier = torch.nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def _transform(self, modality: ModalityData) -> EncodedModality:
        torch = require("torch")
        patches = _as_patches(modality)
        n, n_scales, n_channels, height, width = patches.shape
        device = self._device()
        model = self._backbone().to(device)

        per_scale: list[np.ndarray] = []
        with torch.no_grad():
            for index in range(n_scales):
                block = patches[:, index]
                block = _to_three_channels(block)
                tensor = torch.tensor(block, dtype=torch.float32, device=device)
                if height < 32 or width < 32:
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=(32, 32), mode="bilinear", align_corners=False
                    )
                per_scale.append(model(tensor).cpu().numpy())

        values, names = _combine_scales(
            per_scale,
            mode=str(self.settings()["scale_fusion"]),
            labels=modality.scales or [f"s{i}" for i in range(n_scales)],
            prefix=modality.name,
        )
        return EncodedModality(
            name=modality.name,
            values=values,
            feature_names=names,
            metadata={
                "architecture": self.settings()["architecture"],
                "frozen": True,
                "domain_gap_note": (
                    "ImageNet-pretrained weights applied to environmental rasters; "
                    "interpret with care."
                ),
            },
        )


def _to_three_channels(block: np.ndarray) -> np.ndarray:
    """Coerce an ``(n, c, h, w)`` block to 3 channels for an RGB backbone."""
    n_channels = block.shape[1]
    if n_channels == 3:
        return block
    if n_channels == 1:
        return np.repeat(block, 3, axis=1)
    if n_channels > 3:
        return block[:, :3]
    padding = np.repeat(block[:, -1:], 3 - n_channels, axis=1)
    return np.concatenate([block, padding], axis=1)


def _combine_scales(
    per_scale: list[np.ndarray],
    *,
    mode: str,
    labels: list[str],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    """Combine per-scale embeddings into one vector.

    ``concat`` keeps every scale separable (best for ablation); ``mean``/``max``
    pool them; ``attention`` learns a softmax weighting over scales from each
    scale's own embedding norm, so the weights are interpretable as "how much
    this scale contributed".
    """
    if len(per_scale) == 1:
        vectors = per_scale[0]
        return vectors, [f"{prefix}_{labels[0]}_e{i}" for i in range(vectors.shape[1])]

    if mode == "concat":
        stacked = np.hstack(per_scale)
        names = [
            f"{prefix}_{labels[s]}_e{i}"
            for s in range(len(per_scale))
            for i in range(per_scale[s].shape[1])
        ]
        return stacked, names

    cube = np.stack(per_scale, axis=1)  # (n, scales, dim)
    if mode == "mean":
        pooled = cube.mean(axis=1)
    elif mode == "max":
        pooled = cube.max(axis=1)
    elif mode == "attention":
        # Softmax over per-scale L2 norms: a parameter-free attention that
        # needs no training pass and stays deterministic.
        norms = np.linalg.norm(cube, axis=2)
        norms = norms - norms.max(axis=1, keepdims=True)
        weights = np.exp(norms)
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
        pooled = (cube * weights[:, :, None]).sum(axis=1)
    else:
        raise ValueError(
            f"unknown scale_fusion {mode!r}; use concat, mean, max or attention"
        )
    return pooled, [f"{prefix}_pooled_e{i}" for i in range(pooled.shape[1])]
