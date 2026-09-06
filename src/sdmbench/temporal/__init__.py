"""Temporal encoders.

For modalities that arrive as sequences: annual climate series, seasonal NDVI,
palaeoclimate reconstructions back to the Last Glacial Maximum, future
projections.

The benchmark question: does temporal structure add anything over a summary?
A species' distribution may be shaped by the *variability* or the *trajectory*
of climate rather than its mean, but ecological time series are short and
noisy, so ``temporal-summary`` (mean, SD, trend, min, max) is a strong baseline
that the sequence models have to beat.

Input shape: ``(n_samples, timesteps, n_features)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sdmbench.core.modality import ModalityData, ModalityKind
from sdmbench.optional import require, resolve_device
from sdmbench.representations.base import (
    REPRESENTATIONS,
    BaseEncoder,
    EncodedModality,
    encoder_spec,
)

__all__ = [
    "TemporalSummaryEncoder",
    "RecurrentTemporalEncoder",
    "TemporalCNNEncoder",
    "TemporalTransformerEncoder",
]


def _as_sequence(modality: ModalityData) -> np.ndarray:
    array = np.asarray(modality.values, dtype=float)
    if array.ndim == 2:  # (n, t) single variable
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(
            f"sequence modality {modality.name!r} must be (n, t, f); got {array.shape}"
        )
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


@REPRESENTATIONS.register(
    "temporal-summary",
    "temporal-stats",
    spec=encoder_spec(
        accepts=("sequence",),
        description="Mean, SD, min, max, first, last and linear trend per variable.",
    ),
)
class TemporalSummaryEncoder(BaseEncoder):
    """Hand-crafted temporal summaries -- the baseline.

    Includes a linear trend term, because "is the climate here changing?" is
    often the ecologically relevant question and a plain mean discards it.
    """

    accepts = (ModalityKind.SEQUENCE,)
    trainable = False

    def _transform(self, modality: ModalityData) -> EncodedModality:
        sequence = _as_sequence(modality)
        n, timesteps, n_features = sequence.shape

        # Least-squares slope per series, computed in closed form.
        time = np.arange(timesteps, dtype=float)
        time_centred = time - time.mean()
        denominator = float((time_centred**2).sum()) or 1.0
        trend = (sequence * time_centred[None, :, None]).sum(axis=1) / denominator

        parts = [
            sequence.mean(axis=1),
            sequence.std(axis=1),
            sequence.min(axis=1),
            sequence.max(axis=1),
            sequence[:, 0, :],
            sequence[:, -1, :],
            trend,
        ]
        values = np.hstack(parts)
        labels = ("mean", "sd", "min", "max", "first", "last", "trend")
        names = [
            f"{modality.name}_v{feature}_{label}"
            for label in labels
            for feature in range(n_features)
        ]
        return EncodedModality(
            name=modality.name,
            values=values,
            feature_names=names,
            metadata={"timesteps": timesteps, "n_variables": n_features},
        )


class _TorchSequenceEncoder(BaseEncoder):
    """Shared self-supervised training loop for sequence encoders.

    Trained to reconstruct the sequence, so no labels are involved and the
    encoder can be fitted on training rows without any supervised leakage path.
    """

    accepts = (ModalityKind.SEQUENCE,)
    trainable = True

    defaults: dict[str, Any] = {
        "latent_dim": 32,
        "hidden_dim": 64,
        "epochs": 100,
        "lr": 1e-3,
        "batch_size": 128,
        "seed": 32639,
        "device": "auto",
    }

    def settings(self) -> dict[str, Any]:
        return {**self.defaults, **self.params}

    def _build(self, timesteps: int, n_features: int) -> Any:
        raise NotImplementedError

    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        torch = require("torch")
        settings = self.settings()
        sequence = _as_sequence(modality)
        n, timesteps, n_features = sequence.shape
        device = torch.device(resolve_device(str(settings["device"])))
        torch.manual_seed(int(settings["seed"]))

        model = self._build(timesteps, n_features).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]))
        tensor = torch.tensor(sequence, dtype=torch.float32, device=device)
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
        self._model = model
        self._device = device
        self._shape = (timesteps, n_features)

    def _transform(self, modality: ModalityData) -> EncodedModality:
        torch = require("torch")
        sequence = _as_sequence(modality)
        with torch.no_grad():
            codes = (
                self._model.encode(
                    torch.tensor(sequence, dtype=torch.float32, device=self._device)
                )
                .cpu()
                .numpy()
            )
        return EncodedModality(
            name=modality.name,
            values=codes,
            feature_names=[f"{modality.name}_t{i}" for i in range(codes.shape[1])],
            metadata={"timesteps": self._shape[0], "n_variables": self._shape[1]},
        )


@REPRESENTATIONS.register(
    "temporal-lstm",
    "lstm",
    "temporal-gru",
    spec=encoder_spec(
        accepts=("sequence",),
        description="Recurrent (LSTM/GRU) sequence encoder, trained by reconstruction.",
        requires=("torch",),
        extra="deep",
        trainable=True,
    ),
)
class RecurrentTemporalEncoder(_TorchSequenceEncoder):
    """LSTM or GRU encoder; the final hidden state is the representation."""

    defaults = {**_TorchSequenceEncoder.defaults, "cell": "lstm", "n_layers": 1}

    def _build(self, timesteps: int, n_features: int) -> Any:
        torch = require("torch")
        settings = self.settings()
        hidden, latent = int(settings["hidden_dim"]), int(settings["latent_dim"])
        cell = str(settings["cell"]).lower()
        rnn_cls = torch.nn.GRU if cell == "gru" else torch.nn.LSTM

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = rnn_cls(
                    n_features, hidden, num_layers=int(settings["n_layers"]), batch_first=True
                )
                self.to_latent = torch.nn.Linear(hidden, latent)
                self.decoder = torch.nn.Linear(latent, timesteps * n_features)
                self.shape = (timesteps, n_features)

            def encode(self, x):
                output, _ = self.rnn(x)
                return self.to_latent(output[:, -1, :])

            def forward(self, x):
                return self.decoder(self.encode(x)).view(-1, *self.shape)

        return Net()


@REPRESENTATIONS.register(
    "temporal-cnn",
    spec=encoder_spec(
        accepts=("sequence",),
        description="1-D convolutional sequence encoder.",
        requires=("torch",),
        extra="deep",
        trainable=True,
    ),
)
class TemporalCNNEncoder(_TorchSequenceEncoder):
    """Dilated 1-D CNN over time.

    Often a better fit than an RNN for short ecological series: fewer
    parameters, and it picks up local seasonal structure directly.
    """

    def _build(self, timesteps: int, n_features: int) -> Any:
        torch = require("torch")
        settings = self.settings()
        hidden, latent = int(settings["hidden_dim"]), int(settings["latent_dim"])

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = torch.nn.Sequential(
                    torch.nn.Conv1d(n_features, hidden, 3, padding=1), torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool1d(1),
                )
                self.to_latent = torch.nn.Linear(hidden, latent)
                self.decoder = torch.nn.Linear(latent, timesteps * n_features)
                self.shape = (timesteps, n_features)

            def encode(self, x):
                # (n, t, f) -> (n, f, t) for Conv1d
                return self.to_latent(self.body(x.transpose(1, 2)).squeeze(-1))

            def forward(self, x):
                return self.decoder(self.encode(x)).view(-1, *self.shape)

        return Net()


@REPRESENTATIONS.register(
    "temporal-transformer",
    spec=encoder_spec(
        accepts=("sequence",),
        description="Transformer encoder over time with a learned [CLS] token.",
        requires=("torch",),
        extra="deep",
        trainable=True,
        maturity="emerging",
    ),
)
class TemporalTransformerEncoder(_TorchSequenceEncoder):
    """Transformer over the time axis.

    The natural choice for palaeoclimate sequences, where the informative
    signal may be at an arbitrary distance in the past rather than in the most
    recent steps -- exactly the case recurrent models handle worst.
    """

    defaults = {**_TorchSequenceEncoder.defaults, "n_heads": 4, "n_layers": 2}

    def _build(self, timesteps: int, n_features: int) -> Any:
        torch = require("torch")
        settings = self.settings()
        latent = int(settings["latent_dim"])
        n_heads = int(settings["n_heads"])
        d_model = int(settings["hidden_dim"])
        if d_model % n_heads:
            d_model = max(n_heads, (d_model // n_heads) * n_heads)

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.project = torch.nn.Linear(n_features, d_model)
                self.position = torch.nn.Parameter(
                    torch.randn(1, timesteps + 1, d_model) * 0.02
                )
                self.cls = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    batch_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(
                    layer, num_layers=int(settings["n_layers"])
                )
                self.to_latent = torch.nn.Linear(d_model, latent)
                self.decoder = torch.nn.Linear(latent, timesteps * n_features)
                self.shape = (timesteps, n_features)

            def encode(self, x):
                tokens = self.project(x)
                cls = self.cls.expand(x.shape[0], -1, -1)
                sequence = torch.cat([cls, tokens], dim=1) + self.position[:, : tokens.shape[1] + 1]
                return self.to_latent(self.encoder(sequence)[:, 0])

            def forward(self, x):
                return self.decoder(self.encode(x)).view(-1, *self.shape)

        return Net()
