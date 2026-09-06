"""Learned tabular representations (optional, ``sdmbench[deep]``).

Autoencoder, VAE, masked-feature and contrastive encoders. All are
``trainable=True`` and are fitted on **training rows only** -- fitting a
representation on train+test is representation leakage, and it is the kind that
is hardest to spot afterwards because the downstream classifier looks
perfectly well-behaved.

These exist to test a specific claim in the SDM literature: that learned
representations do not help on the small, correlated, noisy tables typical of
species data. The benchmark should be able to confirm or refute that rather
than assume it.
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
    "TorchEncoderBase",
    "AutoencoderEncoder",
    "VAEEncoder",
    "ContrastiveEncoder",
    "MaskedFeatureEncoder",
]


class TorchEncoderBase(BaseEncoder):
    """Shared training loop for the self-supervised tabular encoders."""

    accepts = (ModalityKind.TABULAR, ModalityKind.EMBEDDING)
    trainable = True

    defaults: dict[str, Any] = {
        "latent_dim": 16,
        "hidden_dim": 64,
        "lr": 1e-3,
        "epochs": 200,
        "batch_size": 256,
        "seed": 32639,
        "device": "auto",
        "noise": 0.1,
    }

    def settings(self) -> dict[str, Any]:
        return {**self.defaults, **self.params}

    def build(self, n_features: int) -> Any:
        raise NotImplementedError

    def loss(self, model: Any, batch: Any) -> Any:
        raise NotImplementedError

    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        torch = require("torch")
        settings = self.settings()
        device = torch.device(resolve_device(str(settings["device"])))
        torch.manual_seed(int(settings["seed"]))

        frame = modality.as_frame()
        array = np.nan_to_num(frame.to_numpy(dtype=float))
        self._columns = list(frame.columns)
        self._device = device

        model = self.build(array.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]))
        tensor = torch.tensor(array, dtype=torch.float32, device=device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tensor),
            batch_size=int(settings["batch_size"]),
            shuffle=True,
        )

        model.train()
        history: list[float] = []
        for _epoch in range(int(settings["epochs"])):
            total = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                value = self.loss(model, batch)
                value.backward()
                optimizer.step()
                total += float(value.detach().cpu()) * len(batch)
            history.append(total / max(len(tensor), 1))
        model.eval()
        self._model = model
        self._history = history

    def _transform(self, modality: ModalityData) -> EncodedModality:
        torch = require("torch")
        frame = modality.as_frame().loc[:, self._columns]
        array = np.nan_to_num(frame.to_numpy(dtype=float))
        with torch.no_grad():
            codes = (
                self._model.encode(
                    torch.tensor(array, dtype=torch.float32, device=self._device)
                )
                .cpu()
                .numpy()
            )
        return EncodedModality(
            name=modality.name,
            values=codes,
            feature_names=[f"{modality.name}_z{i}" for i in range(codes.shape[1])],
            metadata={"final_train_loss": self._history[-1] if self._history else None},
        )


@REPRESENTATIONS.register(
    "autoencoder",
    "ae",
    spec=encoder_spec(
        accepts=("tabular", "embedding"),
        description="Denoising autoencoder latent code.",
        requires=("torch",),
        extra="deep",
        trainable=True,
    ),
)
class AutoencoderEncoder(TorchEncoderBase):
    """Denoising autoencoder; the bottleneck is the representation."""

    def build(self, n_features: int) -> Any:
        torch = require("torch")
        settings = self.settings()
        hidden, latent = int(settings["hidden_dim"]), int(settings["latent_dim"])

        class AE(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc = torch.nn.Sequential(
                    torch.nn.Linear(n_features, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, latent),
                )
                self.dec = torch.nn.Sequential(
                    torch.nn.Linear(latent, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, n_features),
                )

            def encode(self, x):
                return self.enc(x)

            def forward(self, x):
                return self.dec(self.enc(x))

        return AE()

    def loss(self, model: Any, batch: Any) -> Any:
        torch = require("torch")
        noise = float(self.settings()["noise"])
        corrupted = batch + noise * torch.randn_like(batch) if noise else batch
        return torch.nn.functional.mse_loss(model(corrupted), batch)


@REPRESENTATIONS.register(
    "vae",
    spec=encoder_spec(
        accepts=("tabular", "embedding"),
        description="Variational autoencoder; posterior mean is the representation.",
        requires=("torch",),
        extra="deep",
        trainable=True,
    ),
)
class VAEEncoder(TorchEncoderBase):
    """Variational autoencoder.

    The posterior *mean* is used as the representation, not a sample, so the
    encoding is deterministic and a re-run reproduces it exactly.
    """

    defaults = {**TorchEncoderBase.defaults, "beta": 1.0}

    def build(self, n_features: int) -> Any:
        torch = require("torch")
        settings = self.settings()
        hidden, latent = int(settings["hidden_dim"]), int(settings["latent_dim"])

        class VAE(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = torch.nn.Sequential(
                    torch.nn.Linear(n_features, hidden), torch.nn.ReLU()
                )
                self.mu = torch.nn.Linear(hidden, latent)
                self.logvar = torch.nn.Linear(hidden, latent)
                self.dec = torch.nn.Sequential(
                    torch.nn.Linear(latent, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, n_features),
                )

            def encode(self, x):
                return self.mu(self.body(x))

            def forward(self, x):
                h = self.body(x)
                mu, logvar = self.mu(h), self.logvar(h)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                return self.dec(z), mu, logvar

        return VAE()

    def loss(self, model: Any, batch: Any) -> Any:
        torch = require("torch")
        recon, mu, logvar = model(batch)
        n = batch.shape[0]
        reconstruction = torch.nn.functional.mse_loss(recon, batch, reduction="sum") / n
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
        return reconstruction + float(self.settings()["beta"]) * kld


@REPRESENTATIONS.register(
    "contrastive",
    spec=encoder_spec(
        accepts=("tabular", "embedding"),
        description="SimCLR-style contrastive encoder with noise augmentation.",
        requires=("torch",),
        extra="deep",
        trainable=True,
        maturity="emerging",
    ),
)
class ContrastiveEncoder(TorchEncoderBase):
    """Contrastive (InfoNCE) encoder.

    Augmentation is Gaussian noise, which is the honest choice for tabular
    ecological data: the image-style augmentations that make SimCLR work
    (crops, colour jitter) have no meaning for a vector of bioclim variables.
    """

    defaults = {**TorchEncoderBase.defaults, "temperature": 0.5, "noise": 0.15}

    def build(self, n_features: int) -> Any:
        torch = require("torch")
        settings = self.settings()
        hidden, latent = int(settings["hidden_dim"]), int(settings["latent_dim"])

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(n_features, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, latent),
                )
                self.projection = torch.nn.Sequential(
                    torch.nn.Linear(latent, latent), torch.nn.ReLU(),
                    torch.nn.Linear(latent, latent),
                )

            def encode(self, x):
                return self.backbone(x)

            def forward(self, x):
                return self.projection(self.backbone(x))

        return Net()

    def loss(self, model: Any, batch: Any) -> Any:
        torch = require("torch")
        settings = self.settings()
        noise, temperature = float(settings["noise"]), float(settings["temperature"])
        z1 = torch.nn.functional.normalize(model(batch + noise * torch.randn_like(batch)), dim=1)
        z2 = torch.nn.functional.normalize(model(batch + noise * torch.randn_like(batch)), dim=1)
        n = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        similarity = (z @ z.T) / temperature
        similarity.fill_diagonal_(float("-inf"))
        targets = torch.cat(
            [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)]
        )
        return torch.nn.functional.cross_entropy(similarity, targets)


@REPRESENTATIONS.register(
    "masked-feature",
    "masked",
    spec=encoder_spec(
        accepts=("tabular", "embedding"),
        description="Masked-feature modelling: reconstruct hidden predictors.",
        requires=("torch",),
        extra="deep",
        trainable=True,
        maturity="emerging",
    ),
)
class MaskedFeatureEncoder(TorchEncoderBase):
    """BERT-style masked feature modelling for tabular data.

    A fraction of predictors is zeroed and the network reconstructs them. Loss
    is computed on the masked entries only, so the model cannot score well by
    copying the visible inputs.
    """

    defaults = {**TorchEncoderBase.defaults, "mask_rate": 0.25}

    def build(self, n_features: int) -> Any:
        torch = require("torch")
        settings = self.settings()
        hidden, latent = int(settings["hidden_dim"]), int(settings["latent_dim"])

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc = torch.nn.Sequential(
                    torch.nn.Linear(n_features, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, latent),
                )
                self.dec = torch.nn.Sequential(
                    torch.nn.Linear(latent, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, n_features),
                )

            def encode(self, x):
                return self.enc(x)

            def forward(self, x):
                return self.dec(self.enc(x))

        return Net()

    def loss(self, model: Any, batch: Any) -> Any:
        torch = require("torch")
        rate = float(self.settings()["mask_rate"])
        mask = (torch.rand_like(batch) < rate).float()
        reconstruction = model(batch * (1.0 - mask))
        denominator = mask.sum().clamp(min=1.0)
        return (((reconstruction - batch) ** 2) * mask).sum() / denominator
