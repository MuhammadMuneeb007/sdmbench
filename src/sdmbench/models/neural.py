"""Representation-learning adapters (optional, ``sdmbench[deep]``).

These are exploratory extensions rather than part of the published protocol.
They exist to test the paper's framing claim -- that conventional deep learning
has not broken through on SDM data, whereas in-context learning might -- under
the same benchmark rules, so that any comparison is like-for-like.

Everything here builds on ``torch`` primitives (``nn.Linear``, ``nn.LayerNorm``,
``nn.TransformerEncoderLayer``); no optimiser or attention mechanism is
reimplemented.

Unsupervised variants (autoencoder, VAE, contrastive encoder) are two-stage:
learn a representation from the **training** covariates only, then fit a linear
probe on the embedded training data. Fitting the encoder on train+test would be
representation leakage, which the auditor treats as fatal.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from sdmbench.data.base import PreparedSplit
from sdmbench.models.base import FitResult, ModelAdapter, register_model
from sdmbench.optional import package_version, require, resolve_device

__all__ = [
    "TorchTabularAdapter",
    "MLPAdapter",
    "ResidualMLPAdapter",
    "WideAndDeepAdapter",
    "FTTransformerAdapter",
    "AutoencoderAdapter",
    "VAEAdapter",
    "ContrastiveEncoderAdapter",
]


class TorchTabularAdapter(ModelAdapter):
    """Base class for supervised torch models on tabular data."""

    family = "neural"
    supports_categorical = False

    default_params: ClassVar[dict[str, Any]] = {
        "hidden_dim": 128,
        "n_layers": 3,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 200,
        "batch_size": 256,
        "patience": 25,
    }

    def check_available(self) -> None:
        require("torch")
        super().check_available()

    def params(self) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update({k: v for k, v in self.hyperparameters.items() if k != "seed"})
        return merged

    def device_used(self) -> str:
        return resolve_device(self.hyperparameters.get("device", "auto"))

    def build_module(self, n_features: int) -> Any:
        raise NotImplementedError

    def run(self, split: PreparedSplit) -> FitResult:
        torch = require("torch")
        X_train, y_train, X_test, _ = split.as_arrays()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        p = self.params()
        device = torch.device(self.device_used())
        torch.manual_seed(self.seed)

        model = self.build_module(X_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        counts = np.bincount(y_train, minlength=2).astype(float)
        weights = torch.tensor(
            (counts.sum() / np.maximum(counts, 1.0)) / 2.0, dtype=torch.float32, device=device
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        x = torch.tensor(X_train, dtype=torch.float32, device=device)
        y = torch.tensor(y_train, dtype=torch.long, device=device)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(p["batch_size"]), shuffle=True, drop_last=False
        )

        started = time.perf_counter()
        model.train()
        best_loss, best_state, waited = float("inf"), None, 0
        for _epoch in range(int(p["epochs"])):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(xb)
            epoch_loss /= max(len(dataset), 1)
            # Early stopping monitors TRAINING loss; the independent test set
            # never participates in the stopping decision.
            if epoch_loss < best_loss - 1e-5:
                best_loss, waited = epoch_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= int(p["patience"]):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32, device=device))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        predict_seconds = time.perf_counter() - started

        return FitResult(
            probabilities=np.asarray(probabilities, dtype=float),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            device=str(device),
            extra={"final_train_loss": best_loss},
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("torch adapters fit inside run()")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError("torch adapters predict inside run()")

    def effective_hyperparameters(self) -> dict[str, Any]:
        return {**self.params(), "seed": self.seed}

    def version(self) -> str:
        return package_version("torch")


@register_model("mlp")
class MLPAdapter(TorchTabularAdapter):
    """Plain feed-forward network."""

    def build_module(self, n_features: int) -> Any:
        torch = require("torch")
        p = self.params()
        layers: list[Any] = []
        in_dim = n_features
        for _ in range(int(p["n_layers"])):
            layers += [
                torch.nn.Linear(in_dim, int(p["hidden_dim"])),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(p["dropout"])),
            ]
            in_dim = int(p["hidden_dim"])
        layers.append(torch.nn.Linear(in_dim, 2))
        return torch.nn.Sequential(*layers)


@register_model("residual-mlp", "resmlp")
class ResidualMLPAdapter(TorchTabularAdapter):
    """MLP with residual blocks and layer normalisation."""

    def build_module(self, n_features: int) -> Any:
        torch = require("torch")
        p = self.params()
        hidden = int(p["hidden_dim"])
        dropout = float(p["dropout"])
        n_blocks = int(p["n_layers"])

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = torch.nn.LayerNorm(hidden)
                self.fc1 = torch.nn.Linear(hidden, hidden * 2)
                self.fc2 = torch.nn.Linear(hidden * 2, hidden)
                self.drop = torch.nn.Dropout(dropout)

            def forward(self, x):
                h = torch.nn.functional.relu(self.fc1(self.norm(x)))
                return x + self.drop(self.fc2(h))

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(n_features, hidden)
                self.blocks = torch.nn.ModuleList([Block() for _ in range(n_blocks)])
                self.head = torch.nn.Linear(hidden, 2)

            def forward(self, x):
                x = self.stem(x)
                for block in self.blocks:
                    x = block(x)
                return self.head(x)

        return Net()


@register_model("wide-and-deep", "wide_and_deep")
class WideAndDeepAdapter(TorchTabularAdapter):
    """Wide linear path summed with a deep path (Cheng et al. 2016)."""

    def build_module(self, n_features: int) -> Any:
        torch = require("torch")
        p = self.params()
        hidden = int(p["hidden_dim"])
        dropout = float(p["dropout"])
        n_layers = int(p["n_layers"])

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wide = torch.nn.Linear(n_features, 2)
                deep: list[Any] = []
                in_dim = n_features
                for _ in range(n_layers):
                    deep += [
                        torch.nn.Linear(in_dim, hidden),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                    ]
                    in_dim = hidden
                deep.append(torch.nn.Linear(in_dim, 2))
                self.deep = torch.nn.Sequential(*deep)

            def forward(self, x):
                return self.wide(x) + self.deep(x)

        return Net()


@register_model("ft-transformer", "tabular-transformer")
class FTTransformerAdapter(TorchTabularAdapter):
    """Feature-tokeniser transformer (Gorishniy et al. 2021), simplified.

    Each scalar feature becomes a token via a learned per-feature embedding; a
    ``[CLS]`` token is prepended and classified after stock
    ``nn.TransformerEncoderLayer`` blocks.
    """

    default_params: ClassVar[dict[str, Any]] = {
        **TorchTabularAdapter.default_params,
        "hidden_dim": 64,
        "n_layers": 3,
        "n_heads": 8,
    }

    def build_module(self, n_features: int) -> Any:
        torch = require("torch")
        p = self.params()
        d_token = int(p["hidden_dim"])
        n_heads = int(p["n_heads"])
        if d_token % n_heads != 0:
            d_token = max(n_heads, (d_token // n_heads) * n_heads)
        n_layers = int(p["n_layers"])
        dropout = float(p["dropout"])

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n_features, d_token) * 0.02)
                self.bias = torch.nn.Parameter(torch.zeros(n_features, d_token))
                self.cls = torch.nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=d_token,
                    nhead=n_heads,
                    dim_feedforward=d_token * 2,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
                self.norm = torch.nn.LayerNorm(d_token)
                self.head = torch.nn.Linear(d_token, 2)

            def forward(self, x):
                tokens = x.unsqueeze(-1) * self.weight + self.bias
                cls = self.cls.expand(x.shape[0], -1, -1)
                encoded = self.encoder(torch.cat([cls, tokens], dim=1))
                return self.head(self.norm(encoded[:, 0]))

        return Net()


# ---------------------------------------------------------------------------
# Unsupervised representation learning + linear probe
# ---------------------------------------------------------------------------


class _RepresentationAdapter(ModelAdapter):
    """Learn an encoder on training covariates, then fit a probe on the codes.

    The encoder is fitted on the **training** design matrix only. Fitting it on
    train+test would leak the test distribution into the representation, which
    the leakage auditor treats as fatal.
    """

    family = "neural"
    supports_categorical = False

    default_params: ClassVar[dict[str, Any]] = {
        "latent_dim": 16,
        "hidden_dim": 64,
        "lr": 1e-3,
        "epochs": 200,
        "batch_size": 256,
        "beta": 1.0,
        "temperature": 0.5,
        "noise": 0.1,
    }

    def check_available(self) -> None:
        require("torch")
        super().check_available()

    def params(self) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update({k: v for k, v in self.hyperparameters.items() if k != "seed"})
        return merged

    def device_used(self) -> str:
        return resolve_device(self.hyperparameters.get("device", "auto"))

    def build_encoder(self, n_features: int) -> Any:
        raise NotImplementedError

    def encoder_loss(self, model: Any, xb: Any) -> Any:
        raise NotImplementedError

    def run(self, split: PreparedSplit) -> FitResult:
        torch = require("torch")
        X_train, y_train, X_test, _ = split.as_arrays()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        p = self.params()
        device = torch.device(self.device_used())
        torch.manual_seed(self.seed)

        model = self.build_encoder(X_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]))
        x = torch.tensor(X_train, dtype=torch.float32, device=device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x), batch_size=int(p["batch_size"]), shuffle=True
        )

        started = time.perf_counter()
        model.train()
        for _epoch in range(int(p["epochs"])):
            for (xb,) in loader:
                optimizer.zero_grad()
                loss = self.encoder_loss(model, xb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            z_train = model.encode(x).cpu().numpy()
            z_test = model.encode(
                torch.tensor(X_test, dtype=torch.float32, device=device)
            ).cpu().numpy()

        from sklearn.linear_model import LogisticRegression

        probe = LogisticRegression(max_iter=2000, class_weight="balanced")
        probe.fit(z_train, y_train)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        probabilities = probe.predict_proba(z_test)[:, 1]
        predict_seconds = time.perf_counter() - started

        return FitResult(
            probabilities=np.asarray(probabilities, dtype=float),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            device=str(device),
            extra={"latent_dim": int(p["latent_dim"]), "probe": "logistic_regression"},
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("representation adapters fit inside run()")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError("representation adapters predict inside run()")

    def effective_hyperparameters(self) -> dict[str, Any]:
        return {**self.params(), "seed": self.seed}

    def version(self) -> str:
        return package_version("torch")


@register_model("autoencoder", "ae")
class AutoencoderAdapter(_RepresentationAdapter):
    """Denoising autoencoder followed by a logistic probe."""

    def build_encoder(self, n_features: int) -> Any:
        torch = require("torch")
        p = self.params()
        hidden, latent = int(p["hidden_dim"]), int(p["latent_dim"])

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

    def encoder_loss(self, model: Any, xb: Any) -> Any:
        torch = require("torch")
        noise = float(self.params()["noise"])
        corrupted = xb + noise * torch.randn_like(xb) if noise else xb
        return torch.nn.functional.mse_loss(model(corrupted), xb)


@register_model("vae", "variational-autoencoder")
class VAEAdapter(_RepresentationAdapter):
    """Variational autoencoder followed by a logistic probe."""

    def build_encoder(self, n_features: int) -> Any:
        torch = require("torch")
        p = self.params()
        hidden, latent = int(p["hidden_dim"]), int(p["latent_dim"])

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
                # Use the posterior mean as the representation -- deterministic
                # and therefore reproducible.
                return self.mu(self.body(x))

            def forward(self, x):
                h = self.body(x)
                mu, logvar = self.mu(h), self.logvar(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                return self.dec(z), mu, logvar

        return VAE()

    def encoder_loss(self, model: Any, xb: Any) -> Any:
        torch = require("torch")
        recon, mu, logvar = model(xb)
        recon_loss = torch.nn.functional.mse_loss(recon, xb, reduction="sum") / xb.shape[0]
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.shape[0]
        return recon_loss + float(self.params()["beta"]) * kld


@register_model("contrastive", "contrastive-encoder")
class ContrastiveEncoderAdapter(_RepresentationAdapter):
    """SimCLR-style contrastive encoder with Gaussian-noise augmentation."""

    def build_encoder(self, n_features: int) -> Any:
        torch = require("torch")
        p = self.params()
        hidden, latent = int(p["hidden_dim"]), int(p["latent_dim"])

        class Encoder(torch.nn.Module):
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

        return Encoder()

    def encoder_loss(self, model: Any, xb: Any) -> Any:
        torch = require("torch")
        p = self.params()
        noise, temperature = float(p["noise"]), float(p["temperature"])
        z1 = torch.nn.functional.normalize(model(xb + noise * torch.randn_like(xb)), dim=1)
        z2 = torch.nn.functional.normalize(model(xb + noise * torch.randn_like(xb)), dim=1)
        n = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        similarity = (z @ z.T) / temperature
        similarity.fill_diagonal_(float("-inf"))
        targets = torch.cat(
            [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)]
        )
        return torch.nn.functional.cross_entropy(similarity, targets)
