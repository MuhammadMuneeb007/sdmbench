"""Graph neural network adapters (PyTorch Geometric).

The formulation, and why it is the default
------------------------------------------
Coordinates are *not* predictive features here. Feeding longitude and latitude
to a model as covariates lets it memorise where a species has been recorded
rather than which environments it occupies, which inflates interpolation
performance and destroys transferability -- the very thing the spatial scenario
exists to measure.

Instead, geography defines the **topology** and environment defines the
**features**::

    node          one occurrence or background observation
    node features environmental covariates only
    edges         geographic nearest neighbours (or a fixed radius)

A model can therefore smooth its predictions over spatial neighbourhoods
without ever reading a coordinate as a number.

Leakage control
---------------
Message passing moves information along edges, so an edge between a training
node and a test node would let test environment influence the fitted
representation. Two safeguards:

* Graphs are built **separately** for the training and test sets by default
  (``graph_mode="inductive"``): the model fits on the training subgraph and a
  fresh test subgraph is built at prediction time.
* ``graph_mode="transductive"`` builds one graph over both sets, which is a
  legitimate design for some tasks but is *not* leakage-free for this
  benchmark. It is available, must be requested explicitly, and is flagged in
  the results.

All convolutions are the stock PyTorch Geometric operators -- ``GCNConv``,
``SAGEConv``, ``GATv2Conv``, ``TransformerConv``. None is hand-written.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from sdmbench.data.base import PreparedSplit
from sdmbench.exceptions import SdmbenchError
from sdmbench.models.base import FitResult, ModelAdapter, register_model
from sdmbench.optional import package_version, require, resolve_device

__all__ = [
    "GraphAdapter",
    "GCNAdapter",
    "GraphSAGEAdapter",
    "GATv2Adapter",
    "GraphTransformerAdapter",
    "build_knn_edges",
    "build_radius_edges",
]


def build_knn_edges(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """Undirected k-nearest-neighbour edge index from coordinates.

    Returns a ``(2, n_edges)`` array. Self-loops are excluded (PyG layers add
    their own where needed) and each pair appears in both directions so message
    passing is symmetric.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    from sklearn.neighbors import NearestNeighbors

    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(coords)
    _, indices = nn.kneighbors(coords)

    sources = np.repeat(np.arange(n), k_eff - 1)
    targets = indices[:, 1:].ravel()
    both = np.column_stack(
        [np.concatenate([sources, targets]), np.concatenate([targets, sources])]
    )
    both = np.unique(both, axis=0)
    return both.T.astype(np.int64)


def build_radius_edges(coords: np.ndarray, radius: float) -> np.ndarray:
    """Undirected edge index connecting all point pairs within ``radius``."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2 or radius <= 0:
        return np.zeros((2, 0), dtype=np.int64)
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    pairs = np.asarray(list(tree.query_pairs(r=radius)), dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    both = np.vstack([pairs, pairs[:, ::-1]])
    return both.T.astype(np.int64)


class GraphAdapter(ModelAdapter):
    """Base class for graph neural network adapters.

    Hyperparameters
    ---------------
    graph_type: ``"knn"`` (default) or ``"radius"``
    k: neighbours for ``knn``
    radius: distance for ``radius``, in the region's coordinate units
    graph_mode: ``"inductive"`` (default, leakage-free) or ``"transductive"``
    hidden_dim, n_layers, dropout, lr, weight_decay, epochs
    """

    family = "graph"
    requires_gpu = False
    supports_categorical = False

    #: PyG layer class name, resolved from ``torch_geometric.nn``.
    conv_layer: ClassVar[str] = "GCNConv"
    #: Layers whose constructor accepts ``heads=``.
    supports_heads: ClassVar[bool] = False

    default_params: ClassVar[dict[str, Any]] = {
        "graph_type": "knn",
        "k": 8,
        "radius": None,
        "graph_mode": "inductive",
        "hidden_dim": 64,
        "n_layers": 2,
        "dropout": 0.3,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 200,
        "heads": 4,
        "patience": 30,
    }

    def check_available(self) -> None:
        require("torch")
        require("torch_geometric")
        super().check_available()

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        # Transductive graphs see test features; declare it on the instance so
        # the leakage auditor and the results table both know.
        self.uses_test_features = self.params()["graph_mode"] == "transductive"

    def params(self) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update({k: v for k, v in self.hyperparameters.items() if k != "seed"})
        return merged

    def device_used(self) -> str:
        return resolve_device(self.hyperparameters.get("device", "auto"))

    # ----------------------------------------------------------------- graph --
    def _edges(self, coords: np.ndarray) -> np.ndarray:
        p = self.params()
        if p["graph_type"] == "radius":
            if not p.get("radius"):
                raise SdmbenchError("graph_type='radius' requires a positive `radius`")
            return build_radius_edges(coords, float(p["radius"]))
        return build_knn_edges(coords, int(p["k"]))

    def _build_module(self, n_features: int) -> Any:
        """Assemble the network from stock PyG convolutions."""
        torch = require("torch")
        pyg_nn = require("torch_geometric").nn
        p = self.params()

        conv_cls = getattr(pyg_nn, self.conv_layer, None)
        if conv_cls is None:
            raise SdmbenchError(
                f"torch_geometric.nn has no layer {self.conv_layer!r}; "
                f"installed torch-geometric: {package_version('torch-geometric')}"
            )

        hidden = int(p["hidden_dim"])
        n_layers = max(1, int(p["n_layers"]))
        dropout = float(p["dropout"])
        heads = int(p["heads"]) if self.supports_heads else 1
        conv_layer_name = self.conv_layer

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.convs = torch.nn.ModuleList()
                in_dim = n_features
                for _ in range(n_layers):
                    if conv_layer_name in {"GATv2Conv", "TransformerConv"}:
                        self.convs.append(conv_cls(in_dim, hidden, heads=heads, dropout=dropout))
                        in_dim = hidden * heads
                    else:
                        self.convs.append(conv_cls(in_dim, hidden))
                        in_dim = hidden
                self.dropout = torch.nn.Dropout(dropout)
                self.head = torch.nn.Linear(in_dim, 2)

            def forward(self, x, edge_index):
                for conv in self.convs:
                    x = conv(x, edge_index)
                    x = torch.nn.functional.relu(x)
                    x = self.dropout(x)
                return self.head(x)

        return Net()

    # ------------------------------------------------------------------- run --
    def run(self, split: PreparedSplit) -> FitResult:
        torch = require("torch")
        require("torch_geometric")

        X_train, y_train, X_test, _ = split.as_arrays()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        p = self.params()
        device = torch.device(self.device_used())
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = self._build_module(X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        # Weight the rare class so the network does not collapse to predicting
        # "background" for everything under 1:100 imbalance.
        counts = np.bincount(y_train, minlength=2).astype(float)
        weights = torch.tensor(
            (counts.sum() / np.maximum(counts, 1.0)) / 2.0, dtype=torch.float32, device=device
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        transductive = p["graph_mode"] == "transductive"
        if transductive:
            coords = np.vstack([split.coords_train, split.coords_test])
            features = np.vstack([X_train, X_test])
            edge_index = self._edges(coords)
            train_slice = slice(0, len(X_train))
            test_slice = slice(len(X_train), len(features))
        else:
            features = X_train
            edge_index = self._edges(split.coords_train)
            train_slice = slice(0, len(X_train))
            test_slice = None

        x = torch.tensor(features, dtype=torch.float32, device=device)
        ei = torch.tensor(edge_index, dtype=torch.long, device=device)
        y = torch.tensor(y_train, dtype=torch.long, device=device)

        started = time.perf_counter()
        model.train()
        best_loss, best_state, waited = float("inf"), None, 0
        patience = int(p["patience"])
        for _epoch in range(int(p["epochs"])):
            optimizer.zero_grad()
            logits = model(x, ei)
            loss = criterion(logits[train_slice], y)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            # Early stopping on TRAINING loss only. Using the independent test
            # set to stop would be exactly the leakage this framework exists to
            # prevent.
            if value < best_loss - 1e-5:
                best_loss, waited = value, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        model.eval()
        with torch.no_grad():
            if transductive:
                logits = model(x, ei)[test_slice]
            else:
                x_test = torch.tensor(X_test, dtype=torch.float32, device=device)
                ei_test = torch.tensor(
                    self._edges(split.coords_test), dtype=torch.long, device=device
                )
                logits = model(x_test, ei_test)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        predict_seconds = time.perf_counter() - started

        return FitResult(
            probabilities=np.asarray(probabilities, dtype=float),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            device=str(device),
            extra={
                "n_edges": int(edge_index.shape[1]),
                "graph_mode": p["graph_mode"],
                "final_train_loss": best_loss,
            },
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("graph adapters fit inside run()")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError("graph adapters predict inside run()")

    def effective_hyperparameters(self) -> dict[str, Any]:
        return {**self.params(), "seed": self.seed, "conv_layer": self.conv_layer}

    def version(self) -> str:
        return package_version("torch-geometric")


@register_model("gcn")
class GCNAdapter(GraphAdapter):
    """Graph convolutional network (``torch_geometric.nn.GCNConv``)."""

    conv_layer = "GCNConv"


@register_model("graphsage", "sage")
class GraphSAGEAdapter(GraphAdapter):
    """GraphSAGE (``torch_geometric.nn.SAGEConv``)."""

    conv_layer = "SAGEConv"


@register_model("gatv2", "gat")
class GATv2Adapter(GraphAdapter):
    """Graph attention network v2 (``torch_geometric.nn.GATv2Conv``)."""

    conv_layer = "GATv2Conv"
    supports_heads = True


@register_model("graph-transformer", "graph_transformer")
class GraphTransformerAdapter(GraphAdapter):
    """Graph transformer (``torch_geometric.nn.TransformerConv``)."""

    conv_layer = "TransformerConv"
    supports_heads = True
