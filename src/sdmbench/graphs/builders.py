"""Graph construction -- itself a benchmark factor.

How you wire observations into a graph is a modelling decision as consequential
as the choice of convolution, and the literature rarely tests it. sdmbench
treats it as a registry so the question

    Is geographic adjacency better than environmental adjacency?

can actually be answered.

Strategies
----------
``geographic-knn``
    Edges to the k nearest neighbours in space. Encodes "nearby places are
    similar" -- spatial autocorrelation as an explicit prior.
``geographic-radius``
    Edges within a fixed distance. Density-sensitive: clustered records get
    many edges, isolated ones few, which is sometimes exactly right and
    sometimes a bug.
``environmental-knn``
    Edges to the k nearest neighbours in *environmental* space. Encodes
    "environmentally similar places are similar", regardless of distance --
    a niche-space prior rather than a geographic one.
``patch-adjacency``
    Nodes are ecologically homogeneous landscape patches, edges are physical
    adjacency. This is the GNN-SDM formulation and is a genuinely different
    object from a point graph.
``hybrid``
    Weighted combination of geographic and environmental proximity.
``user``
    A user-supplied edge list.

Leakage
-------
Every builder returns edges over the node set it was given. The run engine
builds training and test graphs **separately** by default, so no message passes
between the two. Transductive construction is available but must be requested,
and :meth:`GraphSpec.transductive` records it in the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sdmbench.core.registry import Registry, Spec
from sdmbench.exceptions import DataError

__all__ = [
    "GRAPH_BUILDERS",
    "GraphBuilder",
    "BuiltGraph",
    "GeographicKNNBuilder",
    "GeographicRadiusBuilder",
    "EnvironmentalKNNBuilder",
    "PatchAdjacencyBuilder",
    "HybridGraphBuilder",
    "UserGraphBuilder",
]

#: The graph-construction registry.
GRAPH_BUILDERS: Registry["GraphBuilder"] = Registry("graph_builder")


def _graph_spec(description: str, *, requires: tuple[str, ...] = (), maturity: str = "stable") -> Spec:
    return Spec(
        accepts=("coordinates", "tabular"),
        produces="graph",
        requires=requires,
        description=description,
        maturity=maturity,
    )


@dataclass
class BuiltGraph:
    """An edge index plus the metadata needed to audit and report it."""

    #: ``(2, n_edges)`` array of node indices.
    edge_index: np.ndarray
    n_nodes: int
    strategy: str = ""
    #: Optional per-edge weights.
    edge_weight: np.ndarray | None = None
    #: Node-level features produced by the builder (patch graphs aggregate).
    node_features: np.ndarray | None = None
    #: Which original row each node came from (patch graphs are many-to-one).
    node_to_row: np.ndarray | None = None
    transductive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edge_index = np.asarray(self.edge_index, dtype=np.int64)
        if self.edge_index.size and self.edge_index.shape[0] != 2:
            self.edge_index = self.edge_index.T
        if self.edge_index.size == 0:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def mean_degree(self) -> float:
        return float(self.n_edges / self.n_nodes) if self.n_nodes else 0.0

    @property
    def isolated_nodes(self) -> int:
        if self.n_edges == 0:
            return self.n_nodes
        touched = np.unique(self.edge_index)
        return int(self.n_nodes - len(touched))

    def describe(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "mean_degree": round(self.mean_degree, 3),
            "isolated_nodes": self.isolated_nodes,
            "transductive": self.transductive,
            **self.metadata,
        }


class GraphBuilder:
    """Base class for a graph-construction strategy."""

    name: str = "graph"
    #: Whether this builder needs environmental features as well as coordinates.
    uses_features: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def build(
        self,
        coords: np.ndarray,
        features: np.ndarray | None = None,
        *,
        geographic: bool = False,
    ) -> BuiltGraph:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "params": {k: str(v) for k, v in self.params.items()}}

    # ----------------------------------------------------------------- utils --
    @staticmethod
    def _symmetrise(pairs: np.ndarray) -> np.ndarray:
        """Make an edge list undirected and de-duplicated."""
        if pairs.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        both = np.vstack([pairs, pairs[:, ::-1]])
        both = both[both[:, 0] != both[:, 1]]  # drop self-loops
        return np.unique(both, axis=0).T.astype(np.int64)

    @staticmethod
    def _knn_pairs(points: np.ndarray, k: int) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors

        n = len(points)
        if n < 2:
            return np.zeros((0, 2), dtype=np.int64)
        k_eff = min(k + 1, n)
        model = NearestNeighbors(n_neighbors=k_eff).fit(points)
        _, indices = model.kneighbors(points)
        sources = np.repeat(np.arange(n), k_eff - 1)
        targets = indices[:, 1:].ravel()
        return np.column_stack([sources, targets])


@GRAPH_BUILDERS.register(
    "geographic-knn",
    "knn",
    "geographic_knn",
    spec=_graph_spec("Connect each observation to its k nearest neighbours in space."),
)
class GeographicKNNBuilder(GraphBuilder):
    """k-nearest neighbours in geographic space.

    For a geographic CRS the coordinates are projected to the unit sphere
    first, so that "nearest" means great-circle nearest rather than nearest in
    degrees -- which near the poles is a very different set of neighbours.
    """

    def build(self, coords, features=None, *, geographic: bool = False) -> BuiltGraph:
        coords = np.asarray(coords, dtype=float)
        k = int(self.params.get("k", 8))
        points = _to_metric_space(coords, geographic=geographic)
        edges = self._symmetrise(self._knn_pairs(points, k))
        return BuiltGraph(
            edge_index=edges,
            n_nodes=len(coords),
            strategy=self.name,
            metadata={"k": k, "geographic": geographic},
        )


@GRAPH_BUILDERS.register(
    "geographic-radius",
    "radius",
    spec=_graph_spec("Connect all observation pairs within a fixed distance."),
)
class GeographicRadiusBuilder(GraphBuilder):
    """Fixed-radius geographic graph.

    Degree varies with sampling density, so heavily surveyed areas become
    hubs. That is a real property of the data, not a defect -- but it is worth
    knowing, so the isolated-node count is reported.
    """

    def build(self, coords, features=None, *, geographic: bool = False) -> BuiltGraph:
        from scipy.spatial import cKDTree

        coords = np.asarray(coords, dtype=float)
        radius = self.params.get("radius")
        if not radius:
            raise DataError("geographic-radius needs a positive `radius` (metres)")
        points = _to_metric_space(coords, geographic=geographic)
        threshold = (
            _chord_for_metres(float(radius)) if geographic else float(radius)
        )
        pairs = np.asarray(list(cKDTree(points).query_pairs(r=threshold)), dtype=np.int64)
        edges = self._symmetrise(pairs.reshape(-1, 2) if pairs.size else pairs)
        return BuiltGraph(
            edge_index=edges,
            n_nodes=len(coords),
            strategy=self.name,
            metadata={"radius_m": float(radius), "geographic": geographic},
        )


@GRAPH_BUILDERS.register(
    "environmental-knn",
    "environmental",
    spec=_graph_spec(
        "Connect observations that are similar in environmental space, not in geography."
    ),
)
class EnvironmentalKNNBuilder(GraphBuilder):
    """k-nearest neighbours in environmental (niche) space.

    Tests a genuinely different hypothesis from the geographic graph: that
    what matters is environmental similarity, wherever it occurs. Two sites on
    different continents with the same climate become neighbours.

    Features are standardised before the neighbour search so that a variable
    measured in millimetres does not dominate one measured in degrees.
    """

    uses_features = True

    def build(self, coords, features=None, *, geographic: bool = False) -> BuiltGraph:
        if features is None:
            raise DataError("environmental-knn needs environmental features")
        array = np.nan_to_num(np.asarray(features, dtype=float))
        std = array.std(axis=0, ddof=1) if len(array) > 1 else np.ones(array.shape[1])
        standardised = (array - array.mean(axis=0)) / np.where(std > 0, std, 1.0)
        k = int(self.params.get("k", 8))
        edges = self._symmetrise(self._knn_pairs(standardised, k))
        return BuiltGraph(
            edge_index=edges,
            n_nodes=len(array),
            strategy=self.name,
            metadata={"k": k, "space": "environmental", "n_features": array.shape[1]},
        )


@GRAPH_BUILDERS.register(
    "hybrid",
    spec=_graph_spec(
        "Weighted combination of geographic and environmental proximity.",
        maturity="emerging",
    ),
)
class HybridGraphBuilder(GraphBuilder):
    """Edges from a weighted blend of geographic and environmental distance.

    ``alpha`` = 1 is purely geographic, 0 purely environmental. Sweeping it is
    a direct test of which notion of proximity carries the signal.
    """

    uses_features = True

    def build(self, coords, features=None, *, geographic: bool = False) -> BuiltGraph:
        if features is None:
            raise DataError("hybrid graph needs environmental features")
        coords = np.asarray(coords, dtype=float)
        array = np.nan_to_num(np.asarray(features, dtype=float))
        alpha = float(self.params.get("alpha", 0.5))
        k = int(self.params.get("k", 8))

        space = _standardise(_to_metric_space(coords, geographic=geographic))
        environment = _standardise(array)
        # Scaling each block by sqrt(weight) makes Euclidean distance in the
        # concatenated space equal the weighted sum of the two distances.
        blended = np.hstack(
            [space * np.sqrt(alpha), environment * np.sqrt(1.0 - alpha)]
        )
        edges = self._symmetrise(self._knn_pairs(blended, k))
        return BuiltGraph(
            edge_index=edges,
            n_nodes=len(coords),
            strategy=self.name,
            metadata={"k": k, "alpha": alpha, "space": "geographic+environmental"},
        )


@GRAPH_BUILDERS.register(
    "patch-adjacency",
    "patch",
    spec=_graph_spec(
        "Nodes are ecologically homogeneous landscape patches; edges are adjacency.",
        maturity="experimental",
    ),
)
class PatchAdjacencyBuilder(GraphBuilder):
    """Landscape-patch graph (the GNN-SDM formulation).

    Structurally different from the point graphs above: observations are first
    clustered into ecologically homogeneous patches, each patch becomes a node
    carrying aggregated features, and edges connect spatially adjacent patches.
    The published method forms patches with a Self-Organizing Map; this
    implementation uses k-means over the joint environment-space representation,
    which is a **documented substitution, not a reproduction** of that step.

    Because several observations map to one node, :attr:`BuiltGraph.node_to_row`
    records the assignment -- the run engine needs it to map predictions back.

    NOT VERIFIED against the GNN-SDM reference implementation.
    """

    uses_features = True

    def build(self, coords, features=None, *, geographic: bool = False) -> BuiltGraph:
        if features is None:
            raise DataError("patch-adjacency needs environmental features")
        from sklearn.cluster import KMeans

        coords = np.asarray(coords, dtype=float)
        array = np.nan_to_num(np.asarray(features, dtype=float))
        n_patches = int(self.params.get("n_patches", max(2, len(coords) // 20)))
        n_patches = max(2, min(n_patches, len(coords)))
        env_weight = float(self.params.get("env_weight", 0.5))

        joint = np.hstack(
            [
                _standardise(_to_metric_space(coords, geographic=geographic))
                * np.sqrt(1.0 - env_weight),
                _standardise(array) * np.sqrt(env_weight),
            ]
        )
        labels = KMeans(
            n_clusters=n_patches,
            random_state=int(self.params.get("seed", 32639)),
            n_init=10,
        ).fit_predict(joint)

        centroids = np.vstack(
            [coords[labels == p].mean(axis=0) for p in range(n_patches)]
        )
        patch_features = np.vstack(
            [array[labels == p].mean(axis=0) for p in range(n_patches)]
        )
        k = int(self.params.get("k", 4))
        edges = self._symmetrise(self._knn_pairs(centroids, min(k, n_patches - 1)))

        return BuiltGraph(
            edge_index=edges,
            n_nodes=n_patches,
            strategy=self.name,
            node_features=patch_features,
            node_to_row=labels,
            metadata={
                "n_patches": n_patches,
                "env_weight": env_weight,
                "patching": "kmeans (SUBSTITUTION for the published SOM step)",
                "verified_against_upstream": False,
            },
        )


@GRAPH_BUILDERS.register(
    "user",
    "user-graph",
    spec=_graph_spec("Use a user-supplied edge list or adjacency matrix."),
)
class UserGraphBuilder(GraphBuilder):
    """A graph the user supplies.

    Accepts ``edge_index`` as ``(2, e)``/``(e, 2)``, or ``adjacency`` as a
    dense/boolean matrix.
    """

    def build(self, coords, features=None, *, geographic: bool = False) -> BuiltGraph:
        n_nodes = len(np.asarray(coords))
        edge_index = self.params.get("edge_index")
        adjacency = self.params.get("adjacency")

        if edge_index is not None:
            pairs = np.asarray(edge_index, dtype=np.int64)
            if pairs.shape[0] == 2 and pairs.shape[1] != 2:
                pairs = pairs.T
        elif adjacency is not None:
            matrix = np.asarray(adjacency)
            if matrix.shape != (n_nodes, n_nodes):
                raise DataError(
                    f"adjacency must be ({n_nodes}, {n_nodes}), got {matrix.shape}"
                )
            pairs = np.column_stack(np.nonzero(matrix))
        else:
            raise DataError("user graph needs edge_index= or adjacency=")

        if pairs.size and (pairs.max() >= n_nodes or pairs.min() < 0):
            raise DataError(
                f"edge indices must lie in [0, {n_nodes}); got "
                f"[{pairs.min()}, {pairs.max()}]"
            )
        return BuiltGraph(
            edge_index=self._symmetrise(pairs),
            n_nodes=n_nodes,
            strategy=self.name,
            metadata={"source": "user-supplied"},
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_S2_EARTH_RADIUS_M = 6_371_010.0


def _to_metric_space(coords: np.ndarray, *, geographic: bool) -> np.ndarray:
    """Project coordinates so Euclidean distance is meaningful.

    Geographic coordinates go to the unit sphere; projected ones pass through.
    """
    coords = np.asarray(coords, dtype=float)
    if not geographic:
        return coords
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def _chord_for_metres(distance_m: float) -> float:
    """Unit-sphere chord length for a great-circle distance in metres."""
    arc = min(distance_m / _S2_EARTH_RADIUS_M, np.pi)
    return 2.0 * np.sin(arc / 2.0)


def _standardise(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(array, dtype=float))
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    std = array.std(axis=0, ddof=1) if len(array) > 1 else np.ones(array.shape[1])
    return (array - array.mean(axis=0)) / np.where(std > 0, std, 1.0)
