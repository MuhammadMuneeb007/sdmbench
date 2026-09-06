"""The experiment DAG and representation cache.

An experiment is a directed acyclic graph::

    dataset -> modality preprocessing -> representation -> fusion
            -> spatial/temporal structure -> objective/model
            -> prediction -> evaluation -> explanation

Why the DAG matters
-------------------
Compute is dominated by the early nodes. Encoding terrain with a multi-scale
CNN, or fetching AlphaEarth embeddings for 10,000 background points, costs
orders of magnitude more than fitting the classifier that consumes them. If
twenty methodologies share the same climate encoding, computing it twenty times
is pure waste.

So each node is keyed by a content hash of *everything upstream of it*, and
identical subtrees are computed once. The cache key includes the dataset hash,
the modality spec, the encoder and its parameters, and the split -- change any
of them and the key changes, so a stale artefact can never be silently reused.

Leakage and the cache
---------------------
A fitted encoder's state is part of its cache key via the split hash. That is
not an optimisation detail but a correctness requirement: a PCA basis fitted on
one split must never be reused for another, and keying on the split makes that
impossible by construction.
"""

from __future__ import annotations

import enum
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from sdmbench.paths import cache_root, ensure_dir
from sdmbench.reproducibility.hashes import hash_json

__all__ = ["NodeKind", "ExperimentNode", "ExperimentGraph", "RepresentationCache"]


class NodeKind(str, enum.Enum):
    """Stages of the pipeline, in dependency order."""

    DATASET = "dataset"
    ACQUISITION = "acquisition"
    PREPROCESSING = "preprocessing"
    SPLIT = "split"
    REPRESENTATION = "representation"
    FUSION = "fusion"
    STRUCTURE = "structure"
    MODEL = "model"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"

    @property
    def order(self) -> int:
        return list(NodeKind).index(self)


@dataclass
class ExperimentNode:
    """One computation in the DAG."""

    node_id: str
    kind: NodeKind
    #: Everything that determines this node's output, excluding its inputs.
    spec: dict[str, Any] = field(default_factory=dict)
    #: Node ids this one consumes.
    inputs: list[str] = field(default_factory=list)
    #: Whether the result is worth caching (expensive to recompute).
    cacheable: bool = True
    #: Rough cost hint used to order and report the plan.
    cost: str = "cheap"
    _key: str | None = field(default=None, repr=False)

    def key(self, upstream_keys: dict[str, str]) -> str:
        """Content hash of this node and everything upstream.

        Two nodes share a key exactly when they would compute the same thing.
        """
        if self._key is None:
            self._key = hash_json(
                {
                    "kind": self.kind.value,
                    "spec": self.spec,
                    "inputs": sorted(upstream_keys.get(i, i) for i in self.inputs),
                }
            )
        return self._key

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "spec": self.spec,
            "inputs": list(self.inputs),
            "cacheable": self.cacheable,
            "cost": self.cost,
        }


class ExperimentGraph:
    """A DAG of experiment nodes, with shared-subtree detection."""

    def __init__(self) -> None:
        self.nodes: dict[str, ExperimentNode] = {}

    def add(self, node: ExperimentNode) -> ExperimentNode:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id {node.node_id!r}")
        missing = [i for i in node.inputs if i not in self.nodes]
        if missing:
            raise ValueError(f"node {node.node_id!r} references unknown inputs: {missing}")
        self.nodes[node.node_id] = node
        return node

    def topological_order(self) -> list[ExperimentNode]:
        """Nodes in dependency order, raising on a cycle."""
        ordered: list[ExperimentNode] = []
        state: dict[str, int] = {}  # 0 = unvisited, 1 = visiting, 2 = done

        def visit(node_id: str, trail: tuple[str, ...]) -> None:
            mark = state.get(node_id, 0)
            if mark == 2:
                return
            if mark == 1:
                cycle = " -> ".join([*trail, node_id])
                raise ValueError(f"cycle in the experiment graph: {cycle}")
            state[node_id] = 1
            node = self.nodes[node_id]
            for parent in node.inputs:
                visit(parent, (*trail, node_id))
            state[node_id] = 2
            ordered.append(node)

        for node_id in self.nodes:
            visit(node_id, ())
        return ordered

    def keys(self) -> dict[str, str]:
        """Content key for every node."""
        computed: dict[str, str] = {}
        for node in self.topological_order():
            computed[node.node_id] = node.key(computed)
        return computed

    def shared_subtrees(self) -> dict[str, list[str]]:
        """Content keys shared by more than one node -- the reuse opportunities."""
        by_key: dict[str, list[str]] = {}
        for node_id, key in self.keys().items():
            by_key.setdefault(key, []).append(node_id)
        return {k: v for k, v in by_key.items() if len(v) > 1}

    def savings_report(self) -> dict[str, Any]:
        """What caching will avoid recomputing."""
        shared = self.shared_subtrees()
        duplicated = sum(len(v) - 1 for v in shared.values())
        expensive = [
            n.node_id for n in self.nodes.values() if n.cost in {"expensive", "very_expensive"}
        ]
        return {
            "n_nodes": len(self.nodes),
            "n_unique_computations": len(set(self.keys().values())),
            "n_duplicate_nodes_avoided": duplicated,
            "shared_groups": {k[:12]: v for k, v in shared.items()},
            "expensive_nodes": expensive,
        }

    def to_dict(self) -> dict[str, Any]:
        keys = self.keys()
        return {
            "nodes": [{**n.to_dict(), "key": keys[n.node_id]} for n in self.topological_order()],
            "savings": self.savings_report(),
        }

    def render(self) -> str:
        lines = ["Experiment graph", "=" * 16, ""]
        keys = self.keys()
        for node in self.topological_order():
            arrow = f" <- {', '.join(node.inputs)}" if node.inputs else ""
            lines.append(
                f"  [{node.kind.value:<14}] {node.node_id}  ({node.cost}, "
                f"{keys[node.node_id][:8]}){arrow}"
            )
        savings = self.savings_report()
        lines.append("")
        lines.append(
            f"{savings['n_nodes']} node(s), {savings['n_unique_computations']} unique "
            f"computation(s), {savings['n_duplicate_nodes_avoided']} avoided by caching."
        )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[ExperimentNode]:
        return iter(self.topological_order())


class RepresentationCache:
    """On-disk cache for expensive intermediate artefacts.

    Keyed by content hash, so a hit is guaranteed to be the same computation.
    Values are pickled, which is fine for a private cache directory but means
    the cache must never be shared between users or fetched from a network
    location -- unpickling untrusted data executes code.
    """

    def __init__(self, root: str | Path | None = None, *, enabled: bool = True) -> None:
        self.root = ensure_dir(Path(root) if root else cache_root() / "representations")
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        # Two-level fan-out keeps directories small on large runs.
        return self.root / key[:2] / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)  # noqa: S301 - private, self-written cache
        except Exception:  # noqa: BLE001 - a corrupt entry is a miss, not a crash
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: Any, *, metadata: dict[str, Any] | None = None) -> Path:
        path = self.path_for(key)
        ensure_dir(path.parent)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        if metadata:
            path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        return path

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Return the cached value, or compute and store it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        if self.enabled:
            self.put(key, value, metadata=metadata)
        return value

    def clear(self) -> int:
        """Delete every cached artefact, returning the number removed."""
        removed = 0
        for path in self.root.rglob("*.pkl"):
            path.unlink(missing_ok=True)
            removed += 1
        for path in self.root.rglob("*.json"):
            path.unlink(missing_ok=True)
        return removed

    def stats(self) -> dict[str, Any]:
        entries = list(self.root.rglob("*.pkl"))
        return {
            "root": str(self.root),
            "enabled": self.enabled,
            "n_entries": len(entries),
            "size_mb": round(sum(p.stat().st_size for p in entries) / 1e6, 2),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / max(self.hits + self.misses, 1), 3),
        }


def build_experiment_graph(
    methodology: Any,
    *,
    dataset_hash: str,
    split_hash: str,
) -> ExperimentGraph:
    """Construct the DAG for one methodology.

    Exposed so ``sdmbench plan`` can show what will be computed, and which
    parts are shared with other methodologies, before anything runs.
    """
    graph = ExperimentGraph()
    graph.add(
        ExperimentNode(
            "dataset", NodeKind.DATASET, spec={"dataset_hash": dataset_hash}, cost="cheap"
        )
    )
    graph.add(
        ExperimentNode(
            "split",
            NodeKind.SPLIT,
            spec={"split_hash": split_hash},
            inputs=["dataset"],
            cost="cheap",
        )
    )

    representation_nodes: list[str] = []
    for spec in methodology.modalities:
        acquisition = f"acquire:{spec.name}"
        graph.add(
            ExperimentNode(
                acquisition,
                NodeKind.ACQUISITION,
                spec={"modality": spec.name, "provider": spec.provider, "scales": list(spec.scales)},
                inputs=["dataset"],
                cost="expensive" if spec.provider else "cheap",
            )
        )
        node_id = f"encode:{spec.name}:{spec.representation}"
        graph.add(
            ExperimentNode(
                node_id,
                NodeKind.REPRESENTATION,
                spec={
                    "modality": spec.name,
                    "encoder": spec.representation,
                    "scales": list(spec.scales),
                    "options": dict(spec.options),
                },
                inputs=[acquisition, "split"],
                # Encoding is where the time goes: cache it aggressively.
                cost="expensive" if spec.scales or "cnn" in spec.representation else "moderate",
            )
        )
        representation_nodes.append(node_id)

    graph.add(
        ExperimentNode(
            f"fuse:{methodology.fusion}",
            NodeKind.FUSION,
            spec={"strategy": methodology.fusion, "options": methodology.fusion_options},
            inputs=representation_nodes,
            cost="moderate",
        )
    )
    last = f"fuse:{methodology.fusion}"

    if methodology.graph:
        graph.add(
            ExperimentNode(
                f"graph:{methodology.graph}",
                NodeKind.STRUCTURE,
                spec={"builder": methodology.graph, "options": methodology.graph_options},
                inputs=[last, "split"],
                cost="moderate",
            )
        )
        last = f"graph:{methodology.graph}"

    graph.add(
        ExperimentNode(
            f"model:{methodology.model}",
            NodeKind.MODEL,
            spec={
                "model": methodology.model,
                "objective": methodology.objective,
                "options": methodology.model_options,
            },
            inputs=[last],
            cacheable=False,
            cost="moderate",
        )
    )
    graph.add(
        ExperimentNode(
            "predict",
            NodeKind.PREDICTION,
            spec={},
            inputs=[f"model:{methodology.model}"],
            cacheable=False,
            cost="cheap",
        )
    )
    graph.add(
        ExperimentNode(
            "evaluate", NodeKind.EVALUATION, spec={}, inputs=["predict"], cacheable=False
        )
    )
    return graph
