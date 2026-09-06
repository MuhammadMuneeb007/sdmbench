# Graph methods

Graph **structure** is a benchmark factor in its own right, separate from the
graph **model**. The literature rarely tests it; sdmbench makes it a registry so
the question can be answered.

---

## The formulation

```
node          one occurrence or background observation
node features ENVIRONMENTAL covariates only
edges         geographic (or environmental) neighbours
```

**Coordinates are not node features.** They define the topology. Feeding
longitude and latitude to a model as covariates lets it memorise where a species
has been recorded rather than which environments it occupies — inflating
interpolation performance and destroying transferability, which is the very
thing the spatial scenario exists to measure.

A model can therefore smooth its predictions over spatial neighbourhoods without
ever reading a coordinate as a number.

---

## Builders

| Strategy | Edges connect | Hypothesis it encodes |
|---|---|---|
| `geographic-knn` | k nearest in space | spatial autocorrelation |
| `geographic-radius` | pairs within a distance | fixed-range interaction |
| `environmental-knn` | k nearest in **niche space** | environmental similarity, regardless of distance |
| `hybrid` | weighted blend (`alpha`) | both, tunable |
| `patch-adjacency` | adjacent landscape **patches** | landscape connectivity |
| `user` | a supplied edge list | domain knowledge |

`environmental-knn` tests a genuinely different hypothesis: two sites on
different continents with the same climate become neighbours. Sweeping `hybrid`'s
`alpha` from 1 (pure geography) to 0 (pure environment) directly measures which
notion of proximity carries the signal.

### Geographic distance

For a geographic CRS, coordinates are projected onto the unit sphere before the
neighbour search, so "nearest" means great-circle nearest rather than nearest in
degrees. Near the poles those are very different sets.

### `patch-adjacency` — a documented substitution

Structurally different from the point graphs: observations are clustered into
ecologically homogeneous patches, each patch becomes a node with aggregated
features, and edges connect spatially adjacent patches. This is the GNN-SDM
formulation.

> ⚠️ The published method forms patches with a **Self-Organizing Map**. This
> implementation uses **k-means** over the joint environment-space
> representation. That is a **documented substitution, not a reproduction**.
> `graph.metadata["verified_against_upstream"]` is `False` and
> `metadata["patching"]` says `SUBSTITUTION`.

Because several observations map to one node, `BuiltGraph.node_to_row` records
the assignment so predictions can be mapped back.

---

## Models

Stock PyTorch Geometric operators. No hand-written graph convolutions.

| Adapter | Layer |
|---|---|
| `gcn` | `GCNConv` |
| `graphsage` | `SAGEConv` |
| `gatv2` | `GATv2Conv` |
| `graph-transformer` | `TransformerConv` |

---

## Leakage

Message passing moves information along edges, so an edge between a training
node and a test node lets test environment shape the fitted representation.

**Default: `graph_mode="inductive"`.** Training and test graphs are built
separately; the model fits on the training subgraph and a fresh test subgraph is
built at prediction time.

**`graph_mode="transductive"`** builds one graph over both sets. It is a
legitimate design for some tasks but is **not leakage-free for this benchmark**.
It must be requested explicitly, the adapter sets `uses_test_features=True`, and
`LeakageAuditor.audit_graph_edges` reports any crossing edges.

Early stopping monitors **training** loss only — stopping on test performance
would be selection on the test set even though the labels never enter the loss.

---

## The stage-5 caveat

The planner's spatial stage compares graph strategies against a no-graph arm.
But graph strategies need a graph model and the no-graph arm does not, so **the
stage confounds topology with architecture.** The plan says so:

```
CAVEAT: graph arms use a graph model and the no-graph arm does not, so this
stage confounds topology with architecture. To isolate topology, compare the
graph strategies against each other rather than against the no-graph arm.
```

To isolate the effect of topology, compare `geographic-knn` against
`environmental-knn` against `hybrid` — same model, different structure.

---

## Configuration

```yaml
spatial:
  method: geographic-knn
  k: 8

model:
  model: graph-transformer
  options:
    graph_mode: inductive
    hidden_dim: 64
    n_layers: 2
    heads: 4
    epochs: 300
```

---

## Adding a builder

```python
from sdmbench.graphs.builders import GRAPH_BUILDERS, BuiltGraph, GraphBuilder

@GRAPH_BUILDERS.register("my-graph", spec=_graph_spec("What it connects."))
class MyGraphBuilder(GraphBuilder):
    uses_features = True     # True if it needs environmental features

    def build(self, coords, features=None, *, geographic=False) -> BuiltGraph:
        pairs = ...                       # (n_edges, 2)
        return BuiltGraph(
            edge_index=self._symmetrise(pairs),   # undirected, no self-loops
            n_nodes=len(coords),
            strategy=self.name,
            metadata={"...": "recorded in the results"},
        )
```

Use `_symmetrise` — it removes self-loops and de-duplicates, so message passing
is symmetric and PyG's own self-loop handling is not doubled.
