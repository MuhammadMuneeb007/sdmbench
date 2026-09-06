"""Graph construction strategies.

Graph *structure* is a benchmark factor in its own right, separate from the
graph *model*. See :mod:`sdmbench.graphs.builders`.
"""

from sdmbench.graphs.builders import (
    GRAPH_BUILDERS,
    BuiltGraph,
    EnvironmentalKNNBuilder,
    GeographicKNNBuilder,
    GeographicRadiusBuilder,
    GraphBuilder,
    HybridGraphBuilder,
    PatchAdjacencyBuilder,
    UserGraphBuilder,
)

__all__ = [
    "GRAPH_BUILDERS",
    "BuiltGraph",
    "GraphBuilder",
    "GeographicKNNBuilder",
    "GeographicRadiusBuilder",
    "EnvironmentalKNNBuilder",
    "HybridGraphBuilder",
    "PatchAdjacencyBuilder",
    "UserGraphBuilder",
]
