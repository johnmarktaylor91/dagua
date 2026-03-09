"""Composable constraint loss functions.

Each constraint is a callable: (pos, graph_data) -> scalar loss.

Built-in constraints:
  - DAG: enforce top-to-bottom ordering for directed edges
  - Repel: prevent node overlap via pairwise repulsion
  - Attract: pull connected nodes together (edge attraction)
  - Overlap: hard overlap penalty (complements projection.py)
  - Cluster: group related nodes spatially
  - Align: rank/axis alignment for specific node groups

Users can write custom constraints in ~3 lines by following this protocol.
"""
