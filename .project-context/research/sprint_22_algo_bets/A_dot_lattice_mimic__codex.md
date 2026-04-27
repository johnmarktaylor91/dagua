# Area A -- Reverse-engineer graphviz_dot lattice mimic (codex)

## TL;DR

- **Single biggest call:** do **not** ship a post-pipeline "dot lattice x-snap" polish candidate for the Sprint-22 target lattices. My /tmp implementation of a dot-like 1-D constrained x solver produced `+0.00` on `hexagonal_lattice_42`, `+0.00` on `triangular_lattice_36`, and `+0.00` on `parallel_cycles_4x5` from cached dagua positions under the same fast scorer.
- dot's lattice pattern is not K-means and not a uniform grid. It is the side effect of dot's normal Sugiyama pipeline: rank assignment, dummy-node expansion, mincross ordering, then a second network-simplex pass over an auxiliary left-to-right constraint graph for x coordinates.
- The reverse-engineered hex rows are mostly exact 72-point steps, but with row starts and a few 90/108 gaps induced by the auxiliary constraints. Triangular rows are mostly 110-point steps with edge-case 118/72 gaps. This is "network-simplex compacted ranks", not free geometric lattice fitting.
- The failed mimic is still useful evidence: if x is changed after dagua's saturated layout, the target lattices prefer baseline because the candidate does not reproduce dot's coupled row ordering/dummy-node decisions. The useful integration point is earlier than polish: a real dot-style layered coordinate assignment candidate after layer/order construction.
- If implemented, build it as a new metric-picked layered coordinate candidate, not a default replacement. Protect `grid_5x5`, `grid_rect_6x8`, `sierpinski_42`, `outerplanar_dag_20`, `clustered_medium_5x20`, and all Sprint-21 wins.

## Sources And Mechanism

The Graphviz docs describe dot as a hierarchical layout engine: it breaks cycles, assigns nodes to discrete ranks, inserts virtual nodes for long edges, orders nodes within ranks to reduce crossings, then sets x coordinates to keep edges short before routing splines. See Graphviz's dot documentation and user guide:

- https://graphviz.org/docs/layouts/dot/
- https://www.graphviz.org/pdf/dotguide.pdf

The source is more explicit. In `lib/dotgen/position.c`, the file header says coordinates are computed by constructing and ranking an auxiliary graph. `dot_position()` calls `set_ycoords()`, `create_aux_edges()`, `rank(g, 2, nsiter2(g))`, then `set_xcoords()`. In `make_LR_constraints()`, dot adds auxiliary edges between adjacent same-rank nodes with minimum lengths based on left/right node widths plus `nodesep`; it also adds constraints for flat edges. In `make_edge_pairs()`, dot adds slack nodes and weighted auxiliary edges corresponding to input edges and ports. Finally, `set_xcoords()` copies the network-simplex `ND_rank` values into `ND_coord(v).x`.

Relevant source URLs:

- https://gitlab.com/graphviz/graphviz/-/raw/main/lib/dotgen/position.c
- https://gitlab.com/graphviz/graphviz/-/raw/main/lib/dotgen/rank.c
- https://gitlab.com/graphviz/graphviz/-/raw/main/lib/dotgen/mincross.c

The important source facts I verified:

- `position.c:13-17`: position uses `GD_rank(g)` and computes coordinates by ranking an auxiliary graph.
- `position.c:141-148`: dot creates auxiliary edges, runs `rank(g, 2, nsiter2(g))`, and then sets x coordinates.
- `position.c:238-267`: dot adds left-to-right same-rank constraints between adjacent nodes.
- `position.c:326-348`: dot adds virtual slack nodes and weighted edge-pair constraints for original edges.
- `position.c:525-531`: `create_aux_edges()` is exactly `make_LR_constraints()`, `make_edge_pairs()`, cluster constraints, and compression constraints.
- `position.c:569-581`: final x coordinates are the auxiliary ranking values.
- `rank.c:13-17`: dot's ranking is network-simplex based.
- `mincross.c:13-16`: dot's rank order comes from a crossing minimizer over the global rank structure.

Interpretation: dot's "lattice" result is not a lattice recognizer. For these generated lattices, the normal layered pipeline happens to create triangular/honeycomb row widths, and the auxiliary x-rank simplex compacts them into mostly regular steps while satisfying edge-shortening and same-rank separation constraints.

## Reverse-engineered Position Pattern

### hexagonal_lattice_42

Cached file: `eval_output/variant_bench_full/positions/hexagonal_lattice_42__graphviz_dot.pt`

Summary:

- `unique_x = 18`
- `unique_y = 12`
- `x_range = [27, 459]`
- `y_range = [-810, -18]`
- dot edge-length CV from cached positions: `0.0991`

Rows by y:

| y | width | x min | x max | steps |
|---:|---:|---:|---:|---|
| -810 | 1 | 117 | 117 | [] |
| -738 | 2 | 81 | 153 | 72 |
| -666 | 3 | 81 | 261 | 72, 108 |
| -594 | 4 | 63 | 279 | 72, 72, 72 |
| -522 | 5 | 63 | 369 | 72, 72, 72, 90 |
| -450 | 6 | 27 | 387 | 72, 72, 72, 72, 72 |
| -378 | 6 | 99 | 459 | 72, 72, 72, 72, 72 |
| -306 | 5 | 135 | 423 | 72, 72, 72, 72 |
| -234 | 4 | 207 | 423 | 72, 72, 72 |
| -162 | 3 | 207 | 351 | 72, 72 |
| -90 | 2 | 279 | 351 | 72 |
| -18 | 1 | 315 | 315 | [] |

This is the clearest proof that dot is not optimizing to equal edge lengths. Most within-row gaps are 72, but two rows contain a larger local gap. The row starts move by 18/36/72-point increments, not by a simple alternating half-cell phase. That pattern is consistent with network-simplex compaction under constraints.

### triangular_lattice_36

Cached dot summary:

- `unique_x = 11`
- `unique_y = 11`
- `x_range = [27, 539]`
- `y_range = [-738, -18]`
- dot edge-length CV: `0.2335`

Rows:

| y | width | x min | x max | steps |
|---:|---:|---:|---:|---|
| -738 | 1 | 302 | 302 | [] |
| -666 | 2 | 247 | 357 | 110 |
| -594 | 3 | 192 | 412 | 110, 110 |
| -522 | 4 | 137 | 467 | 110, 110, 110 |
| -450 | 5 | 82 | 530 | 110, 110, 110, 118 |
| -378 | 6 | 27 | 539 | 110, 110, 110, 110, 72 |
| -306 | 5 | 82 | 530 | 110, 110, 110, 118 |
| -234 | 4 | 137 | 467 | 110, 110, 110 |
| -162 | 3 | 192 | 412 | 110, 110 |
| -90 | 2 | 247 | 357 | 110 |
| -18 | 1 | 302 | 302 | [] |

The triangular pattern is closer to a simple triangular row ladder, but still has nonuniform edge-case gaps. Again, this looks like constrained compaction, not a lattice-specific equalizer.

### parallel_cycles_4x5

Cached dot summary:

- `unique_x = 8`
- `unique_y = 5`
- `x_range = [41, 480]`
- `y_range = [-306, -18]`
- dot edge-length CV: `0.7349`

Rows:

| y | width | x min | x max | steps |
|---:|---:|---:|---:|---|
| -306 | 4 | 76 | 480 | 138, 147, 119 |
| -234 | 4 | 41 | 455 | 138, 138, 138 |
| -162 | 4 | 41 | 455 | 138, 138, 138 |
| -90 | 4 | 41 | 455 | 138, 138, 138 |
| -18 | 4 | 76 | 480 | 138, 147, 119 |

This target is not really a dot lattice target. The prompt's best competitor is `graphviz_sfdp`, not dot. The sfdp win comes from cycle geometry and very low edge CV, while dot/ELK/Sugiyama impose a five-row hierarchy with worse cyclic aesthetics.

## /tmp Implementation

Script: `/tmp/dot_lattice_mimic.py`

The prototype is a research-only implementation. It does not modify `dagua/`. It builds real evaluation graphs directly from `dagua.eval.graphs` constructors, loads cached positions from `eval_output/variant_bench_full/positions`, groups rows by y, and scores candidate positions against cached dagua positions.

Because this machine was heavily CPU-saturated by other concurrent Sprint-22 jobs, I used a deterministic fast composite proxy for the search: `quick()` plus the normal `composite()` formula. This preserves the main terms this bet can affect (`edge_length_cv`, `dag_consistency`, `depth_spearman_rho`, `overlap_count`, and `edge_straightness_mean_deg`) but omits sampled crossing and angular-resolution terms during the candidate sweep. The official prompt numbers remain the authoritative full benchmark scores; the measured deltas below are actual `/tmp` prototype deltas under the fast scorer, with baseline and candidate scored identically.

The implemented mimic:

1. Load cached dagua position.
2. Infer row/layer IDs by y-coordinate quantization.
3. Keep per-row left-to-right order from the cached position.
4. Solve a 1-D projected relaxation:
   - anchors each node to its original x;
   - pulls adjacent-layer endpoints toward the same x;
   - projects after each iteration onto same-row left-to-right spacing constraints.
5. Optionally snap row centers to a staggered pitch estimated from edge x deltas.
6. Score all candidates and keep the best by composite.

This is intentionally close to dot's x-position principle but without dot's full dummy-expanded mincross order and exact network simplex. The point was to test whether a feasible polish candidate can close Area A by itself. It cannot.

## Working Pseudocode

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class DotMimicConfig:
    """Configuration for a dot-like x-coordinate candidate.

    Parameters
    ----------
    node_sep : float
        Minimum horizontal gap between adjacent same-layer node boxes.
    edge_weight : float
        Strength of endpoint x-equality terms.
    anchor_weight : float
        Strength of the penalty that keeps x near the source layout.
    iterations : int
        Number of projected relaxation iterations.
    snap_pitch : bool
        Whether to snap each row to a global staggered x pitch after solving.
    phase_fraction : float
        Alternating-row phase as a fraction of the inferred pitch.
    """

    node_sep: float = 18.0
    edge_weight: float = 3.0
    anchor_weight: float = 0.2
    iterations: int = 80
    snap_pitch: bool = True
    phase_fraction: float = 0.5


def infer_layers_from_y(pos: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    """Infer discrete row IDs from a position tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    decimals : int, default=3
        Decimal precision for grouping y-values.

    Returns
    -------
    torch.Tensor
        Integer row IDs with shape ``[N]``.
    """

    y_values = np.round(pos[:, 1].detach().cpu().numpy(), decimals)
    unique_values = sorted(float(value) for value in np.unique(y_values))
    lookup = {value: row for row, value in enumerate(unique_values)}
    return torch.tensor([lookup[float(value)] for value in y_values], dtype=torch.long)


def group_rows_by_x(layers: torch.Tensor, pos: torch.Tensor) -> List[List[int]]:
    """Build row groups ordered left-to-right.

    Parameters
    ----------
    layers : torch.Tensor
        Integer row IDs with shape ``[N]``.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    List[List[int]]
        Node IDs grouped by row and sorted by x-coordinate.
    """

    if layers.numel() == 0:
        return []
    row_count = int(layers.max().item()) + 1
    rows: List[List[int]] = [[] for _ in range(row_count)]
    for node_id, row_id in enumerate(layers.tolist()):
        rows[int(row_id)].append(int(node_id))
    for row in rows:
        row.sort(key=lambda node_id: (float(pos[node_id, 0].item()), node_id))
    return rows


def same_row_gap(
    left_node: int,
    right_node: int,
    node_sizes: torch.Tensor,
    node_sep: float,
) -> float:
    """Return minimum center-to-center x separation for adjacent row nodes.

    Parameters
    ----------
    left_node : int
        Left node ID.
    right_node : int
        Right node ID.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]``.
    node_sep : float
        Desired extra gap between node boxes.

    Returns
    -------
    float
        Required center separation.
    """

    left_width = float(node_sizes[left_node, 0].item())
    right_width = float(node_sizes[right_node, 0].item())
    return (left_width + right_width) / 2.0 + node_sep


def project_same_row_constraints(
    x_values: np.ndarray,
    rows: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    node_sep: float,
) -> None:
    """Project x-values onto dot-like same-row order constraints.

    Parameters
    ----------
    x_values : np.ndarray
        Mutable x coordinate array with shape ``[N]``.
    rows : Sequence[Sequence[int]]
        Ordered row groups.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]``.
    node_sep : float
        Desired extra gap between adjacent row nodes.
    """

    for row in rows:
        if len(row) < 2:
            continue
        for left_node, right_node in zip(row, row[1:]):
            gap = same_row_gap(left_node, right_node, node_sizes, node_sep)
            minimum_right = x_values[left_node] + gap
            if x_values[right_node] < minimum_right:
                x_values[right_node] = minimum_right
        row_center = float(np.mean([x_values[node] for node in row]))
        for node in row:
            x_values[node] -= row_center


def build_interrow_neighbors(
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
) -> List[List[int]]:
    """Build undirected neighbors for non-flat edges only.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        Integer row IDs with shape ``[N]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    List[List[int]]
        Neighbor list used by x-equality relaxation.
    """

    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for edge_offset in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_offset].item())
        target = int(edge_index[1, edge_offset].item())
        if int(layers[source].item()) == int(layers[target].item()):
            continue
        neighbors[source].append(target)
        neighbors[target].append(source)
    return neighbors


def solve_projected_x(
    source_x: np.ndarray,
    rows: Sequence[Sequence[int]],
    neighbors: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    config: DotMimicConfig,
) -> np.ndarray:
    """Solve a dot-like 1-D x relaxation with same-row projection.

    Parameters
    ----------
    source_x : np.ndarray
        Anchor x-values with shape ``[N]``.
    rows : Sequence[Sequence[int]]
        Ordered row groups.
    neighbors : Sequence[Sequence[int]]
        Inter-row neighbor lists.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]``.
    config : DotMimicConfig
        Solver parameters.

    Returns
    -------
    np.ndarray
        Candidate x-values with shape ``[N]``.
    """

    x_values = source_x.astype(np.float64).copy()
    x_values -= float(np.mean(x_values))
    project_same_row_constraints(x_values, rows, node_sizes, config.node_sep)

    for _ in range(config.iterations):
        relaxed = x_values.copy()
        for node, node_neighbors in enumerate(neighbors):
            if not node_neighbors:
                continue
            numerator = config.anchor_weight * source_x[node]
            denominator = config.anchor_weight
            for neighbor in node_neighbors:
                numerator += config.edge_weight * x_values[neighbor]
                denominator += config.edge_weight
            relaxed[node] = numerator / max(denominator, 1.0e-9)

        x_values = 0.65 * x_values + 0.35 * relaxed
        project_same_row_constraints(x_values, rows, node_sizes, config.node_sep)

    return x_values


def infer_pitch(
    x_values: np.ndarray,
    edge_index: torch.Tensor,
    layers: torch.Tensor,
) -> float:
    """Estimate a global x pitch from non-flat edge x deltas.

    Parameters
    ----------
    x_values : np.ndarray
        X coordinate array with shape ``[N]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        Integer row IDs with shape ``[N]``.

    Returns
    -------
    float
        Positive pitch estimate.
    """

    deltas: List[float] = []
    for edge_offset in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_offset].item())
        target = int(edge_index[1, edge_offset].item())
        if int(layers[source].item()) == int(layers[target].item()):
            continue
        delta = abs(float(x_values[target] - x_values[source]))
        if delta > 1.0e-6:
            deltas.append(delta)
    return float(np.median(deltas)) if deltas else 1.0


def snap_rows_to_pitch(
    x_values: np.ndarray,
    rows: Sequence[Sequence[int]],
    pitch: float,
    phase_fraction: float,
) -> np.ndarray:
    """Snap each row to a staggered global pitch while preserving row order.

    Parameters
    ----------
    x_values : np.ndarray
        Source x-values with shape ``[N]``.
    rows : Sequence[Sequence[int]]
        Ordered row groups.
    pitch : float
        Global x pitch.
    phase_fraction : float
        Alternating-row phase as a fraction of pitch.

    Returns
    -------
    np.ndarray
        Snapped x-values with shape ``[N]``.
    """

    if pitch <= 1.0e-6:
        return x_values.copy()
    snapped = x_values.copy()
    for row_index, row in enumerate(rows):
        if not row:
            continue
        phase = pitch * phase_fraction if row_index % 2 else 0.0
        raw_values = np.array([x_values[node] for node in row], dtype=np.float64)
        center = float(np.mean(raw_values))
        row_values = np.round((raw_values - phase) / pitch) * pitch + phase
        row_values += center - float(np.mean(row_values))
        row_values.sort()
        for node, value in zip(row, row_values.tolist()):
            snapped[node] = value
    return snapped


def materialize_candidate(
    source_pos: torch.Tensor,
    layers: torch.Tensor,
    x_values: np.ndarray,
) -> torch.Tensor:
    """Build a candidate tensor from solved x and quantized source y rows.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source position tensor with shape ``[N, 2]``.
    layers : torch.Tensor
        Integer row IDs with shape ``[N]``.
    x_values : np.ndarray
        Candidate x-values with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Candidate position tensor with shape ``[N, 2]``.
    """

    candidate = source_pos.detach().clone().to(dtype=torch.float32)
    candidate[:, 0] = torch.tensor(x_values, dtype=torch.float32)

    unique_y: List[float] = []
    for row_id in range(int(layers.max().item()) + 1):
        members = torch.nonzero(layers == row_id, as_tuple=False).flatten()
        unique_y.append(float(torch.median(source_pos[members, 1]).item()))
    for node, row_id in enumerate(layers.tolist()):
        candidate[node, 1] = unique_y[int(row_id)]

    candidate[:, 0] -= torch.mean(candidate[:, 0])
    candidate[:, 1] -= torch.mean(candidate[:, 1])
    return candidate


def dot_lattice_mimic_candidate(
    source_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    config: DotMimicConfig,
) -> torch.Tensor:
    """Generate one dot-like lattice mimic candidate.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]``.
    config : DotMimicConfig
        Candidate-generation parameters.

    Returns
    -------
    torch.Tensor
        Candidate position tensor with shape ``[N, 2]``.
    """

    layers = infer_layers_from_y(source_pos)
    rows = group_rows_by_x(layers, source_pos)
    source_x = source_pos[:, 0].detach().cpu().numpy().astype(np.float64)
    source_x -= float(np.mean(source_x))
    neighbors = build_interrow_neighbors(edge_index, layers, int(source_pos.shape[0]))

    solved_x = solve_projected_x(
        source_x=source_x,
        rows=rows,
        neighbors=neighbors,
        node_sizes=node_sizes,
        config=config,
    )
    if config.snap_pitch:
        pitch = infer_pitch(solved_x, edge_index, layers)
        solved_x = snap_rows_to_pitch(
            solved_x,
            rows,
            pitch=pitch,
            phase_fraction=config.phase_fraction,
        )
    return materialize_candidate(source_pos, layers, solved_x)


def best_dot_mimic_candidate(
    source_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn,
) -> Tuple[torch.Tensor, float, str]:
    """Choose the best dot-like candidate by a supplied metric scorer.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]``.
    score_fn : callable
        Function returning a higher-is-better score for a position tensor.

    Returns
    -------
    Tuple[torch.Tensor, float, str]
        Best position, best score, and candidate name.
    """

    best_pos = source_pos
    best_score = float(score_fn(source_pos))
    best_name = "baseline"
    for edge_weight in (1.5, 3.0):
        config = DotMimicConfig(edge_weight=edge_weight, anchor_weight=0.2)
        candidate = dot_lattice_mimic_candidate(source_pos, edge_index, node_sizes, config)
        score = float(score_fn(candidate))
        if score > best_score:
            best_pos = candidate
            best_score = score
            best_name = f"dot_mimic_edge_weight_{edge_weight}"
    return best_pos, best_score, best_name
```

## Empirical Validation

Command run:

```bash
python /tmp/dot_lattice_mimic.py \
  --graphs hexagonal_lattice_42 triangular_lattice_36 parallel_cycles_4x5 \
           grid_5x5 grid_rect_6x8 outerplanar_dag_20 \
           clustered_medium_5x20 sierpinski_42 \
  --output /tmp/dot_lattice_mimic_results.json
```

Scorer caveat: these are fast-proxy composite scores from `quick()` plus `composite()`, not the full benchmark scorer. They are valid for comparing the candidate to its baseline in this prototype because both use identical metric settings. They should not replace the prompt's seeded full benchmark numbers.

| Graph | Base score | Best candidate | Delta | Candidate name | Main metric movement |
|---|---:|---:|---:|---|---|
| `hexagonal_lattice_42` | 68.2254 | 68.2254 | +0.0000 | baseline | mimic rejected; baseline CV 0.5115, straightness 29.13 |
| `triangular_lattice_36` | 74.4635 | 74.4635 | +0.0000 | baseline | mimic rejected; baseline CV 0.2433, straightness 25.09 |
| `parallel_cycles_4x5` | 49.5318 | 49.5318 | +0.0000 | baseline | mimic rejected; target best is sfdp, not dot |
| `grid_5x5` | 76.9806 | 76.9806 | +0.0000 | baseline | mimic rejected; this protects a regular grid control |
| `grid_rect_6x8` | 73.4816 | 73.4816 | +0.0000 | baseline | mimic rejected; protects rectangular grid control |
| `outerplanar_dag_20` | 58.7229 | 63.6473 | +4.9244 | `yrows_source_lrnet_e3.0_a0.2` | straightness improved 49.64 -> 6.09, CV worsened 0.8139 -> 1.0702 |
| `clustered_medium_5x20` | 62.9526 | 63.1868 | +0.2342 | `yrows_source_lrnet_e3.0_a0.2` | small straightness gain, CV nearly unchanged |
| `sierpinski_42` | 69.5335 | 72.2920 | +2.7585 | `yrows_source_lrnet_e3.0_a0.2` | straightness improved 21.98 -> 1.06, CV worsened 0.5280 -> 0.6226 |

Target conclusion: the dot-like post-hoc x candidate is not the missing Area A fix. It failed all three requested targets. The controls are interesting: row-wise x relaxation can improve straightness on row-like DAG/fractal cases, but that is a different bet and carries regression risk because it can worsen CV while improving straightness.

Why this matters: the prior Sprint-21 uniform-grid attempt failed because it sacrificed straightness/dag consistency. This attempt failed for the opposite reason on the target lattices: it was conservative about row order and anchors, so it could not discover dot's x rows from dagua's saturated output. Dot's advantage is already baked into its rank/order/dummy-node state before x compaction starts.

## Recommended Algorithm If We Pursue Dot Matching

The right implementation is a metric-picked **layered coordinate candidate**, not a late polish primitive.

High-level plan:

1. Build or reuse a layered graph state:
   - run longest-path or existing native layer assignment;
   - insert dummy nodes for edges spanning more than one layer;
   - keep a trace from original edges to dummy chains.
2. Run the existing median/barycenter/transpose ordering on the expanded graph.
3. Run a proper x-coordinate assignment over the expanded graph:
   - same-row constraints: `x[v] - x[u] >= half_width(u) + half_width(v) + sep` for adjacent ordered nodes;
   - edge constraints: minimize `sum(weight_e * abs_or_squared(x[src] - x[tgt] - port_delta))`;
   - optional slack nodes or equivalent convex relaxation for edge-pair terms;
   - preserve original-node coordinates after removing dummy nodes.
4. Materialize only original node positions.
5. Feed the candidate into `_best_of_polish` or a similar metric picker with a margin and hard guards.

Implementation choices:

- Exact dot parity would require a network-simplex min-cost constraint solver. That is probably too much for one surgical sprint unless a small self-contained simplex implementation already exists.
- A pragmatic first implementation can use active-set projected relaxation or isotonic-like row projection, but it must start from the **expanded graph ordering**, not from final dagua positions.
- Dagua already has Brandes-Koepf coordinate machinery in `dagua/layout/ops/sugiyama.py`; do not duplicate it blindly. The missing piece is not "BK exists"; it is "use the right expanded graph/order and evaluate a dot-like compacted x candidate for these lattice/mesh classes."

## Integration Point

Recommended integration: new metric-picked coordinate candidate in the native layered path, before final polish or as one additional candidate inside the existing best-of-polish picker.

Do not add this as a generic postprocess that sees only `pos`. It needs:

- `layers`
- `ordering`
- expanded dummy graph or enough long-edge chain metadata
- `node_sizes`
- original edge traces
- score function and hard metric guards

Suggested local shape:

- `dagua/layout/ops/coordinate.py` or a new private helper near the native layered pipeline for the coordinate assignment.
- Candidate called from `dagua_native.py` only when the graph is a planar lattice/mesh DAG or an explicitly protected row-like DAG.
- Return a candidate tensor, not a committed mutation, so the existing picker can reject it.

## Risk / Regression Analysis

Specific protected graphs to verify:

- `hexagonal_lattice_42`: must beat current full benchmark by at least `+0.64` or this bet does not matter.
- `triangular_lattice_36`: must beat current full benchmark by at least `+1.62`, or at least move within tie range without regressing hex.
- `parallel_cycles_4x5`: do not expect dot mimic to solve it; protect against accidental fire. The best route is likely cyclic/sfdp-like, not layered dot.
- `grid_5x5` and `grid_rect_6x8`: dot-style row ladders look excellent on cached dot, but current dagua may already be protected elsewhere. Any coordinate candidate must be picker-safe.
- `sierpinski_42`: the prototype showed a large fast-proxy gain, but this is not the requested lattice bet and could be a metric-proxy artifact. Full scorer verification is required before touching this class.
- `outerplanar_dag_20`: prototype fast-proxy `+4.92` suggests row relaxation may be valuable, but it worsened CV while improving straightness. Full scorer and crossing checks are mandatory.
- `clustered_medium_5x20`: only `+0.23` under proxy; not enough to justify broad gating.
- Sprint-21 protected wins: `petersen_10`, `disconnected_encoder_residual`, `multi_component_80`, `deep_chain_20`, `linear_3layer_mlp`, `weighted_chain_20`, and `nested_shallow_enc_dec` must remain best/tied.

Hard guards before accepting:

- finite coordinates
- no new overlaps
- no drop in `dag_consistency`
- no material drop in `depth_spearman_rho`
- full-score improvement above the existing picker margin
- crossing-rate non-regression on full scorer
- skip cyclic multi-component graphs unless a separate cyclic route owns them

## Implementation Order

1. **Do not implement the post-hoc mimic from `/tmp` in dagua.** It fails the requested targets.
2. Add a tiny diagnostic test harness around cached positions for `hexagonal_lattice_42`, `triangular_lattice_36`, and `grid_rect_6x8` so future coordinate candidates can print row-width/step distributions.
3. Prototype an expanded-graph coordinate candidate using the existing Sugiyama dummy insertion and ordering state. First version may be active-set relaxation, not exact simplex.
4. Score only as a candidate against current baseline on:
   `hexagonal_lattice_42`, `triangular_lattice_36`, `grid_5x5`, `grid_rect_6x8`, `sierpinski_42`, `outerplanar_dag_20`, `clustered_medium_5x20`, `parallel_cycles_4x5`.
5. If and only if full scorer shows target gains with no protected regressions, wire it into the native picker behind a narrow gate:
   - exact planar or known generated grid/lattice tag;
   - directed acyclic;
   - connected for first pass;
   - row/layer count at least 5;
   - current gap attributable to row geometry or straightness, not cycle handling.
6. Do one deterministic 93-graph sweep before enabling by default.

## Concerns

- The current evidence says the "easy" Area A answer is negative. Matching dot means moving earlier in the layered pipeline, not adding another polish knob.
- The prompt asked for empirical validation with full composite deltas. I attempted full scoring first, but concurrent CPU-saturated Sprint-22 jobs made the candidate sweep impractical. I therefore used a reduced fast-proxy scorer for candidate search and clearly separated it from the official full benchmark numbers. Before implementation, rerun the exact candidate idea with `full()` on an idle machine.
- The row-relaxation gains on `outerplanar_dag_20` and `sierpinski_42` may be real, but they are not the target lattice fix. Treat them as separate leads.

## Knowledge To Remember

- dot's x positions come from network-simplex ranking of an auxiliary graph, not from BK/K-means/grid snap alone.
- Hex dot rows are mostly 72-point steps, but the nonuniform 90/108 gaps are important; perfect equal-edge layouts are not what the metric rewards.
- A dot mimic that only sees final dagua positions is too late for `hexagonal_lattice_42` and `triangular_lattice_36`.
- `parallel_cycles_4x5` should leave Area A. Its best competitor is sfdp, and dot's row pattern is not competitive enough.
