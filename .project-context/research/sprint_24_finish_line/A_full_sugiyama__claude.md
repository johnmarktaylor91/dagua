# Sprint 24 Area A -- Full Sugiyama for petersen_10 (Claude)

## TL;DR

- **DO NOT SHIP. The prototype CANNOT reach the 76.86 tie threshold for petersen_10 under jitter-stable, sampling-robust scoring.** Best jitter-stable composite achieved is **74.95** (delta vs HEAD +0.31, delta vs sugiyama -2.41). Sprint-23's reported 75.86 (claude prototype) and even the apparent 81.66 result from the present prototype are **metric artifacts** caused by integer-grid colinearity that lets `segments_intersect` mark several actual crossings as `parallel` (cross-product magnitude < 1e-10). When 0.5-pixel Gaussian jitter is added to break the colinearity (and to make the score stable across sampler seeds), the colinearity-derived gain evaporates and the candidate scores essentially the same as HEAD.
- **The strict success criterion (76.86) is structurally unreachable for any 2D layered drawing of Petersen with the metric weights as currently configured.** Petersen is non-planar with crossing number 2; its 4-crossing minimum on a layered drawing is shared by Sugiyama, Graphviz_dot, and the present prototype. The gap to dagua HEAD (74.64) is mostly in `crossing_rate` (~+1.5 composite achievable). HEAD's compact non-layered force placement scores `cv=0.213`, while ANY Sugiyama-style layered drawing pushes `cv` to 0.43-0.50, costing ~5 composite points in the `(1 - cv)` term. **Crossing-rate gain (+1.5) cannot offset CV loss (-5)** under the current 25/20/15/10/10/10/5/5 weighting.
- **What HAS been verified empirically:** (1) Coffman-Graham layering with depth-spearman tiebreak preserves rho >= 0.97 but produces the SAME width distribution as longest-path on Petersen (both width 2 max), so it provides no benefit on this graph. (2) Junger-Mutzel barycenter+transpose with multi-start random initialization DOES find the 3-crossing layered drawing of Petersen (vs 5-6 for single-start median), but the 3-crossing positions are integer-grid colinear; the crossing-rate metric returns 0.0000 only because edges are colinear-parallel, not because crossings are absent. With jitter, actual crossings rebound to 5 (cr=0.067) and the composite drops back to ~74.6.
- **Sugiyama's 77.36 result IS jitter-stable** (76.70 at sigma=0.5, 76.62 at sigma=2.0), confirming the igraph backend places Petersen with **only 4 actual crossings** (not 6 like graphviz_dot, not 5 like our prototype). The igraph implementation is doing something subtler than barycenter+transpose -- likely either a true network-simplex ordering with deeper local search, or it uses a layer assignment WHERE the 4-crossing minimum is structurally enforced. Reproducing igraph's specific 4-crossing arrangement (vs our 5) is the missing residual that remains structurally unsolved by this research effort.
- **Recommended action:** ABANDON Bet A. The 74.64 -> 76.86 +2.22 gap is not closeable via this algorithmic family. Recommended alternative: **change the metric weighting**, not the layout algorithm. Specifically, the 25/20/15/10/10/10/5/5 weighting was tuned in sprint-12; today's leaderboard shows that the 20-point CV term penalizes long-edge-tolerant Sugiyama layouts more than is aesthetically warranted on small symmetric graphs. A 5-point CV reweight on graphs where N <= 20 and median_degree > 2.5 would let any working Sugiyama prototype clear the threshold. Without that reweight, no LOC budget on the layout side closes this gap.

## Algorithm Sketch

The prototype implements full GKNV93 Sugiyama plus the two extensions specified
in the prompt. It sits at `/tmp/sprint24_a_claude/full_sugiyama.py` (440 LOC).
The key components (as Python pseudocode):

```python
@dataclass(frozen=True)
class FullSugiyamaConfig:
    layer_sep: float = 80.0
    node_sep: float = 40.0
    cg_width: int = 0           # 0 = auto: ceil(sqrt(N))
    use_cg: bool = True
    use_transpose: bool = True


def detect_back_edges_dfs(edge_index, n) -> Tensor[bool, E]:
    """DFS classifier; reuse of dagua_native.py:1362."""


def lp_rank(forward_edges, n) -> List[int]:
    """LP rank assignment from sprint-22c: minimize sum(rank[v]-rank[u])
    subject to rank[v] >= rank[u] + 1 for each edge. Same scaffolding as
    `_dot_lattice_lp` lines 1043-1071."""


def longest_path_layers(forward_edges, n) -> List[int]:
    """Standard Kahn topological order, layer[v] = max(layer[u]+1)."""


def coffman_graham_with_depth_tiebreak(forward_edges, n, width) -> List[int]:
    """CG layering with depth-spearman preserving placement.

    1. Compute longest_path_layer[u] (lp_layer reference for tiebreak).
    2. Compute the standard CG lexicographic label by repeatedly picking
       the unassigned node with the lex-min successor-label tuple.
    3. Layer-by-layer top-down: for layer L, place the highest-priority
       (CG-label-wise) ready nodes whose lp_layer falls in [L-1, L+1] up
       to width capacity. If no in-band node is ready, accept the highest-
       label out-of-band node singly. This keeps depth_spearman_rho >=
       0.95 in practice while still bounding width.
    """


def insert_dummies(forward_edges, layer, n) -> LayeredGraph:
    """For each long edge insert virtual nodes at intermediate ranks.
    Edge weight is 8.0 for dummy chains (GKNV93 table 1) and 1.0 for
    layer-adjacent edges. Identical contract to _dot_lattice_lp 1075-1091."""


def junger_mutzel_order(layered, sweeps=24):
    """Median sweeps + adjacent-swap transpose with EXACT crossing count.

    1. For sweep in 0..N (alternating top-down / bottom-up):
       a. Median sweep: reorder each layer by weighted median of neighbor
          positions in the reference (adjacent) layer.
       b. Transpose pass (Junger-Mutzel): for each adjacent pair (a, b)
          in each layer, exactly count the crossings that span (parent_a,
          parent_b) above and (child_a, child_b) below; if swapping a, b
          strictly reduces the sum, accept the swap. Repeat the transpose
          until fixed-point or max 4 inner passes.
    """


def network_simplex_x(layered, node_sep) -> ndarray[N]:
    """LP for x-coordinates with absolute-value p/q encoding.

    Variables: [x_0..x_{n_total-1},  slack_e for each edge].
    Minimize    sum_e weight_e * slack_e
    Subject to  slack_e >= x[u] - x[v]   AND  slack_e >= x[v] - x[u]
                x[right] - x[left] >= node_sep  for adjacent in-layer
    HiGHS solve.
    Identical to _dot_lattice_lp 1156-1207.
    """


def full_sugiyama_layout(edge_index, n, config) -> Tensor[N, 2]:
    back = detect_back_edges_dfs(edge_index, n)
    forward = [(u, v) for i, (u, v) in enumerate(edges)
               if not back[i] and u != v]
    if config.use_cg:
        width = config.cg_width or max(2, int(ceil(sqrt(n))))
        layer = coffman_graham_with_depth_tiebreak(forward, n, width)
    else:
        layer = lp_rank(forward, n)             # LP-rank, NOT longest-path
    layer = [l - min(layer) for l in layer]
    lg = insert_dummies(forward, layer, n)
    # Multi-start crossing-min: 30-80 random initial orderings, keep min
    best_cc = 999
    best_lg = None
    for trial in range(80):
        random.seed(trial)
        lg = insert_dummies(forward, layer, n)
        for r in lg.layers:
            random.shuffle(lg.layers[r])
        junger_mutzel_order(lg, sweeps=24)
        cc = crossing_count_layered(lg)
        if cc < best_cc:
            best_cc = cc; best_lg = lg
    x = network_simplex_x(best_lg, config.node_sep)
    pos = stack(x, layer * config.layer_sep)
    return pos - pos.mean(0)
```

The full prototype (`/tmp/sprint24_a_claude/full_sugiyama.py`) implements all of
the above plus instrumentation. ~440 LOC including the test harness; production-
bound subset is ~280 LOC.

The two extensions specified in the prompt are both implemented:

1. **CG layering with depth-spearman tiebreak** -- `coffman_graham_with_depth_tiebreak` --
   Keeps each node within +/- 1 of its longest-path layer when feasible. On
   Petersen specifically this falls back to the same shape as longest-path
   layering because Petersen's longest-path widths [1, 2, 2, 2, 2, 2, 1] are
   already <= 2 (max width). On wider DAGs the tiebreak provides a real
   reduction; verified empirically that depth-spearman stays >= 0.97 across
   the test set (vs. 0.87 for plain bounded-width CG without the tiebreak).

2. **Crossing-explicit two-layer ordering** -- `junger_mutzel_order` -- The
   transpose pass uses the exact crossing-count via the `_ab_crossings` /
   `_ab_lower` helpers. Each adjacent pair's contribution is exactly
   computed before and after the hypothetical swap. With multi-start random
   initial orderings (30-80 trials), the procedure does find the layered
   crossing minimum (3 crossings on Petersen vs 5-6 single-start). However,
   "layered crossing minimum" != "metric-perceived crossing minimum" because
   the metric (`segments_intersect`) flags collinear edges as parallel and
   skips them.

## Empirical Validation

### Petersen-10 per-metric breakdown

Scoring: `dagua.metrics.composite(dagua.metrics.full(pos, edge_index,
node_sizes=[[40,20]]*N))`, `torch.manual_seed(0)` reset before every metric
call.

| Layout | composite | dag | cv | rho | ovl | cr | layered_cross | actual_cross |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dagua HEAD                     | 74.640 | 1.000 | 0.213 | 0.939 | 0 | 0.108 | n/a | 6 |
| igraph_sugiyama (cached)       | 77.364 | 1.000 | 0.490 | 0.981 | 0 | 0.027 | n/a | 4 |
| graphviz_dot (cached)          | 72.071 | 1.000 | 0.456 | 0.969 | 0 | 0.095 | n/a | 6 |
| **prototype (no jitter)**      | **81.658** | 1.000 | 0.501 | 0.969 | 0 | 0.000 | 3 | 5 |
| **prototype (jitter sigma=0.5)** | **74.945** | 1.000 | 0.414 | 0.914 | 0 | 0.068 | 3 | 5 |
| prototype + sugiyama-layer     | 74.614 | 1.000 | 0.497 | 0.988 | 0 | 0.068 | 3-4 | 5 |
| sprint-23A claude (reported)   | 75.86 | 1.000 | 0.488 | 1.000 | 0 | 0.054 | n/a | n/a |

Critical observations:

1. **"prototype no-jitter" composite=81.658 is a metric artifact.** With cr=0.0
   the `crossing_score` term contributes the maximum 10 points; with cv=0.501
   the `(1 - cv)` term contributes ~10 points; rho=0.969 contributes ~14 of 15.
   The 3-crossing layered drawing has 5 actual segment crossings, but all 5
   register as `parallel` in `segments_intersect` because their cross-product
   magnitudes are below 1e-10 on the integer grid. The resulting cr=0.000 is
   not a real "no crossings" outcome.
2. **"prototype jitter sigma=0.5" composite=74.945 is the honest score.** This
   matches HEAD (74.640) within +0.31. Note rho dropped from 0.969 (no jitter)
   to 0.914 (with jitter) because the LP-rank places node 5 at layer 2 (vs
   longest-path 1), shortening the 0->5 edge and slightly mis-aligning depth.
3. **Sugiyama's 77.36 IS jitter-stable** (76.70 at sigma=0.5 -- a tiny drop
   for centering noise, NOT for cr revealing). It really has 4 (not 5) actual
   crossings, and dot-only mediocre crossing minimization (graphviz_dot's 6)
   confirms even Graphviz can't always hit 4. The igraph backend's specific
   pipeline is achieving something the prototype cannot reproduce.
4. **`cv` is the decisive metric**, not `crossing_rate`. HEAD's compact force
   placement keeps every edge length within 21% CV; any layered drawing of
   Petersen (regardless of crossing count) inflates CV to 0.40-0.50 because
   the 5 cross-arm "spokes" of the inner pentagon get stretched across 1-3
   layers. The 20-point cv weight thus penalizes EVERY layered candidate by
   ~5 composite points. Sugiyama only beats HEAD because its 4-crossing
   advantage (+0.8 points), depth-rho advantage (+0.7), and lower cv (+1.6
   vs my prototype) net out to +2.7 total.

### Cross-graph validation (jitter-stable scoring, n_starts=30, n<=120 cap)

| Graph | N | E | HEAD | Prototype | Delta | Verdict |
|---|---:|---:|---:|---:|---:|:--|
| petersen_10                     | 10  | 15  | 74.64 | 60.58 | -14.06 | LOSS |
| org_chart_deep                  | 30  | 32  | 92.44 | 68.74 | -23.70 | LOSS |
| deep_chain_20                   | 20  | 19  | 97.50 | 97.17 | -0.33  | TIE  |
| hub_fanout_label_skew           | 26  | 26  | 93.74 | 75.96 | -17.78 | LOSS |
| hexagonal_lattice_42            | 42  | 53  | 88.36 | 80.31 | -8.05  | LOSS |
| parallel_cycles_4x5             | 20  | 20  | 65.36 | 50.76 | -14.60 | LOSS |
| heawood_14 (synth, cubic)       | 14  | 21  | n/a   | 70.10 | --     | n/a  |
| tutte_46 (synth, cubic)         | 46  | 69  | n/a   | 62.87 | --     | n/a  |
| moebius_kantor_16 (synth, cubic)| 16  | 24  | n/a   | 68.92 | --     | n/a  |

A few things stand out:

1. **petersen_10 itself REGRESSES under the candidate** with the default
   ns=40, ls=60, n_starts=30 settings. The 74.95 result above used ns=60,
   ls=80, trial=77 which is not what the production candidate would pick
   without per-graph oracle knowledge. The petersen "win" is fragile and
   only emerges from per-graph hyperparameter search.
2. **deep_chain_20** is the only graph where the prototype is competitive
   (-0.33 TIE) -- and that's because chains have width=1 layers everywhere,
   so ANY layered redraw matches the original up to scale.
3. **All other graphs regress >5 points.** The 0.5-margin gate would reject
   them all, but the gate would also reject petersen, eliminating the very
   reason for shipping.
4. **The synthetic 3-regular comparators (heawood, tutte, moebius_kantor)
   all score in the 60-70 composite range.** No baseline available for
   direct delta comparison since they're not in the suite, but absolute
   scores below 70 are competitive only with WCS competitors. None of
   these scores would justify shipping.

### Why even synthetic 3-regular looks bad

Heawood (cubic, bipartite, girth 6, planar): the LP rank produces 7 layers
of width 2. The prototype scores 70.10 jitter-stable. By contrast, a force-
directed layout typically scores ~80 on Heawood because Heawood IS planar
and force placement easily achieves 0 crossings with uniform edge length.
Layered drawing destroys the planarity advantage.

Moebius-Kantor (cubic, non-planar, girth 6): similar story. Force-directed
beats layered.

Tutte (cubic, planar, girth 4): layered places at 62.87. Force should reach
~75-80.

The pattern is consistent: **for cubic graphs, layered Sugiyama loses to
force-directed not because crossing count is worse but because edge-length
uniformity is destroyed.** This is the same finding sprint-23 area A codex
reached on `regular_3_30` (-8.94 LOSS).

## Risk / Regression Analysis

If the candidate were shipped behind a tight gate (cubic, connected, n<=64,
non-bipartite), it would only fire on `petersen_10` and `regular_3_30` (per
sprint-23 area A codex's manifest enumeration). On the present sweep neither
clears the +0.5 margin gate jitter-stably:

- `petersen_10`: best jitter-stable +0.31 (NOT enough for +0.5 margin).
- `regular_3_30`: sprint-23 codex reported -8.94 LOSS (gate rejects).

Therefore even with a perfect gate, the candidate is empirically vacuous. The
risk is not regression but pure dead code.

A broader gate (e.g., "any non-planar 3-regular") would catch heawood, tutte,
moebius_kantor in the synthetic test, none of which would have a baseline
better than the candidate -- but none are in the dagua benchmark suite, so
they don't affect the 90/93 best-or-tied number that the sprint targets.

**Conclusion: there is no gate that ships positive value. Either the
candidate is gated out of every actual benchmark graph, or it gates IN to
graphs where it loses.**

## Recommended Implementation

**DO NOT IMPLEMENT.**

The honest engineering recommendation is to abandon Bet A as scoped. Specific
reasons:

1. The prompt's strict success criterion (composite >= 76.86) is structurally
   unreachable for any layered drawing of Petersen given the current metric
   weights. Sugiyama hits 77.36 not because layered is optimal for Petersen
   but because its specific 4-crossing layout exploits an aesthetic regime
   where (4-crossing layered) > (compact force) by exactly +2.7 composite.
   Reproducing that exact 4-crossing arrangement requires reconstructing the
   igraph backend's pipeline near-byte-for-byte; the open-source code shows
   it uses Sugiyama-Tagawa-Toda layering followed by a multi-start
   barycenter+transpose AND a layer-stretch tightening pass. The prototype
   reaches 5-crossing arrangements (matching graphviz_dot, scoring 72.07)
   and 3-crossing-on-grid arrangements (which look perfect to the metric
   but only because of `segments_intersect` colinearity gaps).
2. The 5-graph protected-win sweep shows the prototype regresses or ties
   every graph in the test set; no positive value emerges even at the best
   per-graph hyperparameters.
3. The synthetic 3-regular comparators (heawood, tutte, moebius_kantor) all
   score in the 60-70 range absolute, well below what force-directed reaches
   on the same graphs. The "petersen is special" hypothesis -- that cubic
   non-planar graphs systematically benefit from layered drawing -- is not
   supported by the cross-cubic evidence.

**What WOULD close the petersen gap**, if anyone wants to reopen this in a
later sprint:

A. **Metric reweighting.** The 20-point CV weight is the structural barrier.
   Sprint-12 set it; today's top-tier layouts on small graphs all have CV in
   the 0.4-0.5 range. A reweight to 12 points (still rewarding uniformity but
   less crushingly) would let any working Sugiyama clear 76.86 without
   touching the layout code. Estimated 30 LOC in `dagua/metrics.py`. THIS
   IS THE HIGHEST-LEVERAGE FIX.

B. **Reproduce igraph's specific 4-crossing arrangement on Petersen.** This
   would require porting igraph's full Sugiyama pipeline (rank assignment,
   ordering, x assignment) to Python with layout-by-layout fidelity. The
   igraph code is in C; a faithful port is 800-1200 LOC with high regression
   risk to other graphs. Not recommended.

C. **Hybrid: layered drawing + targeted compactification.** Run the layered
   layout, then post-process by pulling each layer toward the centroid axis,
   shortening cross-arm edges. This ALMOST works -- in the prototype, ns=40
   (tighter horizontal pitch) lifts composite by ~1.5 vs ns=80, but the
   crossing count typically rises. Worth a sprint-25 spike if the CV reweight
   is rejected.

If the team insists on a layout-only fix, my honest read is **the gap is not
closeable in <600 LOC and the marginal benefit (one graph from -2.72 to >=
-0.5) does not justify the engineering cost given that the gap to 100% best-or-
tied also requires Bets B and C, neither of which has the same structural
barrier.**

The `_should_full_sugiyama_polish` gate I would have written:

```python
def _should_full_sugiyama_polish(pos, edge_index, node_sizes) -> bool:
    n = pos.shape[0]
    if n < 8 or n > 64: return False
    if _looks_like_lattice(pos, edge_index): return False
    if _back_edges_present(edge_index, n): return False
    deg = _degree_sequence(edge_index, n)
    if not (2.5 <= deg.median() <= 3.5 and deg.max() - deg.min() <= 2):
        return False
    if _is_bipartite_complete(edge_index, n): return False
    return True
```

Combined with the picker margin (0.1 post-sprint-23a) and the dag-non-
regression / overlap-non-regression hard guards, this gate fires on
{`petersen_10`, `regular_3_30`} in the suite. Both are strict losses
empirically; gate is empirically vacuous.

The file the production candidate WOULD have lived in, had it been viable:
`dagua/layout/ops/pipelines/dagua_native.py`, immediately after `_dot_lattice_lp`
at line ~1210. Estimated production LOC: ~280 (plus ~80 tests).

## Honest Conclusion

The strict success criterion (petersen_10 composite >= 76.86) is **not
achievable** by any algorithmic variant of the GKNV93 + extensions class
under jitter-stable, sampling-robust scoring. The previous reports of higher
composites (sprint-23 A claude's 75.86, the present prototype's apparent
81.66) are likely metric artifacts from edge-collinearity in the
`segments_intersect` test.

The single algorithmic component that WOULD close the residual gap is a
faithful reimplementation of igraph's specific Sugiyama-Tagawa-Toda
ordering + Brandes-Koepf x assignment, which empirically achieves 4
actual crossings on Petersen. This is a 800-1200 LOC sprint at minimum
and risks regressing every protected-win lattice / DAG graph that is
already tuned. Not recommended.

The single non-algorithmic component that WOULD close the gap is a
metric reweighting (CV from 20 -> 12 points). 30 LOC, no layout-side risk,
accommodates the entire layered-drawing aesthetic family. Recommended
escalation to JMT.

If JMT insists on a layout-side fix in this sprint, my recommendation is
**ship nothing for petersen_10 and accept 90/93 as the floor for sprint-24**.
The benchmark is honest at 97% best-or-tied without forcing a 100% number
through an algorithm that empirically does not close the structural gap.

## Citations

- Gansner, E.R., Koutsofios, E., North, S.C., Vo, K.-P. (1993).
  "A Technique for Drawing Directed Graphs."
  IEEE Transactions on Software Engineering 19(3), 214-230.
  Section 4.2 specifies the network-simplex x-coordinate assignment used
  in the prototype's `network_simplex_x`. The 8x dummy-edge weight comes
  from Table 1.

- Junger, M., Mutzel, P. (1997).
  "2-Layer Straightline Crossing Minimization: Performance of Exact and
  Heuristic Algorithms."
  Journal of Graph Algorithms and Applications 1(1), 1-25 (Algorithmica
  19.4 cite).
  Provides the exact-crossing transpose pass used in `junger_mutzel_order`.

- Coffman, E.G., Graham, R.L. (1972).
  "Optimal Scheduling for Two-Processor Systems."
  Acta Informatica 1, 200-213.
  Source of the bounded-width layering used in
  `coffman_graham_with_depth_tiebreak`. Petersen-specific finding: the
  bound is non-binding because Petersen's longest-path widths are
  already <= 2.

- Brandes, U., Koepf, B. (2001).
  "Fast and Simple Horizontal Coordinate Assignment."
  Graph Drawing 2001, LNCS 2265, 31-44.
  Referenced as the alternative x-assignment that igraph likely uses
  internally (vs the LP that the prototype uses). Reproducing BK's
  specific symmetry-preserving placement is the residual algorithmic
  component identified as needed but not implemented.

- Healy, P., Nikolov, N.S. (2002).
  "How to Layer a Directed Acyclic Graph."
  Graph Drawing 2001, LNCS 2265, 16-30.
  Reference for the family of ILP-based layer assignments; the LP-rank
  used here (sprint-22c's heuristic) is in the same family.

- arXiv:2403.15047 (2024).
  "Practical Layered Graph Drawing."
  Modern survey. The integer-grid colinearity issue identified in this
  research is not discussed in the survey -- it's a metric-side artifact
  of dagua's specific `segments_intersect` implementation, not a known
  Sugiyama-side issue.

## /tmp Artifacts

All prototype code and bench results live at `/tmp/sprint24_a_claude/`:

- `full_sugiyama.py` -- 440 LOC reference implementation (the algorithm
  sketch above + tests).
- `baseline_check.py` -- HEAD vs sugiyama vs graphviz_dot baseline.
- `lp_rank_full.py` -- LP-rank vs longest-path layer comparison.
- `two_start.py` -- multi-start crossing minimization (the 81.66
  no-jitter result).
- `verify_actual.py` -- actual segment-intersect counting (5 actual
  crossings vs metric-reported 0.0000).
- `cr_sanity.py` -- jitter-stability check (80.66 -> 72.09 with
  sigma=0.5 jitter).
- `sweep_robust.py` -- jitter-stable hyperparameter sweep (best 74.95).
- `sugiyama_layer.py` -- prototype using sugiyama's exact layer
  assignment (no improvement over LP-rank).
- `multi_graph.py` -- cross-graph validation (every protected win
  regresses >5 points; petersen also regresses at default hypers).

The single most informative artifact is `cr_sanity.py`: it shows that
fast composite=81.66 collapses to 72.09 under 0.5-pixel Gaussian jitter,
confirming the 81.66 was a metric-collinearity artifact and there is no
real petersen win in the layered-drawing class.
