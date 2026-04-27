# Sprint-37 Quality Research: Angular-Resolution Loss Term -- Claude

## TL;DR

**SHIP** — but as a **re-tuning of the existing `FanoutDistributionLoss`**
(lower `degree_threshold` from 5 → 2, raise `w_fanout` 0.3 → ~0.8), gated
structurally on `avg_degree >= 3`. **Do NOT introduce a new loss op.** The
principled all-pairs incident-edge angular-separation term already lives in
`dagua/layout/constraints.py:fanout_distribution_loss`; it is wired through
`resolve.py:468` but its default config makes it a no-op on ~76% of the
93-graph suite.

## 1. Existing-op survey (the first move)

There is already exactly one principled angular-separation loss in dagua,
and it is already wired into the active pipeline:

| Layer | Symbol | File:line |
|---|---|---|
| Math | `fanout_distribution_loss(pos, edge_index, degree_threshold=5, ...)` | `dagua/layout/constraints.py:1766-1901` |
| Op wrapper | `FanoutDistributionLoss(LossOp)` (registered, weight_key `w_fanout`, default 0.3) | `dagua/layout/ops/loss_engine.py:1284-1331` |
| Op config | `FanoutDistributionLossConfig` (only field is implicit `degree_threshold=5`; the dataclass itself has no fields, so this isn't tunable from the op layer today) | `dagua/layout/ops/loss_engine.py:704-708` |
| Pipeline wiring (active) | `if config.w_fanout > 0.0: losses.append(FanoutDistributionLoss())` | `dagua/layout/resolve.py:468-469` |
| Schedule | `"w_fanout": lambda step, total_steps: self.config.w_fanout` (constant) | `dagua/layout/ops/anneal.py:1144` |
| Default weight | `w_fanout: float = 0.3` | `dagua/config.py:157`, `anneal.py:1059` |
| Default threshold | `degree_threshold: int = 5` (out-degree) | `constraints.py:1769`, `loss_engine.py:1324-1331` |

**Mechanism:** for every node with out-degree >= `degree_threshold`,
compute child angles `atan2(dy, dx)` around the hub, sort, take consecutive
angular gaps + the wraparound gap, and penalize `(gap - 2π/degree)²`
summed per-hub. This is the textbook "uniform angular spread around a hub"
formulation (Eades 1984, Tutte 1963; specifically the form used in `(SGD)²`
(Ahmed et al. 2020, §3.4 "Angular Resolution") and the Hu/Shi MaxEnt-stress
angular term). Differentiable everywhere except at degenerate co-located
children.

**Why it doesn't lift `angular_res_mean_deg` rank.** Three independent
reasons compound:

1. **Threshold mismatch with the metric.** `dagua.metrics.angular_resolution`
   measures the min angle between incident edges at every node with
   **degree >= 2** (undirected incidence). The loss only fires for
   **out-degree >= 5**. Only 13/81 dagua-OK graphs (16%) have
   `avg_degree >= 5`. On the median graph the loss is identically zero by
   construction.

2. **Directionality mismatch.** The loss looks at **outgoing** edges only.
   The metric looks at the **undirected incidence**. A node with in-degree 4
   and out-degree 1 contributes `min(angle)` to the metric but is invisible
   to the loss.

3. **Weight is too low to compete.** Even when the loss does fire,
   `w_fanout = 0.3` is dwarfed by `w_dag = 10`, `w_length_variance = 8.0`,
   `w_overlap = 5.0`, `w_attract = 2.0`, `w_straightness = 2.2`. In the
   equilibrium gradient sum, fanout contributes < 5% of the force.

## 2. Loss formulation (use the existing one, two principled changes)

**No new loss math is needed.** The mathematical form already in
`fanout_distribution_loss` matches the standard angular-resolution proxy used
by `(SGD)²` (Ahmed et al., GD 2020, §3.4) and Hu & Shi (WIREs 2015, §4.2).

```
L_angular(v) = (1/deg(v)) * Σ_i (gap_i(v) - 2π/deg(v))^2
   where deg(v) = number of incident edges,
         gap_i(v) = sorted consecutive angular gaps at v (incl. wraparound)
L_total = mean over hubs v with deg(v) >= K_min of L_angular(v)
```

**Two principled changes to the call site:**

1. **Switch from out-degree to undirected incidence** to match the metric.
   Same math, now the loss sees both endpoints of every edge.
2. **Drop the threshold from 5 to 2** — every degree-2+ node participates.
   This is principled: "uniform angular spread around v" is well-defined for
   any deg(v) >= 2 (deg=2 = "edges should leave on opposite sides").
3. **Raise the weight** `w_fanout: 0.3 → 0.8` and add a late-phase ramp
   (mirror `w_straightness`: `w_fanout * (1.0 + 0.5 * progress)`) so it
   grows alongside straightness during convergence rather than fighting
   attraction in the early force-build-out phase.

## 3. Pipeline insertion point

No new wiring; only edits to existing call sites.

| Change | File:line | What changes |
|---|---|---|
| Add fields to existing config dataclass | `dagua/layout/ops/loss_engine.py:704-708` (`FanoutDistributionLossConfig`) | Add `degree_threshold: int = 2` and `symmetrize: bool = True` |
| Pass them through | `dagua/layout/ops/loss_engine.py:1322-1331` (`FanoutDistributionLoss.evaluate`) | `degree_threshold=self.config.degree_threshold`, build a symmetrized `edge_index` when `self.config.symmetrize` |
| Update default weight + late ramp | `dagua/layout/ops/anneal.py:1059` and `anneal.py:1144` | `w_fanout: float = 0.8`; replace constant lambda with `self.config.w_fanout * (1.0 + 0.5 * progress)` |
| Mirror in user-facing config | `dagua/config.py:157` | `w_fanout: float = 0.8` |
| Class gate (structural) | `dagua/layout/resolve.py:468-469` | `if config.w_fanout > 0.0 and _avg_degree(problem) >= 3.0:` |

## 4. Class predicate (the structural gate)

**Predicate:** `avg_degree(g) = 2 * |E| / |N| >= 3.0`, computed on the
original graph. Optional secondary: `max_degree(g) >= 4`.

Population over 81 dagua-OK graphs (from `eval_output/benchmark_full/results.json`):

| avg_deg bucket | count | examples |
|---|---:|---|
| < 3.0 | 44 | binary_tree, deep_chain_20, transformer_layer, etc. |
| [3.0, 4.0) | 17 | center_port_backedge_hub, sierpinski_42, hub_and_spoke_3x20, small_world_100 |
| [4.0, 5.0) | 7 | bipartite_4_3_4, real_karate_34, regular_4_40 |
| [5.0, 7.0) | 7 | densenet_block, dependency_graph_100, planar_60, scale_free_ba_120 |
| [7.0, 12.0) | 5 | complete_bipartite_8x12, dense_pair_50, real_lesmis_77, real_football_115 |
| [12.0+) | 1 | rgg_100 |

37 of 81 graphs (45.7%) hit the gate. Protected wins (deep_chain_20,
random_dag_200, ba_500, org_chart_deep, hub_fanout_label_skew) all sit
**below** `avg_degree=3` -- gate guarantees they are untouched
(verify hub_fanout_label_skew empirically).

## 5. Empirical predictions

Per-class lift on `angular_res_mean_deg`:

| Graph class | n graphs | Expected angular Δ |
|---|---:|---:|
| [12.0+) rgg_100 | 1 | +15-25° |
| [7.0, 12.0) | 5 | +10-20° |
| [5.0, 7.0) | 7 | +8-15° |
| [4.0, 5.0) | 7 | +6-12° |
| [3.0, 4.0) | 17 | +3-8° |
| < 3.0 | 44 | 0° (gate off) |

Aggregate prediction: angular rank **6.40 → 4.5-5.2**.

Cost on `edge_straightness_mean_deg`: 2.07 → **2.4-2.7** (Pareto trade).
Cost on `edge_length_cv`: 3.94 → **~4.5**.
Cost on `crossing_rate`, `dag_consistency`, `depth_rho`: ~neutral.

Composite mean rank prediction: **1.21 → 1.25-1.40**, well below 1.5 gate.

## 6. Validation gate predictions

| Gate | Target | Predicted | Pass? |
|---|---|---|---|
| Angular rank improvement >= 1.0 | 6.40 → ≤5.40 | 6.40 → 4.5-5.2 | YES |
| Composite mean rank < 1.5 | 1.21 → <1.5 | 1.21 → 1.25-1.40 | YES |
| Protected wins ±0.5 | 5 graphs | All have avg_deg < 3 (verify hub_fanout) | LIKELY YES |
| Jitter angular mean > 0 | 3 high-deg graphs | Convex around equal-spread minimum | YES |
| Jitter composite min > -2.0 | Same | Worst case is straightness/cv hit | LIKELY YES |

## 7. Risks

1. **edge_length_cv regression cascades.** If the new fanout pressure
   forces equilibrium where length-CV gradient flips sign, CV could go up
   more than predicted. Mitigation: keep ramp on fanout. Detection:
   if probe shows Δ edge_length_cv rank > +1.0, revert weight to 0.5.

2. **degree_threshold=2 on tree-shaped subgraphs.** A degree-2 node with
   two near-collinear edges may get pushed toward 180° split. Crossing/DAG
   weights win, but check `dag_consistency` and `depth_rho`.

3. **Undirected symmetrization changes which graphs the loss sees.** A
   deg-1-out / deg-7-in node was invisible; after symmetrize it's a deg-8
   hub. Intended fix; protected wins double-protected by the gate.

4. **Looks like a tuning sprint** -- the retro warns about "ratchet"
   sprints. Counter: changes 4 numeric defaults and one structural gate.
   None keyed on graph identity. Math unchanged. Output ~15 lines of diff.

5. **"No fix found" outcome.** If probe shows angular lift < 0.7 OR
   composite > 1.5, the honest verdict is no-ship. Don't bargain by
   tightening the gate to chase the test suite.

## Bottom line

The principled angular-separation term already exists, is wired in, and is
dormant by misconfiguration. The proposed fix is to re-tune three numeric
defaults and add one structural gate, justified per-change by the metric's
own definition (degree >= 2, undirected) and a refereed paper formulation
(Ahmed et al. (SGD)² §3.4). Predicted angular rank: 6.40 → 4.5-5.2,
composite rank: 1.21 → 1.25-1.40, all gates passing. If the live probe
disagrees, no-fix is the honest outcome.
