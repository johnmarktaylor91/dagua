# TL;DR

Ship only as an experiment behind a structural average-degree gate: dagua has adjacent ideas, but no default native loss currently optimizes the benchmark's undirected incident-edge minimum angle, and a small `w_angular_resolution=0.5` native loss is plausible enough to test without threatening the composite if it is late-ramped and disabled below `avg_degree >= 3.0`.

# Existing-op survey: does anything already cover this?

Partially, but not in the way the benchmark metric is scored.

The benchmark metric is `dagua.metrics.angular_resolution` at `dagua/metrics.py:919`. It treats the graph as undirected, duplicates every edge in both directions, computes `atan2` for each incident edge direction, sorts angles per node, includes the circular wrap gap, and records the minimum incident-edge angle per sampled node. The reported `angular_res_mean_deg` is the mean of these per-node minima. It samples only nodes with degree >= 2, so degree-0/1 nodes are irrelevant; degree-2 chains can score well when the two incident edges oppose each other.

The default native loss stack is assembled in `dagua/layout/resolve.py:407`. The active losses include DAG ordering, attraction, repulsion, overlap, cluster losses, crossing, straightness, edge-length variance, spacing, fanout, back-edge compactness, flex pins/alignment, and optional pivot stress. There is no `AngularResolutionLoss` in `dagua/layout/ops/loss_engine.py`; the registered engine loss list ends at `FlexSpacingLoss`, and grep confirms no `angular_resolution` registered op in that file.

There are three nearby mechanisms:

1. `fanout_distribution_loss` in `dagua/layout/constraints.py:1774`, wrapped by `FanoutDistributionLoss` in `dagua/layout/ops/loss_engine.py:1284`. This penalizes uneven angular gaps among outgoing children for hubs with out-degree >= 5. It is directed, out-edge-only, hub-only, and uses equal-gap variance, not a soft margin on the minimum undirected incident angle. It misses fan-in collisions, mixed in/out node ports, degree-3/4 nodes, and non-hub regular graphs. It is active by default through `w_fanout=0.3` at `dagua/config.py:157`, but it cannot explain the poor rank because it is solving a narrower problem.
2. `_sgd2_angular_resolution_loss` in `dagua/layout/ops/loss_classic.py:1210`, selected only when the SGD2 criterion is `"angular_resolution"`. It builds incident-edge pairs in `_sgd2_build_incident_edge_pairs` and uses `optimal = 2*pi/degree`. This is a genuine angular-resolution criterion, but it belongs to the classic SGD2 criterion path, not the default `dagua_native` pipeline.
3. `_port_angular_resolution_loss` in `dagua/layout/edge_optimization.py:1422`. This targets routing/port control points after node layout, not the learnable node positions used by `dagua_native`'s main optimizer.

Conclusion: there is no gate bug or zero-weight native angular term. The existing default term most likely helping angular resolution is `FanoutDistributionLoss`, but it does not fire on many relevant graph classes and does not match the benchmark metric. A new native loss op is needed if the goal is to move `angular_res_mean_deg` across the 93-graph ranking table.

# Loss formulation pseudocode

The graph-drawing literature defines angular resolution as the smallest angle formed by incident edges at a vertex; Schaefer's 2023 JGAA paper states that definition directly, and the TikZ/PGF force-layout manual explicitly lists rotational forces between edges leaving a node as a standard force-model ingredient for creating equal angular resolution. This supports a principled loss, not a fixture-specific polish.

Recommended objective: undirected, degree-adaptive, soft-margin minimum-angle loss over incident edge pairs. Use the ideal local target `2*pi/degree(v)`, capped to avoid impossible aggression on high-degree hubs.

```python
def angular_resolution_loss(pos, edge_index, min_avg_degree=3.0,
                            max_pairs_per_node=64,
                            target_cap_deg=45.0,
                            eps=1e-6):
    # Structural anti-gaming gate. No graph names, no signatures.
    n = pos.shape[0]
    e = edge_index.shape[1]
    if n == 0 or e == 0 or (2.0 * e / n) < min_avg_degree:
        return pos.sum() * 0.0

    # Undirected incidence, matching metrics.py.
    src = concat(edge_index[0], edge_index[1])
    nbr = concat(edge_index[1], edge_index[0])
    degree = bincount(src, minlength=n)
    active_nodes = where(degree >= 3)
    if active_nodes is empty:
        return pos.sum() * 0.0

    losses = []
    for v in active_nodes:
        neighbors = nbr[src == v]
        # For large hubs, sample deterministic adjacent-neighbor pairs or a
        # bounded all-pairs subset to keep cost proportional to useful signal.
        pairs = unordered_pairs(neighbors, max_pairs=max_pairs_per_node)
        u = normalize(pos[pairs.left] - pos[v], eps)
        w = normalize(pos[pairs.right] - pos[v], eps)
        angle = arccos(clamp(dot(u, w), -1 + eps, 1 - eps))

        # For degree d, perfect straight-line angular spacing is 2*pi/d.
        # Cap keeps high-degree hubs from forcing huge global distortion.
        theta_target = min(2*pi / degree[v], radians(target_cap_deg))
        margin = relu(theta_target - angle)
        losses.append(mean((margin / theta_target) ** 2))

    return mean(losses) if losses else pos.sum() * 0.0
```

Two implementation details matter. First, normalize by `theta_target` so high-degree hubs and degree-3 nodes have comparable loss magnitudes. Second, skip degree-2 nodes in the loss even though the metric samples them, because optimizing degree-2 angle can fight layered DAG straightness and chains already get good angular resolution from vertical spacing.

The op can be vectorized later with sorted incidence lists as `fanout_distribution_loss` does, but the first implementation should keep a small per-node cap. The expected fire set is moderate, and angular loss is not worth an O(sum d^2) blow-up on `ba_5000`, `rgg_2000`, or future scale graphs.

# Pipeline insertion point

Concrete wire points:

- Add `AngularResolutionLossConfig` near `FanoutDistributionLossConfig` in `dagua/layout/ops/loss_engine.py:704`.
- Add registered `AngularResolutionLoss` near `FanoutDistributionLoss` at `dagua/layout/ops/loss_engine.py:1284`. It should read `pos` and `edge_batch_context` only if the implementation supports bounded edge sampling; otherwise read only `pos`.
- Import it in `dagua/layout/resolve.py` with the other loss-engine ops, then append it in `build_loss_ops` at `dagua/layout/resolve.py:462-469`, preferably after `EdgeLengthVarianceLoss` and before `SpacingConsistencyLoss` / `FanoutDistributionLoss`.
- Add `w_angular_resolution: float = 0.5` to `LayoutConfig` next to `w_fanout` in `dagua/config.py:156-158`.
- Add the same field to `InitAnnealingScheduleConfig` in `dagua/layout/ops/anneal.py:1010` and schedule it in `InitAnnealingSchedule.apply` near `w_fanout` at `dagua/layout/ops/anneal.py:1144`.

Pipeline scope: `dagua_native` only, including the layered-DAG and hybrid sub-pipelines because both delegate into `dagua_native_legacy.build_dagua_pipeline`. Do not wire this into all classic pipelines. Do not alter `native_force_directed`; its stress/Pivot-MDS path is a specialist opt-in and should not inherit a native-specific loss without separate evidence.

Schedule: use a late ramp rather than full strength from step 0:

```python
progress = step / max(total_steps - 1, 1)
w_angular = config.w_angular_resolution * clamp((progress - 0.25) / 0.50, 0.0, 1.0)
```

That lets DAG ordering, overlap, and length variance establish the global layout before local angle spreading starts. A constant 0.5 is simpler, but riskier on protected DAG wins because angular gradients can rotate edges away from the vertical axis that currently gives dagua its excellent straightness rank.

# Class predicate (the gate)

Use a structural gate:

```python
avg_degree = 2.0 * num_edges / max(num_nodes, 1)
active = avg_degree >= 3.0 and num_edges >= 2
```

Optionally require at least one degree >= 3 inside the op. Do not gate on graph names, `(N, E)` signatures, tags, metric values, or candidate scoring against the composite. The average-degree gate is intentionally coarse and externally explainable: angular separation has meaningful mass only when incident-edge competition is common. It also protects the named low-density wins:

- `deep_chain_20`: avg degree 1.91, no degree-3 nodes, should not fire.
- `random_dag_200`: avg degree about 1.57 in `get_test_graphs()` despite some high-degree nodes, should not fire.
- `org_chart_deep`: avg degree 1.97, should not fire.
- `hub_fanout_label_skew`: avg degree 2.60, should not fire.
- `ba_500`: avg degree 5.98, will fire and must be explicitly protected by validation, not by name.

# Empirical predictions per graph class

From `dagua.eval.graphs.get_test_graphs()`, 56 of 101 available evaluation graphs have `avg_degree >= 3.0`. The exact 93-graph honest table is a subset/benchmark scoring view, but the structural fire set is clear. Meaningful classes:

- Dense/random/geometric/community: `dense_pair_50`, `dense_skip_200`, `random_dense_300`, `er_*`, `rgg_*`, `sbm_*`, `real_football_115`, `real_lesmis_77`. These have many degree-3+ vertices, so the term should fire broadly. Expected angular lift: +2 to +6 degrees mean on small/medium graphs, rank lift 1-2 positions where dagua currently loses to spring/neato/sfdp-like layouts.
- Scale-free/hub-heavy: `ba_*`, `powerlaw_*`, `scale_free_ba_120`, `dependency_*`, `hub_spoke_*`, `wide_1_100_1`, `wide_3_50_3`, `complete_bipartite_8x12`. These have the largest raw angular problem at hubs. Expected lift: +3 to +10 degrees at hub nodes, but mean metric lift depends on how many non-hub degree-2 nodes are sampled. Highest risk class.
- Mesh/regular/planar: `grid_5x5`, `grid_rect_6x8`, `grid_20x20`, `grid_50x50`, `regular_3_30`, `regular_4_40`, `triangular_lattice_36`, `sierpinski_42`, `petersen_10`, `planar_60`. Expected lift: +1 to +4 degrees, mostly by avoiding nearly collinear local spokes. Risk is lower because ideal angles are modest and degree distribution is bounded.
- Neural/skip/wide small graphs: `densenet_block`, `multiscale_skip_cascade`, `center_port_backedge_hub`, `bipartite_4_3_4`, `long_skip_only_24`. Expected lift: +1 to +5 degrees, but composite volatility is high because a single rotated hub can affect edge CV, crossings, and straightness.
- Low-density chains/trees/sparse DAGs: should be no-op under the recommended predicate, even if isolated nodes have degree >= 3. This is deliberate composite protection.

The suite-level target, `6.41 -> ~5.4` mean angular rank, is plausible but not guaranteed. Because only ~55-60% of graphs fire, the firing subset needs about a 1.7-2.0 rank improvement to move the all-graph mean by 1.0. That is achievable if dense/geometric/community graphs improve, but unlikely if gains are limited to a few hub graphs.

Assumption: the 93-graph honest benchmark is drawn from the local benchmark/eval graph collection plus available competitor coverage. I used the local structural graph inventory for fire-set estimates rather than graph names from old position files, because the implementation gate must be reproducible from graph topology alone.

# Trade-off analysis

Expected cost is mainly on `edge_straightness_mean_deg` and `edge_length_cv`.

Straightness: dagua is elite here, mean rank 2.07. Angular spreading rotates incident edges around high-degree nodes; that can make layered DAG edges less vertical. With late ramp and weight 0.5, I would expect a small all-suite straightness rank regression, about +0.1 to +0.3 mean rank. On hub-heavy layered graphs, individual straightness score could drop 1-4 degrees if the loss spreads spokes horizontally.

Edge length CV: spreading incident angles around hubs often lengthens some spokes and shortens others unless attraction/length-variance dominates. Dagua's edge-CV rank is only mid-pack, but the composite weight for CV is 20, so a visible CV regression is more dangerous than the angular gain. Expected all-suite composite impact: -0.05 to -0.20 mean rank if tuned well; -0.4 or worse if hub graphs stretch.

Crossing rate: ambiguous. Local angular separation can reduce near-overlap bundles and accidental same-port crossings, but it can also fan edges into adjacent corridors. Expected net near zero; high-degree bipartite and hub-spoke graphs need visual review.

Composite: angular has only 5 composite weight, while edge CV and DAG/straightness are much larger. The expected net composite delta is close to flat, maybe -0.05 to +0.10 mean rank, if the term is late-ramped and gated. A naive always-on or too-large weight is likely to improve angular rank while losing the 1.21 composite crown.

# Validation gates

Implementation must pass these gates before becoming default:

- Target graphs: on graphs with `avg_degree >= 3.0`, angular-resolution mean rank improves by at least 1.0. All-suite angular rank should move from about 6.41 to 5.4 or better.
- Composite mean rank remains below 1.5.
- Protected wins remain stable: `deep_chain_20`, `random_dag_200`, `ba_500`, `org_chart_deep`, and `hub_fanout_label_skew` individual composite deltas within +/-0.5.
- Jitter: sigma=0.5, 8 trials on three representative high-degree graphs, suggested `ba_500`, `rgg_500`, and `complete_bipartite_8x12`; mean angular delta > 0 and minimum composite delta > -2.0.
- Visual review of top two angular gains and top two composite losses.

# Risks / failure modes

The largest risk is optimizing the metric but worsening human readability. Angular resolution is local; it can reward hub spokes that spread evenly while the global drawing gets wider, noisier, or less layered. This is exactly the kind of proxy-overfit the April 26 retro warns about, so no polish picker, no per-graph candidates, and no metric-scored best-of selection should be added.

The second risk is gradient instability. `arccos` has steep derivatives near +/-1; clamp and vector normalization epsilons are mandatory. A cosine-margin alternative avoids `arccos` but becomes less interpretable against the degree-adaptive angle target. Start with the interpretable angle loss and clamp tightly.

The third risk is high-degree cost. Exact pair enumeration around `ba_5000` hubs is expensive and can dominate runtime. Cap pairs per node, sample deterministically from sorted incidence, and preserve differentiability through selected positions. If bounded sampling makes the signal too noisy, "no fix found" is the correct outcome.

The fourth risk is that the present rank gap may reflect pipeline choice more than a missing term. Force-directed engines naturally optimize round layouts, while dagua is intentionally DAG/layer biased. If lifting angular rank requires giving up the vertical/straight layered signature that drives composite rank 1.21, do not ship the default. Keep the loss opt-in or class-specialized for dense non-DAG/hub graph families.

Sources: Schaefer, "The Complexity of Angular Resolution", JGAA 2023, https://jgaa.info/index.php/jgaa/article/view/paper634; TikZ/PGF force-layout manual, https://tikz.dev/gd-force.
