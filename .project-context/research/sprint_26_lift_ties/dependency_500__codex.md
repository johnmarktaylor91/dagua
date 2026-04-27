# dependency_500 sprint-26 research

## TL;DR

**Ship, narrowly.** The best candidate is not a new LP solver; it is a scored
global x-coordinate compression candidate on the existing HEAD output. On
`dependency_500`, fixed `node_sizes=torch.tensor([[40.0, 20.0]] * N)`, and the
normal `dagua.metrics.full()` + `dagua.metrics.composite()` scorer:

| Layout | Composite | Delta vs HEAD | Delta vs ELK |
|---|---:|---:|---:|
| un-polished gradient pipeline, `edge_equalize_polish=False` | 55.284 | -2.600 | -2.905 |
| HEAD default | 57.884 | 0.000 | -0.304 |
| `global_x_scale_0.40` candidate | **58.870** | **+0.985** | **+0.681** |
| `elk_layered` cached competitor | 58.189 | +0.304 vs HEAD | 0.000 |

The strict sprint bar is `current + 0.5`; this candidate clears it by about
`+0.485` points. Jitter validation is stable: eight sigma-0.5 Gaussian trials
on the candidate scored mean `58.869593`, min `58.869343`, max `58.869778`,
population std `0.000131`. The gain is not a sampled metric artifact.

The corrected bottleneck diagnosis: HEAD already beats ELK on edge-length CV,
but loses enough angular-resolution contribution to stay below ELK. The
candidate compresses x by `0.40` around the global center, preserving y-order,
edge crossings, and overlap status, while improving both edge CV and angular
resolution. It is surprisingly simple, but it targets the measured residual
directly.

## Measurement Setup

I ran fresh native layouts from the sprint-25 HEAD workspace:

- Target graph: `dagua.eval.graphs.get_test_graphs(...), name=="dependency_500"`
- HEAD layout: `dagua.layout(g, LayoutConfig(seed=42, device="cpu"))`
- Un-polished comparison: `LayoutConfig(seed=42, device="cpu", edge_equalize_polish=False)`
- Scoring: `dagua.metrics.full(pos, g.edge_index, node_sizes=fixed_40x20)` followed by
  `dagua.metrics.composite()`
- Competitor: cached
  `eval_output/variant_bench_full/positions/dependency_500__elk_layered.pt`
- Scratch artifacts: `/tmp/sprint26_dependency_500_codex/`

The default node-size tensor matters. These numbers use the prompt-required
`[[40.0, 20.0]] * N`, not graph label-derived sizes.

No package tests, ruff, or mypy were run because this was a read-only research
assignment and no `dagua/` files were modified. The verification commands were
metric/research commands only: target `full()` scoring, candidate jitter
scoring, all-registry gate enumeration, and six protected sample score checks.
All generated experiment scripts and tensors live under `/tmp/sprint26_dependency_500_codex/`;
the only project file written is this report.

## Per-Metric Breakdown

Raw metric values:

| Metric | Un-polished | HEAD | Candidate | ELK |
|---|---:|---:|---:|---:|
| composite | 55.284 | 57.884 | **58.870** | 58.189 |
| dag_consistency | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| edge_length_cv | 0.905426 | 0.748121 | **0.718740** | 0.787419 |
| depth_spearman_rho | 0.993923 | 0.993923 | **0.993923** | 0.968184 |
| overlap_count | 0 | 0 | 0 | 0 |
| edge_straightness_mean_deg | 71.810 | 81.345 | 72.985 | 59.070 |
| crossing_rate | 0.100672 | 0.120709 | 0.120709 | 0.107776 |
| angular_res_mean_deg | 7.870 | 3.503 | **6.684** | 15.314 |

Weighted composite contributions:

| Term | HEAD | Candidate | Delta | ELK |
|---|---:|---:|---:|---:|
| DAG consistency | 25.000 | 25.000 | 0.000 | 25.000 |
| edge CV contribution | 5.038 | **5.625** | **+0.588** | 4.252 |
| depth rho contribution | 14.909 | 14.909 | +0.000 | 14.523 |
| no-overlap contribution | 10.000 | 10.000 | 0.000 | 10.000 |
| straightness contribution | 0.000 | 0.000 | 0.000 | 0.000 |
| crossing contribution | 0.000 | 0.000 | 0.000 | 0.000 |
| angular contribution | 0.438 | **0.836** | **+0.398** | 1.914 |
| neutral cluster contribution | 2.500 | 2.500 | 0.000 | 2.500 |

The losses against ELK are therefore narrow and specific. HEAD loses
`~1.476` composite points on angular resolution and `~0.386` on depth rho,
while winning `~0.786` on edge CV. Straightness and crossings are already at
the composite floor for both layouts. The candidate improves enough CV and
angular contribution to pass ELK even though it still does not match ELK's
absolute angular resolution.

## Variants Tried

1. **Un-polished vs HEAD.** This confirmed that sprint-23c/current polish is
   doing real work: `55.284 -> 57.884`, mostly edge CV `0.905 -> 0.748`.

2. **x-blend between un-polished and HEAD.** This failed. Blending x coordinates
   recovers some angular spread, but creates 18-24 fixed-size node overlaps and
   falls to `45.48..47.78`. The overlap cliff makes this unsuitable.

3. **Layer-local x scaling.** Best was `layer_scale_0.66` at `58.014`, a small
   `+0.129` lift. It is stable but below the sprint bar. Compressing each layer
   independently helps angular resolution but does not improve CV enough.

4. **Global x scaling.** This is the winner. The sweep found:
   `0.65 -> 58.241`, `0.55 -> 58.425`, `0.50 -> 58.544`,
   `0.45 -> 58.689`, `0.40 -> 58.870`, and `0.35 -> 49.098` because three
   overlaps appear. The useful band is narrow, so production should score
   candidates and reject overlap regressions rather than hard-replacing.

5. **Hub fan spread.** Moving high-degree children around hub x positions caused
   4-6 overlaps and scored around `47.46..48.00`. Reject.

6. **SLSQP/LP-style x refinement.** I started a constrained SLSQP prototype, but
   it did not return in useful research time after the target layouts were
   cached. Given the simple global-scale candidate clears the strict bar and is
   easier to gate, I would not ship an optimizer for this sprint.

## Candidate Sketch

```python
def dependency_500_x_scale_candidate(base_pos, edge_index, node_sizes, score_fn):
    """Return a scored x-compression polish candidate for dependency_500."""
    if not should_dependency_500_x_scale(base_pos, edge_index):
        return base_pos

    best_pos = base_pos
    best_score = score_fn(base_pos)

    # Include a small sweep because 0.35 creates overlaps while 0.40 wins.
    # The score callback must be the same full()+composite() picker path.
    for alpha in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        cand = base_pos.clone()
        center_x = cand[:, 0].mean()
        cand[:, 0] = center_x + alpha * (cand[:, 0] - center_x)
        cand = cand - cand.mean(dim=0, keepdim=True)

        if not torch.isfinite(cand).all():
            continue

        # Cheap fixed-size overlap precheck avoids wasting full() calls.
        # Production may use the existing count_overlaps_detailed helper.
        if count_overlaps(cand, node_sizes) > count_overlaps(base_pos, node_sizes):
            continue

        score = score_fn(cand)
        if score > best_score + 0.1:
            best_pos = cand
            best_score = score

    return best_pos


def should_dependency_500_x_scale(pos, edge_index):
    """Conservative topology gate; accepts only the sprint target locally."""
    n = pos.shape[0]
    e = edge_index.shape[1] if edge_index.numel() else 0
    if n != 500:
        return False
    if e < 1450 or e > 1485:
        return False

    depth = longest_path_layering(edge_index, n)
    unique_depths = torch.unique(depth)
    layer_count = int(unique_depths.numel())
    max_layer_width = max(int((depth == d).sum()) for d in unique_depths)
    if layer_count < 15 or layer_count > 25:
        return False
    if max_layer_width < 70 or max_layer_width > 100:
        return False

    src = edge_index[0]
    tgt = edge_index[1]
    out_degree = torch.bincount(src, minlength=n)
    in_degree = torch.bincount(tgt, minlength=n)

    if int((out_degree >= 25).sum()) < 7:
        return False
    if int(in_degree.max()) > 3:
        return False
    return True
```

## Gate Predicate

Recommended production gate:

- `N == 500`
- `1450 <= E <= 1485`
- longest-path layer count in `[15, 25]`
- max layer width in `[70, 100]`
- at least seven source/hub nodes with out-degree `>= 25`
- max in-degree `<= 3`
- candidate must be accepted only through the existing full-composite picker
  and must not increase `overlap_count`

I tested this predicate over the local benchmark registry exposed by
`get_test_graphs(max_nodes=10000)`. This checkout exposes 101 graphs, not 93;
the gate accepted only `dependency_500` and rejected the other 100. That is
stricter than the sprint requirement to reject all other 92 benchmark graphs.

The exact `N/E` part is intentionally conservative. `ba_500` is the closest
confounder (`N=500`, `E=1494`, similar hub count), and it is rejected by the
edge-count upper bound. `small_world_500` is rejected by `E=1500` and one-layer
depth. `er_500`, `rgg_500`, and `powerlaw_500` miss the edge-count and/or degree
signature.

## Protected Sample Checks

Because the gate rejects these samples, the candidate is a no-op and the score
is identical by construction. I still scored the default layouts once:

| Graph | N/E | Gate | HEAD | Candidate | Delta |
|---|---:|---:|---:|---:|---:|
| dependency_graph_100 | 100 / 285 | reject | 59.706 | 59.706 | 0.000 |
| random_dag_200 | 383 / 300 | reject | 73.991 | 73.991 | 0.000 |
| compound_10x20 | 200 / 308 | reject | 80.239 | 80.239 | 0.000 |
| small_world_500 | 500 / 1500 | reject | 57.400 | 57.400 | 0.000 |
| hexagonal_lattice_42 | 42 / 53 | reject | 89.114 | 89.114 | 0.000 |
| triangular_lattice_36 | 36 / 85 | reject | 87.058 | 87.058 | 0.000 |

## LOC Estimate

Implementation should be small:

- Gate helper: 35-45 LOC
- x-scale candidate helper: 25-35 LOC
- wiring into `_best_of_polish`: 5-8 LOC
- tests: 60-90 LOC

Total production change estimate: **125-175 LOC**, mostly tests and docstrings.

## Assumptions, Concerns, Knowledge

Assumption: the candidate should be deployed as a scored polish candidate, not a
forced transform. The `0.35` scale result proves there is an overlap cliff close
to the winning band.

Controversial choice: the gate is target-shaped. That is deliberate for
sprint-26: the mandate is to lift a specific tie without disturbing protected
wins. Generalizing this to other dependency graphs should be a separate sprint
because `dependency_graph_100` is already rejected and would need a different
scale band.

Concern: this improves the composite while making the drawing more horizontally
compressed. It does improve measured straightness degrees (`81.345 -> 72.985`)
and angular resolution, and it stays overlap-free under fixed `40x20` nodes, but
visual review is still warranted before merging.

Knowledge to remember: after sprint-23c, `dependency_500` is no longer primarily
behind ELK on edge CV. HEAD's residual gap is angular resolution plus a small
depth-rho difference; a global aspect adjustment beats another local swap pass.
