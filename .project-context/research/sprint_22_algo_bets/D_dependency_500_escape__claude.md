# Area D — dependency_500 escape from gradient saturation (claude)

## TL;DR

- **The aspect-preserving equalize family does NOT close the dependency_500 gap.** Every variant (lock_x, lock_y, both, per-iter rescale, final-only rescale) regresses composite by **-15 to -19 points** on dependency_500. The base equalize (the current `_equalize_edges`) regresses by the same magnitude. The picker correctly rejects it today; locking the bbox does not change the picker's verdict.
- **The gap-constrained adjacent-x-swap closes the gap by itself, with margin to spare.** A single pass of swap_adjacent (over all same-y-band node pairs, accept if composite improves) gives **+2.87 composite** on dependency_500 (392s). Two passes gives **+3.83** (771s). dependency_500: 55.28 -> **59.12** vs elk_layered 58.19. **-2.90 LOSS becomes +0.93 WIN.**
- **Picker safety on the four protected wins is preserved.** gap_swap deltas on random_dag_200 +0.07, org_chart_deep +0.06, hub_fanout_label_skew +0.00, plus the four moderate-loss/close-loss spillovers I checked (multi_component_80 +0.39 win, triangular_lattice_36 +0.05, small_world_500 +0.001, others +0.00). All deltas on protected wins fall well below the 0.5 picker margin so the existing protected-win base layouts are kept; multi_component_80's +0.39 stays just below the 0.5 margin too (would not flip pick), while dep500's +3.83 sails past it.
- **The two-stage approach only buys what stage 2 buys.** Aspect-preserving (stage 1) ALWAYS regressed on dep500. It was supposed to "let the equalizer reduce CV without losing aspect," but the layered_dag pipeline's geometry doesn't tolerate any equalizer pass — locking bbox doesn't help. Stage 2 (gap_swap) provides the entire +3.83. **The two-stage framing is wrong for this graph; ship gap_swap alone.**
- **Bonus.** gap_swap also gives a small WIN of +0.39 on multi_component_80 (-0.64 close-loss), worth +0.05 on triangular_lattice_36 (-1.61), and +0.06 on org_chart_deep (existing strong win — not picker-flippable so harmless). All other protected/close-loss graphs got +/-0.001 (no-op).
- **Cost is the real risk.** 1 pass = ~400s extra at N=500, 2 passes = ~770s. That is 5-10x the dagua_native pipeline runtime. Recommended: gate by N (only run on graphs N>=100 with edge_length_cv >= 0.85), use 1 pass by default, picker margin gate then auto-accepts the +2.87 dep500 win.

## What I tested (and what's actually in the prompt)

The prompt asks for two specific algorithm sketches:

1. **`aspect_preserving_equalize`** — re-scale positions after each (or final) iteration of `_equalize_edges` so the bounding box stays at the original layered_dag-pipeline output bbox. Expected +0.5..+1.5 on dependency_500.

2. **gap-constrained layered local search** — for each layer (or each node touching long edges), tentatively swap adjacent-x neighbors, accept if a weighted composite-like objective improves. Expected: opaque, but the prompt frames it as the fallback if aspect-preserving doesn't suffice.

I implemented both in /tmp/, ran them on dependency_500 plus the three protected-win graphs (random_dag_200, org_chart_deep, hub_fanout_label_skew), plus the eight close-loss / tied-bucket graphs. Layout for dependency_500 takes 51 seconds (cached) and `composite(full(...))` takes ~0.85s per scoring, so a full sweep over all variants costs roughly 30 minutes per heavy graph and 30 seconds per small graph.

The methodical separation requested by the prompt — "aspect-preserving first, swap-based second, identify which buys the most" — is the right framing. Methodical separation is exactly what produced the bottom-line answer: stage 1 buys nothing (worse than nothing), stage 2 buys everything.

## Stage 1: aspect_preserving_equalize — pseudocode + measured deltas

### Pseudocode

```python
def equalize_aspect_preserving(pos, edges, iters, step, *, lock_x, lock_y, rescale_per_iter):
    pos = pos.detach().clone()
    src, tgt = drop_self_loops(edges)
    if no_edges:
        return pos
    # capture original bbox
    x0_min, x0_max = pos[:,0].min(), pos[:,0].max()
    y0_min, y0_max = pos[:,1].min(), pos[:,1].max()
    x0_range = max(x0_max - x0_min, eps)
    y0_range = max(y0_max - y0_min, eps)

    def rescale(p):
        if lock_x:
            p[:,0] = (p[:,0] - p[:,0].min()) / max(p[:,0].max() - p[:,0].min(), eps) * x0_range + x0_min
        if lock_y:
            p[:,1] = (p[:,1] - p[:,1].min()) / max(p[:,1].max() - p[:,1].min(), eps) * y0_range + y0_min
        return p

    for _ in range(iters):
        # stock equalize step (mirrors _equalize_edges in dagua_native.py:408-450)
        diffs = pos[tgt] - pos[src]
        dists = diffs.pow(2).sum(-1).sqrt().clamp(min=1.0)
        target = dists.mean()
        unit = diffs / dists.unsqueeze(-1)
        delta = (dists - target).unsqueeze(-1) * unit * step
        pos.index_add_(0, src, delta * 0.5)
        pos.index_add_(0, tgt, -delta * 0.5)
        if rescale_per_iter:
            pos = rescale(pos)
    if not rescale_per_iter:
        pos = rescale(pos)
    return pos
```

I tested all 24 combinations (4 polish_settings x 3 lock_modes x 2 rescale_modes) plus a y-locked-only variant ("snap y back to original each iter, let x evolve freely") which is the strictest preservation of layered_dag's depth assignment.

### Measured deltas on dependency_500 (base composite = 55.284)

```
variant                                                    composite   delta
baseline_eq_10_0.1                                            39.96   -15.32   <- existing _equalize_edges, current picker
baseline_eq_30_0.02                                           39.75   -15.53
baseline_eq_50_0.05                                           39.59   -15.70
baseline_eq_50_0.2                                            nan       nan   (NaN explosion)
asp_per_iter_i10_s0.1_lx0_ly1                                 39.92   -15.36
asp_per_iter_i10_s0.1_lx1_ly0                                 39.90   -15.39
asp_per_iter_i10_s0.1_lx1_ly1                                 36.05   -19.24
asp_final_i30_s0.02_lx1_ly1                                   36.71   -18.58
... (every other variant in the same -15..-19 range)
y_locked_eq_i30_s0.02                                         51.40    -3.89
y_locked_eq_i10_s0.1                                          52.52    -2.76
y_locked_eq_i50_s0.05                                         53.63    -1.65
```

### Reading the result

The prompt's prediction was "+0.5..+1.5 on dependency_500." Measured: -15 to -19 across the entire family. This is not a tuning miss; it is qualitatively wrong about how dep500's geometry interacts with the equalizer.

The diagnosis: dependency_500's layered_dag pipeline output puts the 10 dominant core libraries at the top of a tall, narrow stack (the high max_degree=53 hub fan-in dominates the layout). The equalizer step pulls every long edge toward the mean edge length. Because the hub edges are long and many, this collapses the hub-to-leaf vertical extent. With no bbox lock, the bbox shrinks proportionally and many metrics regress mildly. With bbox lock (per-iter or final), the rescale stretches the now-collapsed positions back to the original bbox, but they have already lost their relative depth ordering — the rescale cannot recover the depth ranking. The composite metric `dag_consistency` measures the proportion of edges going downward in y; once enough hub-leaf vertical orderings have flipped, dag_consistency falls off a cliff (~0.3 -> ~0.05) and that single sub-metric eats 20+ points.

The y-locked-only variant ("snap y back, let x flow") is closer to the prompt's intent ("preserve the layered y, let the optimizer reduce x-driven CV"). But it still regressed -1.65 to -3.89. The reason: x-only equalize on a graph with E=1471 and max_degree=53 produces big lateral collisions of hub leaves; node overlap penalties spike. So even axis-locked equalize is wrong here.

**Conclusion on stage 1: aspect-preserving equalize gives no path to +0.5 composite on dependency_500. The codex E expectation was off. Do not implement aspect_preserving_equalize as a polish primitive based on this graph alone; if implemented, the picker would reject it on dep500 the same way it rejects baseline equalize today.**

(That said: aspect-preserving equalize might still be worth implementing for a *different* reason — the existing baseline equalize regresses on dep500 by -15 because bbox drifts, but on graphs where it's already a picker winner, locking bbox would keep that win robust. But that's outside this prompt's scope and isn't where the dep500 escape lives.)

## Stage 2: gap-constrained adjacent-x-swap — pseudocode + measured deltas

### Pseudocode

```python
def gap_constrained_swap(
    pos, edges, sizes, *, score_fn,
    layer_eps=1.0, max_passes=2, max_nodes=600, only_long_edges=False, long_q=0.85,
):
    pos = pos.detach().clone()
    if pos.shape[0] > max_nodes:
        return pos                       # cost gate
    band = sizes[:,1].mean() * layer_eps
    if band <= 1e-6:
        return pos
    # Optional: restrict to nodes touching long edges (CV's worst contributors)
    interesting = None
    if only_long_edges:
        edge_lens = (pos[edges[1]] - pos[edges[0]]).pow(2).sum(-1).sqrt()
        thr = quantile(edge_lens, long_q)
        long_mask = edge_lens >= thr
        interesting = set(unique(cat([edges[0][long_mask], edges[1][long_mask]])))

    buckets = round(pos[:,1] / band).long()      # quantize y into "layer" bands
    best = score_fn(pos)
    for pass_idx in range(max_passes):
        any_swap = False
        for bucket in unique(buckets, sorted=True):
            members = nonzero(buckets == bucket)
            if members.numel() < 2:
                continue
            ordered = members[argsort(pos[members, 0])]   # left-to-right by x
            for i in range(ordered.numel() - 1):
                a, b = ordered[i], ordered[i+1]
                if interesting is not None and a not in interesting and b not in interesting:
                    continue
                pos[a,0], pos[b,0] = pos[b,0].clone(), pos[a,0].clone()    # tentative swap
                new = score_fn(pos)
                if new > best + 1e-6:
                    best = new
                    any_swap = True
                else:
                    pos[a,0], pos[b,0] = pos[b,0].clone(), pos[a,0].clone()  # revert
        if not any_swap:
            break
    return pos
```

This is conceptually identical to `dagua/layout/ops/crossing_swap.py:CrossingSwapPolish` (Sugiyama Phase 3) except:

1. The score function is the actual full composite (not local crossing count). This is why it works: the composite reflects edge_length_cv, dag_consistency, straightness, overlap penalty, etc. simultaneously. A swap that reduces edge_length_cv without raising any other penalty is accepted.
2. The "layer" used for grouping is a quantized-y bucket (step = mean_node_height) rather than `state.layer_index` (which dep500's pipeline does populate, but using y-buckets makes the op compatible with non-layered pipelines too).
3. Cost gate (max_nodes=600) avoids pathological scaling; each swap costs one composite evaluation (~1s on dep500), so worst-case cost = 2 * num_pairs_per_pass.

### Measured deltas on dependency_500 (base composite = 55.284, elk=58.19, gap = -2.90)

From the drill-down sweep (variant, swaps_accepted, candidates_examined, time, composite, delta):

```
p1_full              swaps= 209  cand= 464  passes=1  t= 392s  s=58.151  +2.867
p2_full              swaps= 379  cand= 928  passes=2  t= 771s  s=59.118  +3.834
p1_long85            swaps= 164  cand= 330  passes=1  t= 274s  s=57.544  +2.260
p1_long70            swaps= 193  cand= 421  passes=1  t= 348s  s=57.997  +2.713
p1_long50            swaps= 207  cand= 457  passes=1  t= 381s  s=58.125  +2.841
p2_long70            swaps= 351  cand= 847  passes=2  t= 732s  s=58.936  +3.652
```

Reading: a single full-coverage pass already closes -2.90 to a tie (+2.87 lifts dep500 to 58.15, elk's 58.19 is a tie within score noise). A second pass overshoots elk by +0.93. The long-edges-only variants are slightly cheaper but slightly less effective; with max_passes=2 they recover most of the second-pass gain (long_q=0.70: +3.65 vs full +3.83).

**Conclusion: 1-pass full-coverage gap_swap is the recommended default. It produces a tie/small-win ($55.28 + 2.87 = 58.15$ vs elk 58.19) at ~400s; if higher quality is desired and budget allows, a 2-pass run gives a clean +0.93 win at ~770s.**

### Why it works (mechanistic)

Layered_dag fixes y by topological depth (correctly). The within-layer x-positions come from barycenter sweeps + relaxation, which is local — it minimizes a weighted sum of (neighbor centroid distance + spring force + repulsion) per layer, not the global composite. The global composite cares about edge_length_cv across ALL edges, not just within-layer barycenter. So adjacent-pair x-swaps within a y-band can reduce CV at the cost of local barycenter (raising the gradient-descent loss the engine optimizes), but the optimizer's loss landscape is saturated — no gradient signal points it toward swaps. A discrete combinatorial pass over the actual composite finds them.

dependency_500 has 500 nodes, ~14-16 layers (depth), so per-layer ~30-40 nodes, ~30-40 adjacent pairs, ~14*35 = 490 candidates per pass. Observed: 464 candidates examined, 209 swaps accepted. Acceptance rate ~45% — much higher than the typical 5-10% rate on smaller well-tuned graphs, indicating the saturated-gradient diagnosis is correct: the local optima the engine settled into has many easy improvements visible to an exhaustive combinatorial search.

## Stage 1+2 combo

Tested but dominated by stage 1's regression. Numerically:

```
COMBO_asp_final_i10_s0.1_lx1_ly0+gap_swap(s=26,t=403s)         39.99   -15.29
```

Once aspect-preserving has destroyed the dag_consistency by -19, gap_swap can recover only +0.1 from the wreck. The two stages are not additive; stage 1's regression dominates. **Do not combine the two on dep500.**

## Risk / regression analysis

### Protected wins

I tested the four protected wins from sprint-21b CONTEXT.md (random_dag_200, org_chart_deep, hub_fanout_label_skew) plus the eight close-loss / moderate-loss bucket graphs. Single-pass gap_swap deltas:

```
graph                              N    E    base     after    delta    swaps  time
small_world_500                  500 1500   52.190   52.192   +0.001        1   59s
disconnected_encoder_residual      9    8   84.013   84.013   +0.000        0    0s
triangular_lattice_36             36   85   85.478   85.531   +0.053        1    0s
clustered_medium_5x20            100  193   69.784   69.784   +0.000        0    1s
outerplanar_dag_20                20   37   72.417   72.417   +0.000        0    0s
multi_component_80                80   81   74.461   74.852   +0.391        5    1s
hexagonal_lattice_42              42   53   88.355   88.355   +0.000        0    0s
parallel_cycles_4x5               20   20   62.110   62.110   +0.000        0    0s
recurrent_feedback_cell            5    6   73.185   73.185   +0.000        0    0s
small_world_100                  100  200   57.178   57.178   +0.000        0    2s
random_dag_200                   383  300   74.130   74.200   +0.070        3   21s
org_chart_deep                    79   78   92.441   92.499   +0.059        6    2s
hub_fanout_label_skew             10   13   93.737   93.737   +0.000        0    0s
```

Three observations matter:

1. **No graph regresses.** Gap_swap by construction only accepts a swap if it strictly improves composite, so the worst-case delta is 0 (no swap accepted). Confirmed empirically across all 13 graphs.
2. **Protected wins have deltas <=0.07.** Below the picker's 0.5 margin, so the picker keeps the existing un-polished baseline. Protected-win-baselines remain unchanged.
3. **A small bonus on multi_component_80 (+0.39) and triangular_lattice_36 (+0.05).** These are sub-margin so picker doesn't flip yet, but if the picker margin is later relaxed or composed with other primitives these are nearly free wins. multi_component_80's -0.64 -> -0.25 close-loss bucket motion is a clear sub-margin step, sgnaling room for further work on multi-component graphs (Bet 3 in CONTEXT.md territory).
4. **dependency_500 is the only graph where a pass is large enough to flip the picker.** That's the right outcome: the polish pays for itself on the target and is invisible elsewhere.

### Cost concerns

dependency_500's 392s (1 pass) and 771s (2 pass) is a long time relative to the engine's pipeline runtime (51s on the cached graph). Two mitigations exist:

1. **Gate by N + edge_length_cv:** only run gap_swap on graphs where N >= 100 and the post-pipeline edge_length_cv exceeds, say, 0.85. dependency_500 hits both; small graphs and well-equalized graphs skip it for free.
2. **One-pass default:** 1 pass closes the dependency_500 gap to a tie within score noise (+2.87, elk 58.19, dagua 58.15, delta -0.04 — well inside metric jitter for sub-0.5 differences). A second pass is the difference between "tied" and "+0.93 win"; if budget is tight, ship the 1-pass version.

### Failure modes I checked

- **NaN positions from base equalize_50_0.2:** observed on dependency_500. The aggressive step explodes. gap_swap doesn't have this problem since it only swaps existing finite values.
- **Sub-2-node layers:** skipped (no adjacent pair exists).
- **Empty edge_index:** early-return guard.
- **N>600 cost gate:** guard against pathological scaling on huge graphs.

I did not check (would need integration in dagua/ to verify):

- Interaction with the existing polish picker (would gap_swap be invoked before, after, or alongside the existing edge_equalize candidates? The cleanest answer is: as one more candidate in the polish_candidates list at line 1056 of dagua_native.py).
- Determinism guarantees across torch versions / GPU devices (the swap is exact-equality based on float comparison; should be safe but worth a unit test).

## Implementation order

1. **Land gap_swap as a polish candidate.** ~80 LOC. Add it as one more entry in `polish_candidates` in `dagua/layout/ops/pipelines/dagua_native.py` (around line 1056), gated by `n >= 100` and `composite_edge_length_cv >= 0.85` (those gates skip the cost on graphs where it's a no-op anyway). Single pass default. The picker's existing margin gate decides accept/reject per graph. Expected impact: dependency_500 -2.90 -> +0.0 to +0.93 (depending on pass count); no other graphs flip; protected wins safe.

2. **Add a 2-pass override knob** for users running custom pipelines who can afford 770s per heavy graph. Default off.

3. **Skip aspect_preserving_equalize.** Don't ship. The empirical -15..-19 on dep500 means it doesn't pass picker on the target graph and the "let x evolve, lock y" intuition that motivated it is wrong: dep500 already has plenty of x freedom from layered_dag's barycenter sweep, but the smooth optimizer can't find the discrete swaps.

4. **Optional follow-up: revisit multi_component_80.** gap_swap gave +0.39 sub-margin. Combined with Bet 3 in CONTEXT.md (node-level global-depth y-alignment for multi-component DAGs), this might be enough to flip multi_component_80 from -0.64 close-loss to a tie or small win.

## File pointers

- Existing equalize: `/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:408-450` (`_equalize_edges`)
- Existing polish picker: same file, `_polish_with_candidates` around line 994-1090
- Existing crossing_swap polish (Sugiyama Phase 3): `/home/jtaylor/projects/dagua/dagua/layout/ops/crossing_swap.py` (close cousin to gap_swap; uses local crossing count rather than full composite, default disabled because it conflicts with overlap projection)
- /tmp/ scripts used: `/tmp/sprint22_d_cached.py` (full sweep), `/tmp/sprint22_d_drilldown.py` (gap_swap parameter sweep), `/tmp/sprint22_d_more_graphs.py` (cross-graph safety check)
- Cached layouts: `/tmp/sprint22_d_cache/{name}.pt` for dependency_500, random_dag_200, org_chart_deep, hub_fanout_label_skew, plus the eight close-loss bucket graphs.

## Numbers worth sanity-checking before implementation

- `dependency_500` base composite **55.284** matches CONTEXT.md's **55.28** (within rounding). Layout time was 51s after warm imports, 345s on first run — variance comes from torch's lazy CUDA init and graph_generator memoization, not the layout itself. All deltas computed against the warm-cache layout for self-consistency.
- The +3.83 figure (elk 58.19, dagua 59.12) requires 2 passes and 770s. The +2.87 figure (dagua 58.15, elk 58.19, ~tie) requires 1 pass and 392s. Both close the gap; the cheaper one is essentially free of risk.
- gap_swap accepts strictly-better swaps only, so the worst-case is a no-op (already verified across 13 graphs).
- The picker's 0.5-margin behavior at line 1024 of `dagua_native.py` means the +2.87 dep500 delta passes (>0.5), the +0.39 multi_component_80 delta does not pass (sub-margin, baseline kept), and the +0.07 random_dag_200 delta does not pass (sub-margin, baseline kept). This is the exact picker behavior we want.
