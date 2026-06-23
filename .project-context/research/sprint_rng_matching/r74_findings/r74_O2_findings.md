# r74 O2 Findings — SFDP cluster (184 divergent, mode A)

Analyst: read-only survey. Sources cited inline. Date 2026-06-22.

## TL;DR partition of the 184

| Bucket | Combos | Cause | Class | Confidence |
|---|---:|---|---|---|
| **A. p_neg2 force-law bug** | **73** | dagua passes `repulsive_exponent=-2` straight to the kernel; graphviz *clamps* `repulsiveforce<0` to p=-1. dagua computes `pow(dist,3)`, graphviz `pow(dist,2)`. Deterministic, non-FP. | SYSTEMATIC (one-line) | **HIGH** |
| **B. Disconnected-graph: no component packing** | **48** | graphviz sfdp lays out each connected component independently + `packSubgraphs`; dagua runs all components in one shared force field. disp median **0.121** (8x too compact). | SYSTEMATIC (port gap) | **HIGH** |
| **C. Connected, disp≈1.0 basin floor** | **~63** | force/coarsening source-faithful; native vs ref land in different attractors with equal seed variance (disp median 1.000). | LIKELY TRUE FLOOR — needs proof | MED |
| Overlap note | — | A and B are disjoint (p_neg2 excluded from B count). Some of C may still hide a scale issue (see C-2). | | |

Salvageable now: **>=121 of 184** (A+B). Likely true floor: **<=63** (C) — and even C is *under-proven*; r71's "floor" verdict was asserted, never demonstrated.

---

## BUCKET A — p_neg2 repulsive-exponent bug (73 combos, HIGHEST ROI)

### Evidence
Per-variant rung distribution (`per_combo.json`, engine grouping):

```
classic_sfdp_default            rung4=21   (of ~101)
classic_sfdp_graphviz_fidelity  rung4=21
classic_sfdp_theta04            rung4=21
classic_sfdp_theta08            rung4=23
classic_sfdp_steps200           rung4=25
classic_sfdp_p_neg2             rung4=73   <-- 3x every other variant
```
p_neg2 alone is 73/184 = 40% of the entire divergent bucket, and 72% of its own 101 combos diverge vs ~21% for siblings. A whole VARIANT systematically off => one-line cause, not scattered FP.

### Root cause (graphviz source)
`_references/graphviz/lib/sfdpgen/sfdpinit.c:212`:
```c
ctrl->p = -1.0*late_double(g, agfindgraphattr(g,"repulsiveforce"), -AUTOP, 0.0);
```
`late_double(obj, attr, default, minimum)` clamps the *parsed value* to `>= minimum` (`_references/.../lib/common/utils.c:66`: `if (rv < minimum) return minimum;`). Here `minimum = 0.0`.

Trace for `repulsiveforce=-2.0`:
1. parsed rv = -2.0; `rv < 0.0` -> clamped to **0.0**.
2. `ctrl->p = -1.0 * 0.0 = 0.0`.
3. `spring_electrical.c:287` (and :437): `if (p >= 0) ctrl->p = p = -1;` -> since 0.0>=0, **p = -1.0**.

=> In graphviz, `repulsiveforce=-2.0` produces **exactly the default p=-1 layout** (kernel uses `pow(dist, 1-p)=pow(dist,2)`).

dagua (`dagua/layout/ops/pipelines/sfdp.py:539`):
```python
denominator = pow(distance, 1.0 - self.repulsive_exponent)   # p=-2 -> pow(dist, 3.0)
```
plus `_sfdp_force_scales` (`sfdp.py:406-426`) uses p=-2 in `CRK = C**((2-p)/3)/K` and `KP = K**(1-p)`. Nothing in dagua negates or clamps `repulsive_exponent`. So dagua runs a genuinely different force law than the reference it is being scored against.

The harness param-match is correct (`variants.py:1619/1621`: dagua `repulsive_exponent:-2.0` <-> graphviz `repulsiveforce:-2.0`) — the bug is that graphviz *internally* discards the -2.

### Fix sketch (pick ONE; this is a value judgement for JMT)
The variant's *intent* matters. Two readings:
- **(i) Faithful-to-graphviz (recommended):** Replicate graphviz's parse: clamp `repulsive_exponent` per the negate+`>=0->-1` rule. Then `classic_sfdp_p_neg2` becomes identical to `classic_sfdp_default` and should inherit its ~21-divergent profile. Net win ~52 combos (73->~21). One-liner in `layout_sfdp_pipeline`/`_sfdp_force_scales` input normalization. **Source-faithful, no delegation.**
- **(ii) The variant is meant to probe p=-2 as a real exponent:** Then graphviz CANNOT produce p=-2 via CLI at all -> there is no valid reference, and the variant should be dropped or its reference regenerated with an instrumented graphviz that bypasses the clamp. Do NOT score dagua-p=-2 against graphviz-p=-1.

Either way, scoring the current pair is invalid. (i) is the clean, ship-it path. **Confidence HIGH that 73 are mis-scored; HIGH that ~52 convert to non-divergent under (i).**

Caveat to VERIFY on benchmark path: confirm graphviz reference rows for `classic_sfdp_p_neg2` are byte-identical to `classic_sfdp_default` rows (same seed). If they ARE identical, that *proves* graphviz clamped — strongest possible evidence; run before implementing.

---

## BUCKET B — disconnected graphs, missing component packing (48 combos, HIGH ROI)

### Evidence
Among the 111 non-p_neg2 divergent:
- **48 flagged `disconnected=True`**, disp **median 0.121** (native ~8x too compact).
- The 63 *connected* divergent have disp **median 1.000** (the real floor band).
- Enrichment: disconnected = **48/111 (43%)** of divergent vs **10/394 (2.5%)** of non-divergent sfdp. Overwhelming.
- Graph names confirm: `disconnected_encoder_residual`, `disconnected_label_cycle_collage`, `multi_component_80`, `clustered_medium_5x20`, etc. fail across all 5 non-p_neg2 variants (20 graphs diverge in >=5 variants).

### Root cause (graphviz source)
`_references/graphviz/lib/sfdpgen/sfdpinit.c:269-288`:
```c
ccs = ccomps(g, &ncc, 0);
if (ncc == 1) { sfdpLayout(g, &ctrl, pad); ... }
else {
    for (i=0;i<ncc;i++){ sg=ccs[i]; ... sfdpLayout(sg,&ctrl,pad); ... }
    packSubgraphs(ncc, ccs, g, &pinfo);   // pack/pack.c polyomino tiling
}
```
graphviz lays out **each component in its own multilevel spring-electrical run** (own coarsening, own `srand(random_seed)` reset inside each `sfdpLayout->spring_electrical_embedding`), then tiles them with the polyomino packer. dagua's pipeline (`pipelines/sfdp.py:910-920`) builds ONE hierarchy over the whole graph and runs ONE force field — inter-component repulsion has no attractive counterbalance, so the global normalization (`SFDPFinalizePositions`) collapses each component to a fraction of its reference scale -> disp 0.12. Confirmed: `grep component/pack` in `sfdp.py`/`pipelines/sfdp.py` finds **zero** component handling (only an unrelated `_principal_component_rotate`).

This is the **identical bug class** already fixed for siblings: git log `b3f7395` "connected-component decomposition (OGDF DIVIDE_ET_IMPERA) was the over-dispersion root cause" (FMMM), `79a2ac5` "FMMM OGDF component packing", `786f32b` "neato: port polyomino component packing". **sfdp was simply missed in that sweep.**

### Fix sketch
Reuse neato's existing graphviz polyomino packer — `dagua/layout/ops/pipelines/neato.py` already has `_weak_components` (`:82`), `_slice_component_edges` (`:124`), `_compute_polyomino_step` (`:227`), all ports of graphviz `pack.c`. Wrap `build_sfdp_pipeline`:
1. `_weak_components(edge_index, num_nodes)`.
2. if `len==1`: current path.
3. else: per component, slice local edges, run the graphviz-fidelity sfdp sub-pipeline **with `srand`/`GraphvizRandom` reset to `seed` per component** (mirror graphviz calling `sfdpLayout` fresh each time — the per-component RNG reset is essential for mode-A seed match), then map back into global indices and call the neato polyomino pack.
- CRITICAL: match graphviz component ORDER (`ccomps` order) for the RNG-stream and packing-order match, and the per-component `random_seed` reset, or you trade a scale bug for an RNG bug. VERIFY against a 2-3 component reference graph (e.g. `multi_component_80`) on the benchmark path.

**Confidence HIGH on root cause; MED-HIGH that the full 48 convert** (some multi-component graphs may also touch Bucket-C floor inside a component, but disp 0.12 -> ~1.0 is the dominant fix). Expect most of the 48 to reach rung 1-3.

---

## BUCKET C — connected, disp≈1.0: the (under-proven) FP floor (~63 combos)

### Current state of evidence
- 63 connected divergent, disp median **1.000**, range still wide (0.046-8.36 over all 111; the connected subset clusters at 1.0).
- **0/63 pass the stress battery** (`battery_stress_direct_equivalent`/`stress_direct_equivalent` all False). So r71's claim that 42 SFDP cases were "quality/stress-equivalent" does NOT survive r73's stricter battery — these are real divergences, not laundered floor. Good (guardrail 1 intact) but means "floor" here = "we can't fix it", NOT "it's equivalent".
- r71 diagnosis (`r71_sfdp_basin_divergence_diagnosis.md`) verified: RNG seed match (libc srand/rand vs GraphvizRandom for seeds 1/42/100/123), coarsening structure match, force-kernel source parity. BUT it explicitly could NOT extract a reference iteration-0 or per-iteration trace (graphviz 7.0.5 ignores `maxiter` for sfdp on that build), so the floor was **asserted from source-parity, never demonstrated by a divergence experiment.** Per guardrail 5, that is insufficient.

### FLOOR-PROVING EXPERIMENT (spec for a Codex implementer)
Goal: distinguish TRUE basin chaos (1-ULP perturbation -> different attractor by iter ~100) from a hidden systematic offset (perturbation -> same attractor + constant drift = bug).

Build an **instrumented graphviz 7.0.5** (the missing piece — r71 was blocked exactly here):
1. Patch `spring_electrical_embedding` to dump `x[]` to a file every iteration (gate behind env `SFDP_TRACE`), and to honor a real `maxiter` so a 0-iter coordinate dump is available. Build only `libsfdpgen` + a tiny driver; no full graphviz install needed.
2. Pick 5 connected disp≈1.0 graphs: `real_karate_34`, `weighted_karate_34`, `er_100`, `small_world_100`, `hexagonal_lattice_42`.

Experiment 1 — **dagua self-chaos** (no graphviz needed, run first, cheap):
- Run dagua sfdp fidelity twice: baseline, and with the coarsest-level initial positions perturbed by 1 ULP on a single coordinate (`np.nextafter`). Trace per-iter RMSD(baseline, perturbed) on the FINEST level.
- TRUE FLOOR signature: RMSD grows ~exponentially (positive finite-time Lyapunov exponent), saturating near the inter-attractor distance by iter ~50-100; final layouts Procrustes-distinct.
- BUG signature: RMSD stays ~1 ULP * condition number, bounded, never reaches attractor-scale -> the divergence vs graphviz is NOT chaos, it's a deterministic offset -> hunt the offset.

Experiment 2 — **dagua-vs-instrumented-graphviz per-iteration** (the decisive one):
- Same seed, same coarsening (verify hierarchy sizes match — r71 already showed they do). Align both traces by Procrustes at iter 0, 1, 5, 10, 50, 100, 500.
- CONFIRM FLOOR if: traces agree to ~1e-12 for the first few iterations then diverge at a rate consistent with Exp-1's Lyapunov exponent (chaos amplifying libm `sqrt`/`pow`/quadtree-summation-order ULP differences). The early-iteration agreement is the proof the port is correct.
- REFUTE FLOOR (=bug) if: traces diverge at iter 1 (force-law/scale bug), OR diverge with a *systematic constant ratio* (e.g. dagua coords always 1.07x — a normalization/K-decay bug), OR diverge only after a specific level transition (prolongation/`refinement_k_decay` bug). A constant-ratio or single-step jump is a one-liner, not floor.

Decision rule: Exp-1 bounded RMSD OR Exp-2 iter-1 mismatch => REFUTE, open a bug. Exp-1 Lyapunov growth AND Exp-2 early-agreement-then-chaotic-divergence => CONFIRM floor, label `SFDP_FP_BASIN_RESIDUAL`, accept the ~63.

### C-2 systematic suspects to rule out DURING the floor experiment (don't assume floor)
- **Quadtree summation order** (`pipelines/sfdp.py:556-578`, `barnes_hut_threshold=45`): for N>=45, dagua uses `GraphvizQuadTree`. Aggregation order vs graphviz `QuadTree_get_repulsive_force` is the most likely *systematic* (not FP) divergence — a different traversal order is a deterministic difference, not basin chaos. Many of the 63 are N>=45 (`er_100`, `small_world_100`, `random_dag_200`). Worth a direct single-iteration force-vector diff vs instrumented graphviz on one N>=45 graph BEFORE accepting floor.
- **Adaptive cooling split** (`pipelines/sfdp.py:656`, coarsest `adaptive_cooling=True`, finer `False`): confirm graphviz uses the same per-level split (`update_step(adaptive_cooling,...)` is called every level in graphviz — check whether graphviz disables adaptive cooling on prolongated levels or keeps it ON). If dagua turns it OFF on finer levels but graphviz keeps it ON, that is a systematic step-size divergence affecting ALL connected graphs, not floor. **This is the single most suspicious line** — graphviz `spring_electrical_embedding` does not appear to vary `adaptive_cooling` by level; verify. MED confidence this is a real bug hiding in the floor bucket.
- **`refinement_k_decay=0.75`** (`pipelines/sfdp.py:797`): graphviz K-decay per level — confirm 0.75 matches graphviz's prolongation K schedule (graphviz uses `ctrl->K` fixed + coordinate scaling on prolongation, NOT a 0.75 decay per level in all builds). If dagua decays K but graphviz doesn't, systematic.

These three (quadtree order, adaptive-cooling-by-level, K-decay) are the highest-value things to check before declaring the 63 a floor — any one being wrong is a deterministic bug affecting the WHOLE connected bucket, exactly like r72/r73 "floors" that were one-liners.

---

## ROI-ordered action list
1. **A (p_neg2 clamp)** — 73 combos, one-liner, HIGH conf. First verify graphviz ref rows for p_neg2==default (proves clamp). ~52 net win.
2. **B (component packing)** — 48 combos, reuse neato polyomino packer + per-component RNG reset, HIGH conf root cause. Most reach rung1-3.
3. **C-2 adaptive-cooling-by-level + quadtree-order check** — could reclaim a chunk of the 63 connected "floor"; do this BEFORE labeling floor. MED conf there's a bug here.
4. **C floor proof** — instrument graphviz, run Exp-1/Exp-2; CONFIRM/REFUTE. Only after 3.

Honest split: **>=121 of 184 salvageable** (A=73 mis-scored, B=48 systematic), **<=63 candidate-floor and even those under-proven** — at least the N>=45 / adaptive-cooling subset of C should be treated as suspect, not floor, until Exp-2 shows iter-1 agreement.

## Guardrail compliance
- No laundering: 0/63 connected pass stress battery; not reclassifying via weakened FDR.
- No runtime delegation in any fix (port graphviz parse + reuse dagua's own polyomino packer).
- All fixes to be verified on the benchmark path with matched seed+params (esp. p_neg2 ref-row check, multi_component_80 packing check).
- Floor claim explicitly gated on the perturbation/Lyapunov experiment, not asserted.
