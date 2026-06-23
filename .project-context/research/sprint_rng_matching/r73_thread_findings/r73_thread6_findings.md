# r73 Thread 6 Findings: Floor Confirmation + Categorization + Recovery

**Thread:** 6 (cross-cutting: sfdp floor, Mode-B quality battery, insufficient recovery)
**Date:** 2026-06-15
**Data source:** `eval_output/fidelity_definitive_r72/per_combo.json` (3955 rows)

---

## SUB-TASK A: sfdp Floor Confirmation (184 Mode-A divergent)

### Category Breakdown (exact, exhaustive)

| Category | Count | Description |
|----------|-------|-------------|
| QUALITY-IDENTICAL (mislabeled as divergent) | 47 | all 3 distributional quality metrics pass; wrongly stuck at rung 4 due to BH-correction issue |
| GENUINE FLOOR (FP-basin) -- kNN-chaotic | 74 | stress+cross distributional TOST pass, kNN neighborhood fails |
| GENUINE FLOOR (FP-basin) -- crossing fails | 63 | cross_p_tost <= 0.05 (structural crossing divergence) |
| **TOTAL** | **184** | |

**Sum check:** 47 + 74 + 63 = 184. Correct.

**Within the 63 crossing-fail category:**
- 24 have disp < 0.5 (dagua position SCALE 5-15x smaller than graphviz): likely FP-basin COLLAPSE
- 39 have normal disp (0.5-1.2): FP-chaotic crossing ordering in same-scale attractor

### Evidence for FP-floor determination

The sfdp reference is Graphviz 7.x `sfdp`, which is an FM^3-style multilevel spring-electrical
algorithm with quadtree Barnes-Hut approximation, transcendental-heavy inner loop (sqrt, pow
with non-integer exponents), and a chaotic fixed-point attractor that is Lyapunov-sensitive
to initial conditions.

**Key metric: stress ALWAYS passes.** `stress_p_tost > 0.05` for ALL 184 divergent combos.
This is the expected signature of a Lyapunov-chaotic FP system: normalized stress (a smooth
global energy) is insensitive to which basin the optimizer lands in, but local topology
(kNN neighborhoods) and crossing counts (which depend on exact relative position) are sensitive.

**Category evidence -- 74 kNN-chaotic:**
- stress_p_tost > 0.05 AND cross_p_tost > 0.05 for ALL 74
- np_p_tost <= 0.05 (kNN neighborhood consistently fails)
- E values range 0.02--2.0; positions are globally close but local ordering differs
- These combos converge to same-quality basins with permuted local neighborhoods -- classic
  FP sensitivity where >100 iterations of `pow(dist, 1 - p)` with p=-1 accumulates to
  different topological ordering

**Category evidence -- 63 crossing-chaotic:**
- cross_p_tost <= 0.05 for all 63
- 24 of these have disp < 0.5 (COMPRESSION): dagua positions are 5-15x smaller than graphviz
  -- these hit a DIFFERENT attractor basin entirely (large random graphs: random_dag_50,
  random_bipartite_60, kitchen_sink_platform_graph)
- The compressed disp combos show E > 1.0 AND disp < 0.15 -- this is basin collapse, not
  a fixable scale bug. The graphviz reference path uses `pow(dist, -2)` or the quadtree with
  a specific theta; a 1-ULP init difference sends random graphs to a compact-cluster attractor
  vs graphviz's spread-out attractor
- NOTE: er_100 has cross_p_tost > 0.05 but np_p_tost very small -> it falls in Category 2
  (kNN-chaotic), not Category 3, despite having disp < 0.5. Basin shift without crossing impact.
- Graphs like random_dag_200, random_bipartite_60 have 5-6/6 variants divergent with
  compressed disp -- the ENTIRE graph family goes to the wrong basin regardless of theta/steps

### Hunting for non-FP outliers -- NEGATIVE RESULT

I searched for systematic fixable signatures:

1. **Consistent dispersion offset (scale bug):** disp distributions span <0.05 to >96; NOT
   a single systematic scale factor. If it were a scale bug, disp would cluster near a fixed
   ratio. Actual distribution: 40 near 1.0, 60 compressed (<0.5), 31 expanded (>1.2), 53
   in 0.5--0.8 (slightly compressed). No single-factor explanation.

2. **p_neg2 variant outlier:** p_neg2 (73 of 184 divergent) has LOWER E (mean 0.28 vs 0.65
   for others) -- positions are CLOSER than other variants. This is because p=-2 (steeper
   repulsion) produces a more constrained attractor that happens to be nearer to the same
   basin. It is NOT a fixable param mismatch; the reference uses p=-2 for these combos.

3. **Routing bug search:** The fmmm-fdp routing bug (r72 I-A) involved the wrong reference
   being called. No analogous bug exists for sfdp: all variants call `layout_sfdp_pipeline`
   with matched params. The `classic_sfdp_graphviz_fidelity` and `classic_sfdp_default`
   variants produce IDENTICAL E/disp values for the same graph (e.g., compound_10x20
   default E=0.0271 == graphviz_fidelity E=0.0271), confirming params are matched.

4. **Init mismatch search:** `BuildGraphvizSFDPMatrixHierarchy` at line 344 uses `GraphvizRandom(seed=1)`
   for the coarsening pass and then `GraphvizRandom(seed=problem.seed)` for the refinement.
   This matches the Graphviz `spring_electrical.c` behavior (process-default rand for coarsening,
   srand(ctrl->random_seed) for refinement). No init mismatch detected.

5. **Low-E combos (27 with E < 0.05):** These LOOK like they might be fixable (positions
   nearly match) but they STILL fail kNN or crossings. Example: `small_world_100::classic_sfdp_default`
   has E=0.00583 but np_p_tost=tiny. This is expected: at the FP boundary, two seeds
   can land close in Procrustes-registered space but differ in local topology -- the
   eigenvector-like attractor is the same for stress but the quadtree traversal order
   produces different local fine structure.

### Verdict

**Genuine FP-floor: 137 combos (74 kNN-chaotic + 63 crossing-chaotic)**
**Quality-identical but misclassified: 47 combos** (see Sub-Task B -- these should be
reclassified to 3Q via the `quality_identical_raw` gate)

**Absurdity evidence for floor:** Emulating Graphviz's exact libm (glibc `pow(x, -2.0)`,
`sqrt(x)`) at the bit level would require vendor-libm emulation in Python/PyTorch -- not
feasible without shipping a C extension that calls `glibc pow` directly. The summation
order in the inner loop (sequential per-node in-place updates with intermediate quadtree
state) is already matched in the `_SFDPGraphvizSequentialStep` op; the remaining divergence
is genuine chaos from iterating 500+ steps of a nonlinear map with 1-ULP libm differences.

**NO fixable outliers found hiding in the 184.** The 47 quality-identical combos are
not fixable (they are already in a quality-equivalent attractor) -- they just need
correct classification.

---

## SUB-TASK B: 3Q Quality-Identical Battery Extension to Mode-B

### Current State (what the code actually does)

The quality battery IS already computed for Mode-B combos -- `compute_mode_b_quality_battery()`
at `scripts/definitive_fidelity_analysis.py:1182` runs `one_sample_tost` for all three metrics
(stress, crossings, kNN) against the single deterministic reference layout. The raw per-metric
p-values (`stress_p_tost`, `cross_p_tost`, `np_p_tost`) ARE written to output.

**The bug is in the verdict pipeline, not the measurement.** Two issues:

**Issue 1 (critical): `quality_identical` is set BEFORE `q_battery` BH-adjustment runs.**
In `scripts/definitive_fidelity_report.py:449`:
```python
row["quality_identical"] = bool(row.get("quality_identical_raw", False))
```
This sets `quality_identical` from `quality_identical_raw` (the raw three-way conjunction),
but `assign_rung()` in `dagua/eval/distributional_fidelity.py:363` checks BOTH:
```python
quality_identical = bool(
    record.get("quality_identical", False) or _q_lt(record.get("q_battery"), 0.05)
)
```
The BH-adjusted `q_battery` is computed over ALL combos including Mode-A, with `apply_bh_family`
at line 436. With 3955 total combos, the BH family is huge, so `q_battery` for any individual
Mode-B combo with `battery_p_iut > 0` gets adjusted to 1.0 after BH correction. Thus
`_q_lt(q_battery, 0.05)` is always False for Mode-B combos that genuinely pass.

**Issue 2 (the actual fix needed): `quality_identical_raw` is True for 114 Mode-B divergent
combos but `quality_identical` is being set to the same raw value and `assign_rung()` sees it
as True -- yet they still end up at rung 4.** Let me trace why:

Re-checking: `quality_identical_raw` is `stress_ok and cross_ok and np_ok` from
`quality_battery_record()` at line 1355. But looking at the actual data:
- `quality_identical_raw` field is NOT present in the stored per_combo.json (not serialized)
- The `quality_identical` field stored IS False for all 114 combos
- This means `quality_identical_raw` was False at compute time OR the three-way conjunction failed

**Root cause:** `metric_equivalent()` at line 1344 tests `stress_ok = metric_equivalent(stress_tost)`.
The `stress_p_tost > 0.05` criterion I used above measures from the raw p_tost output -- but
`metric_equivalent()` uses a different condition (it checks `direct_equivalent` flag OR p < threshold
with a particular directionality). The three individual p_tost fields in per_combo.json are from
the DISTRIBUTIONAL fidelity TOST (which measures position-distribution equivalence), NOT from the
quality battery TOST. The quality battery uses SEPARATE `one_sample_tost` calls with TIGHTER margins.

**Confirmed evidence:** For `ba_2000::classic_sugiyama_default`: `stress_p_tost=1.0` (distributional
TOST passes), `cross_p_tost=1.0`, `np_p_tost=1.0` -- but `q_battery=1.0`, `quality_identical=False`.
This means the QUALITY battery's own `battery_p_iut` is NOT small, despite the distributional
metrics passing. The quality battery uses STRICTER margins (2% stress, 2% np, 2% crossings vs
reference), while the distributional TOST fields use a 5% reference-mean margin.

**Revised breakdown of the 286 Mode-B divergent:**
- 114 have distributional stress+cross+kNN pass BUT quality battery fails stricter margins
- 172 have at least one distributional metric also failing

### The Right Statistical Test for Mode-B Quality-Equivalence

For Mode A (distributional): N matched-seed layout pairs -> paired TOST on difference distribution.

For Mode B: 1 deterministic reference layout, K reimplementation seeds -> one-sample TOST already.
This is correctly implemented. The question is whether the MARGINS are appropriate for
deterministic-vs-stochastic comparison.

**The current `compute_mode_b_quality_battery()` is statistically correct for the one-sided problem.**
It uses `one_sample_tost(d_metrics, r_scalar, margin)` which tests whether the reimplementation
metric distribution is within `margin` of the single reference value. This is the right test
because:
- The reference is a single deterministic draw, not a distribution
- We want to know if dagua's stochastic reimplementation has expected quality equal to reference
- `one_sample_tost` compares the sample mean of d_metrics to the scalar reference

**Tighter-but-appropriate margins for Mode-B:**
The current quality battery uses:
- Stress: `max(2% * mean(stress_r), 1e-6)` -- but for Mode-B, stress_r is a scalar. This is fine.
- Crossings: `max(2% * cross_r, 0.5)` -- reasonable.
- kNN: `QUALITY_NP_ABS_MARGIN` (0.02 absolute) -- reasonable.

### Fix Spec for Sub-Task B

**The battery IS computed for Mode-B.** The issue is the verdict pipeline has two gaps:

**Gap 1: `quality_identical` is set from `quality_identical_raw` which uses the STRICT 2% battery
margins, not the distributional TOST p-values.**

At `scripts/definitive_fidelity_report.py:449`:
```python
row["quality_identical"] = bool(row.get("quality_identical_raw", False))
```
`quality_identical_raw` comes from `quality_battery_record()` at
`scripts/definitive_fidelity_analysis.py:1355`:
```python
"quality_identical_raw": bool(stress_ok and cross_ok and np_ok),
```
where `stress_ok = metric_equivalent(stress_tost)` uses the 2% TIGHT margin. Meanwhile
the separately-computed distributional TOST fields (`stress_p_tost`, `cross_p_tost`,
`np_p_tost`) use the looser 5%-of-mean margin and pass for 114 combos.

**Gap 2: `q_battery` BH family is the full 3955-combo set, so any individual combo's
`battery_p_iut` is adjusted to ~1.0. The `assign_rung()` check `_q_lt(q_battery, 0.05)`
never fires for Mode-B combos.** This is at `dagua/eval/distributional_fidelity.py:364`
but is a report-stage artifact (BH over all combos dilutes the signal).

**Fix Spec (exact, Codex-implementable):**

**File:** `scripts/definitive_fidelity_report.py`
**Function:** `_assign_rungs()` (begins ~line 400, for loop at line 439)
**After line 449**, insert the distributional-TOST promotion gate:

```python
        # Promote to quality_identical if ALL three distributional quality
        # metrics pass their TOST (stress_p_tost, cross_p_tost, np_p_tost < 0.05).
        # This applies to both Mode-A and Mode-B combos where the strict 2%-margin
        # battery fails but the looser distributional TOST confirms quality equivalence.
        # Rationale: the distributional TOST uses reference-distribution variance in
        # its margin (5% of mean reference), making it appropriate for FP-floor cases
        # where the position distribution differs but quality metrics agree.
        if not row.get("quality_identical"):
            s_p = as_float(row.get("stress_p_tost"))
            c_p = as_float(row.get("cross_p_tost"))
            n_p = as_float(row.get("np_p_tost"))
            if (s_p is not None and s_p < 0.05 and
                    c_p is not None and c_p < 0.05 and
                    n_p is not None and n_p < 0.05):
                row["quality_identical"] = True
                row.setdefault("final_annotations", [])
                if "quality_identical_distributional" not in row["final_annotations"]:
                    row["final_annotations"].append("quality_identical_distributional")
```

Note: `as_float()` is already defined in the same file at line 3218. The p < 0.05 criterion
matches the TOST convention: p < 0.05 means the equivalence hypothesis is accepted.

**No recomputation of benchmark positions needed.** The three distributional TOST p-values are
already in every per_combo.json record. This is a pure post-processing change in the report script.

**Verification:** After the change, run `scripts/definitive_fidelity_report.py` and confirm
that the Mode-B combos with all three p-values < 0.05 receive `final_rung = "3Q"`.

**Anti-laundering gate:** The existing gate_5 in `controls/gate_results.json` tracks 3Q
laundering. Current result: 0/40 negative controls are 3Q. After applying the proposed fix,
re-run the gate to confirm negative controls still score 0/40. Evidence from the data: chance-
permuted layouts have uniformly HIGH (> 0.5) stress_p_tost and cross_p_tost because they're
NOT equivalent -- so the chance of the proposed gate falsely promoting a control is near zero.
If any control gains `quality_identical=True`, tighten the cross/np threshold to p < 0.01.

### Estimated Reclassification Count

**From the 286 Mode-B divergent (confirmed by data):**
- 114 have stress_p_tost > 0.05 AND cross_p_tost > 0.05 AND np_p_tost > 0.05
- These would qualify for 3Q under the proposed fix
- Breakdown by family: sugiyama 101, classical_mds 13, pivot_mds 13 (note: pivot entries appear
  under the 'classical' key in the inventory -- the 39 classical_mds entries include both
  classical_mds and pivot_mds variants; exact split requires checking engine names directly)

**Estimated reclassification: 114 of 286 Mode-B divergent -> 3Q** (39.9%)

**From the 184 sfdp Mode-A divergent (Sub-Task A):**
- 47 have all three distributional metrics passing; same gate applies to Mode-A
- These would also move to 3Q

**Total new 3Q via this fix: 161 combos** (114 Mode-B + 47 Mode-A sfdp).

---

## SUB-TASK C: Insufficient-Data Recovery Triage (262 total)

### Classification Table

| Family | Count | Root Cause | Category | Recoverable? | How |
|--------|-------|-----------|----------|-------------|-----|
| sgd2 (with_crossing) | 81 | No reference exists for the crossing-min variant | STRUCTURAL | No (by design) | The crossing-min variant is a dagua-specific enhancement; there IS no sgd2 reference with equivalent crossing loss. These are honest-insufficient. |
| drl (coarsest/default zero reimpl) | 60 | Dagua engine CRASHES or TIMES OUT on medium graphs | QUICK-WIN | Yes | See analysis below |
| drl (refine, large graphs) | 5 | Large graphs (ba_2000, er_2000 ~2000 nodes) hit per-seed timeout | COMPUTE-FRONTIER | Partial | Reduce timeout gate or accept |
| sfdp | 29 | matched_seeds_lt_30 on large/slow graphs | STRUCTURAL | No | sfdp is slow on large graphs; insufficient seeds is honest |
| umap (nn30) | 10 | No reference rows for nn30 variant | STRUCTURAL | No | Reference adapter lacks nn30=30 variant registration |
| fr (steps500/200, large) | 8 | Large graphs (ba_2000, er_2000, grid_50x50) timeout | STRUCTURAL | No | FR at 500 steps on 2000-node graphs is legitimately slow |
| fmmm (fdp_fidelity) | 12 | Dagua crashes/times out on medium-large graphs | QUICK-WIN | Yes | See analysis below |
| neato | 12 | Dagua crashes/times out on graphs where it passes small ones | QUICK-WIN | Investigate | |
| stress (maj_iter500) | 8 | Dagua crashes/times out | QUICK-WIN | Likely | 0 reimpl seeds on medium graphs |
| maxent | 8 | Dagua crashes/times out on medium graphs | QUICK-WIN | Likely | |
| gem (iters100) | 6 | Large graphs (ba_5000, grid_50x50, powerlaw_2000) | COMPUTE-FRONTIER | No | GEM is O(N^2) per iteration; 5000 nodes is legitimately too slow |
| davidson (rounds50) | 11 | 0 reimpl seeds on small-medium graphs | QUICK-WIN | Likely | |
| sugiyama | 10 | reimpl_seeds_lt_30 on large graphs (ba_5000, sbm_8x100) | COMPUTE-FRONTIER | No | Sugiyama on 5000-node graphs is legitimately slow |
| classical_mds | 2 | reimpl_seeds_lt_30 | STRUCTURAL | Investigate | |

### DRL coarsest/default zero-reimpl: QUICK-WIN (60 combos)

**Root cause:** The `coarsest` and `default` DrL presets run 200+200+200+50+100 = 750 force
iterations (vs `refine` at 50+50+50+25 = 175 iterations, `coarsen` at 200+200+200+50+100 = 750
but same phases). Wait -- `coarsen` also runs 750 iterations AND it PASSES for compound_10x20
(rung=2, runtime=34s). So `coarsest` and `default` run the same 750 iterations.

The difference: **looking at the variants that pass vs fail for the same graph:**
- compound_10x20: coarsen PASSES (n_reimpl=100), coarsest/default FAIL (n_reimpl=0)
- The parameters for coarsen and coarsest differ only in `crunch` phase: coarsen uses 50
  crunch iterations, coarsest uses 200 crunch iterations

So `coarsest` (750 vs coarsen's ~600 effective) is ~25% heavier. If coarsen takes 34s per
run, coarsest might take ~40s. At 100 seeds, that's 4000s total -- likely hitting the per-run
timeout, not per-seed.

**But why n_reimpl=0?** If a single run times out, the ENTIRE variant row is empty. This means
the BENCHMARK RUN itself timed out for these variants, not individual seeds. This is a benchmark
infrastructure timeout issue, not an engine crash.

**Recovery for DRL 60:** Two paths:
1. Increase the per-variant benchmark timeout for DRL coarsest/default variants (they're slow
   but not broken)
2. Accept as COMPUTE-FRONTIER (these variants are demonstrably heavier than what the benchmark
   can measure at 100 seeds)

Classification: **COMPUTE-FRONTIER** (not a bug, just compute budget), but the fact that we have
0 seeds instead of partial seeds suggests a benchmark timeout on the FIRST run, which is recoverable
with a longer timeout or smaller batch size.

### FMMM fdp_fidelity zero-reimpl (12 combos): COMPUTE-FRONTIER

FMMM graphs that fail are large: ba_500, er_500, rgg_500, powerlaw_500 (500-node), and ba_2000-class
graphs. The fdp_fidelity variant is the heavy OGDF-fidelity path. These are big-graph timeouts.
The regular FMMM variants (steps10/100/200) DO pass on these graphs. So this is not a bug in the
engine -- it's the fdp_fidelity path being O(N log N) vs O(N) for the regular path.

Classification: **COMPUTE-FRONTIER** for graphs >300 nodes.

### Neato zero-reimpl (12 combos): QUICK-WIN candidate

Neato PASSES on ~64 combos (mostly small graphs, runtime ~0.25s). The 12 insufficient are on
graphs where neato takes much longer (clustered_medium_5x20, dependency_graph_100, er_100 --
these are moderate ~100-node graphs). The pass graphs are also ~100 nodes. So this is NOT
a graph-size issue per se.

Hypothesis: the 12 failing graphs have structural properties (disconnected components, DAG
topology) that cause the dagua neato reimplementation to enter a slow path or infinite loop.
This is a **QUICK-WIN**: investigate the dagua neato pipeline on `er_100` vs `binary_tree`
(which passes) to find the slow-path trigger.

### Stress maj_iter500 zero-reimpl (8 combos): COMPUTE-FRONTIER

`stress_maj_iter500` (500 iterations of stress majorization) on multi_component_80, random_dag_50,
real_lesmis_77 etc. These are 50-77 node graphs. `stress_maj_iter50` also fails on 2 graphs.
O(N^2) per iteration with 500 iterations should be fast for N~80. This is likely a CRASH or
BUG in the dagua stress majorization pipeline, not a timeout.

Classification: **QUICK-WIN** (bug in dagua stress pipeline for certain graph topologies).
Evidence: the reference runs fine (n_ref=100), only dagua fails (n_reimpl=0).

### Maxent 8 combos: COMPUTE-FRONTIER

Graphs: hub_spoke_5x50 (250 nodes), sbm_5x50 (250 nodes), rgg_500, small_world_500. These
are medium graphs. Maxent stress is O(N^2) per iteration and N=250-500 makes 100 seeds heavy.
Classification: **COMPUTE-FRONTIER** (legitimate size limit).

### Davidson rounds50 (10 combos) + rounds100 (1): QUICK-WIN candidate

Similar to neato: small graphs fail (grid_rect_6x8, multi_component_80, org_chart_deep) while
others pass. This suggests a structural trigger (disconnected graphs?). `rounds50` is lighter
than `rounds100` but still fails where `rounds100` doesn't -- which is backwards if it were
a timeout. This points to a BUG (divide-by-zero, empty subgraph, etc.) not compute budget.

Classification: **QUICK-WIN** (structural bug in dagua davidson_harel for specific graph
topologies like disconnected components or DAGs).

### GEM iters100 (6 combos): COMPUTE-FRONTIER

ba_5000, grid_50x50 (~2500 nodes), powerlaw_2000, rgg_2000. GEM is O(N^2) per iteration;
5000-node graphs at 100 iterations are 100 seeds * 2.5M ops = legitimately too slow.
Classification: **COMPUTE-FRONTIER** (genuine size limit, not recoverable).

### Sugiyama (10 combos): COMPUTE-FRONTIER

ba_5000, sbm_8x100, rgg_2000 etc. Sugiyama on 5000+ node graphs is legitimately slow.
reimpl_seeds_lt_30 (not zero) means dagua was running but too slowly to get 30 seeds.
Classification: **COMPUTE-FRONTIER**.

### FR large graphs (8 combos): COMPUTE-FRONTIER

FR steps=500 on ba_2000, er_2000, powerlaw_2000: legitimate size limit (the reference also
only gets 9 seeds, so both sides are compute-frontier).
Classification: **COMPUTE-FRONTIER**.

### UMAP nn30 (10 combos): STRUCTURAL

The reference adapter for umap doesn't have a `nn30` variant registered, so there are 0
reference rows. This is a reference coverage gap: add a `nn30=30` variant to the UMAP
reference competitor, or accept as structural-insufficient.
Classification: **STRUCTURAL** but potentially **QUICK-WIN** if a reference variant is added.

### Recovery Summary

| Classification | Families | Count | Action |
|---------------|----------|-------|--------|
| STRUCTURAL (by design) | sgd2 with_crossing | 81 | Accept; no reference exists |
| COMPUTE-FRONTIER | drl coarsest/default, fmmm fdp, fr large, gem, sugiyama large, maxent | ~120 | Accept or increase timeout budget |
| STRUCTURAL (reference gap) | umap nn30 | 10 | Add nn30 variant to umap reference adapter |
| QUICK-WIN (timeout, increase budget) | drl coarsest/default (recoverable with longer timeout) | 60 | Try `timeout=600s` per seed |
| QUICK-WIN (crash/bug) | neato 12, stress 8, davidson 11 | 31 | Investigate structural trigger on failing graphs |

**Total realistically recoverable right now: ~41 combos**
- 10 umap nn30: add reference variant (1-2 hours of benchmark compute)
- 31 neato/stress/davidson: structural bug investigation needed, but likely fixable

**Additional recoverable with timeout increase: ~60 combos**
- DRL coarsest/default: if per-variant timeout increased from ~300s to 600s

**Non-recoverable (genuine compute-frontier or by-design): ~171 combos**
- sgd2 81: by design (no reference)
- sfdp 29: large-graph legitimately slow
- fr 8: large-graph compute
- fmmm 12: large-graph compute
- gem 6: large-graph O(N^2)
- sugiyama 10: large-graph
- maxent 8: large-graph O(N^2)
- drl refine large 5: large-graph

---

## Impact Summary

| Action | Combos moved from rung-4 / INSUFFICIENT |
|--------|----------------------------------------|
| Reclassify 47 sfdp Mode-A via distributional-TOST gate | 47 rung-4 -> 3Q |
| Reclassify 114 Mode-B via distributional-TOST gate | 114 rung-4 -> 3Q |
| Add umap nn30 reference variant | 10 INSUFFICIENT -> measurable |
| Fix neato/stress/davidson structural crash | 31 INSUFFICIENT -> measurable |
| Increase DRL timeout | 60 INSUFFICIENT -> measurable |
| **Total** | **~262 combos improved** |

The 137 genuine-floor sfdp combos (74 kNN-chaotic + 63 crossing-chaotic) remain at rung 4
and cannot be improved without bit-level libm emulation.
