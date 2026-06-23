# r74 Cluster O6 — Cross-Cutting Methodology, Divergent Tails, Genuine 3Q Promotion

READ-ONLY research. Data source: `eval_output/fidelity_definitive_r73/per_combo.json` (3,955 rows;
574 rung-4 DIVERGENT; 36 in 3Q; 0 controls in this file — controls live in
`eval_output/fidelity_definitive/controls/*.jsonl`). Gate logic verified in
`dagua/eval/distributional_fidelity.py::assign_rung` and
`scripts/definitive_fidelity_report.py::{finalize_rows,apply_bh_family,deterministic_quality_identical}`.

## THE EXISTING 3Q GATE (verified, exact)
`assign_rung` promotes to 3Q iff `quality_identical = quality_identical_raw==True OR q_battery<0.05`,
where `q_battery` = BH-FDR-corrected `battery_p_iut` over the full non-control family (size 3,692).
`battery_p_iut` is the IUT across the STRICT battery: stress (2% rel margin, `battery_stress_*`),
crossings (`max(2%*cross_R, 0.5)`), kNN/np (absolute 0.02). The loose `stress_p_tost` (5% margin)
is NOT in this gate. Controls gate `gate_5_quality_identical_laundering` requires <=5% of the 40
chance+negative controls to land in 3Q. Current measured: **0/40 = 0.0% (clean).**

================================================================================
## MISSION 3 (do first — it's the cleanest result): GENUINE 3Q PROMOTIONS = ZERO
================================================================================

**HONEST COUNT UNDER THE EXISTING GATE: 0 legitimate 3Q promotions beyond the current 36.**

Hard evidence (all 574 rung-4 rows):
- rung-4 with `quality_identical_raw==True`: **0**
- rung-4 with recomputed `q_battery<0.05`: **0**
- rung-4 with stored `q_battery<0.05`: **0**
- rung-4 with **RAW `battery_p_iut<0.05` (pre-BH, the most generous possible read)**: **0**

Not a single divergent combo is even a borderline candidate. By raw D/R margins, **0/574** divergent
rows pass ALL three battery components; the universal binding failure is kNN/np (fails in all 570
rows that have battery fields). Closest-to-passing combo by family (min `battery_p_iut`):
- fmmm 0.059 (`broken_symmetry_residual_pair::classic_fmmm_steps100`) — still > 0.05 BEFORE BH
- gem 0.242, sfdp 0.555, umap 0.833, maxent 0.995, drl 0.999, neato 0.9999, sugiyama 1.0, mds 1.0

Current 36 3Q are all legitimate: 36/36 via `quality_identical_raw==True` (31 also have stored
`q_battery<0.05`); families fmmm(32)+classical_mds(4); all mode A. They pass the anti-laundering gate
(0/40 controls). **Do not touch them; do not invent a looser rule. The well is dry for 3Q.**

### ANTI-LAUNDERING: the tempting "parity" rule is the r73 trap, re-armed
There ARE 142 rung-4 rows where dagua's distance-to-ref <= ref's own seed-to-seed spread (D<=1.05R)
on ALL three metrics — they *look* quality-equal. This is exactly the laundering vector r73 killed.
Tested directly against the 40 chance+negative controls:
- "FULL_PARITY (D<=1.05R on all 3)" rule: **22/40 = 55.0% controls pass** (catastrophic).
- Tightest "D<=1.0R on all 3" rule: **5/40 = 12.5%** (same level r73 rejected; > 0% bar).

Why it fails: chance/negative controls (random/wrong layouts) ALSO have D≈R because the metrics are
noisy — a parity framing cannot separate a true match from chance. The strict ABSOLUTE-margin battery
is correct precisely because it rejects this. **No reclassification. Clean negative confirmed.**

================================================================================
## MISSION 1 (highest-leverage real find): SYSTEMATIC CROSS-FAMILY CAUSE
================================================================================

### FINDING: DISCONNECTED-COMPONENT (and multi-edge) HANDLING is a shared-convention divergence
This is a genuine cross-family FIX target (not a reclassification, not laundering).

Evidence:
- **145/574 divergent rows are `disconnected=True`**, spanning sfdp(57), sugiyama(40),
  classical_mds(20), fmmm(18), maxent(3), gem(3), neato(2), umap(2).
- A handful of SHARED GRAPH INPUTS break nearly every engine at once:
  - `random_dag_50` → DIVERGENT in **8 of 10** divergent families
    (classical_mds, fmmm, gem, maxent, neato, sfdp, sugiyama, umap)
  - `random_dag_200` → 6 families
  - `parallel_cycles_4x5` → 5 families (classical_mds, fmmm, neato, sfdp, sugiyama)
  - `disconnected_encoder_residual`, `disconnected_label_cycle_collage`,
    `kitchen_sink_platform_graph`, `multi_component_80` → 4 families each
- Including multi-edge graphs (`parallel_*`, `*multiedge*`): **152/574 (26%)** of all divergence is
  touched by disconnected-or-multiedge inputs.

Confirmation in code: `dagua/layout/ops/pipelines/classical_mds.py` line 228 — "disconnected-graph
DLA packing is the only stochastic behavior"; line 355 — OGDF MDS *requires* a connected graph. So
the disconnected pathway is a known, separately-handled seam where dagua's component packing differs
from the reference's. r73 ported SOME packing fixes (neato polyomino, FMMM MAARPacking Best-Fit), but
the data shows MDS disconnected handling and the multi-edge graphs remain unmatched.

Split of the 145 disconnected divergent rows by stress D/R:
- **73 PARITY (D/R<=1.05)** — equal-quality, components just packed differently (convention mismatch).
  Matching the reference's component arrangement would tighten these, but they are NOT individually
  3Q-promotable (laundering bar above). Value: fidelity-of-arrangement, not rung flip via battery.
- **40 MILD (1.05<D/R<=2.0)**.
- **32 PATHOLOGICAL (D/R>2.0, real layout bugs)** — sfdp(17), classical_mds(10), sugiyama(4), fmmm(1).
  Top blowups are MDS on disconnected graphs: `disconnected_label_cycle_collage` D/R=1,240,780;
  `parallel_cycles_4x5` D/R=124; `disconnected_encoder_residual` D/R=37; `multi_component_80` D/R=13.
  Classical MDS on a disconnected graph has undefined inter-component distances; dagua produces a
  pathological spread vs igraph's packed arrangement. **This is the single highest-ROI fix:** correct
  MDS disconnected-component packing to mirror igraph, which removes the astronomical-stress rows AND
  likely flips many of the 73 parity rows once arrangement matches.

### ROI ranking of the systematic lever
1. **Classical MDS disconnected-component packing** — 10 pathological + ~20 total MDS disconnected
   rows; D/R up to 1.2M. Clear bug, source seam identified (line 355). HIGH ROI, bounded.
2. **Shared multi-edge graph handling** (`parallel_cycles_4x5`, `parallel_multiedge_bundle`) — breaks
   5+ families. r73 added umap CSR coalescing (umap.py:146-158) but kNN still fully diverges (see
   tails). Worth a cross-family multi-edge audit.
3. **General disconnected component-packing convention** — 73 parity + 40 mild rows; matching
   reference arrangement is fidelity polish, not rung flips. Lower ROI per the laundering bar.

### NEGATIVES checked (no systematic cause found in these dimensions)
- **Scale/normalization**: stress D/R medians are NOT systematically >>1; several families sit <1.0
  (umap 0.35, gem 0.58, sugiyama 0.93). No global scale bug.
- **Aspect/coordinate-frame**: `free_aspect=True` for exactly 231 rows = the whole sugiyama-B bucket;
  it tracks the family, not a cross-family frame bug.
- **Seed/RNG stream**: `n_ref_seeded_ok` is 100 for 339 rows and 0 for 235 (deterministic refs);
  `has_ref_deterministic=True` for 561/574. No matched-seed mismatch signature beyond the known
  deterministic-ref split. (matched_seeds field is uniformly None — not populated here.)

================================================================================
## MISSION 2: DIVERGENT TAILS — ROOT CAUSE + BEST ACHIEVABLE TIER
================================================================================

### UMAP (8 mode A) — TWO sub-causes
- **6× `parallel_multiedge_bundle::classic_umap_*`** (default/mindist001/mindist05/nn30/nn5/spread2):
  stress D (0.03–0.10) is BELOW ref spread R (0.119) → global embedding MATCHES; cross D=R=0 (no
  crossings). The ONLY failure is **kNN D=R=1.0** (total k-nearest-neighbor identity disagreement).
  r73 already added parallel-edge CSR coalescing (`dagua/layout/ops/umap.py:146-158`) so distances
  match, but on a pure multi-edge bundle the fuzzy-simplicial neighbor SET / tie-ordering still
  differs completely → kNN cannot agree. Root cause = multi-edge neighbor-tie degeneracy in the
  kNN metric, not a gross layout bug.
  - **Best achievable tier: FLOOR / accepted DIVERGENT.** Stress already passes-by-spread; kNN=1.0 is
    a metric-identity artifact on a degenerate multi-edge graph. Could attempt to match igraph's exact
    neighbor tie-break, but high effort, low certainty; not sprint-scale. Honest tier: 4 (FP/identity
    floor), NOT promotable.
- **2× `random_dag_{50,200}::classic_umap_nn5`** (disconnected=True): D≈R on all three (e.g.
  random_dag_50 stress 0.78 vs 0.71, kNN 0.29 vs 0.24). Disconnected-component placement variance.
  - **Best tier: 4, part of the disconnected systematic (Mission 1).** Could improve via component
    packing match but is genuinely near-floor.

### DRL (5 mode A) — `real_karate_34` (×3) + `real_lesmis_77` (×2)
All five show D≈R on every metric (karate: stress 0.109 vs 0.105, cross 143 vs 138, kNN 0.524 vs
0.524). dagua is as close to ref as ref is to itself — a statistical near-tie that the strict
absolute battery (kNN 0.02) can't certify on these small dense social graphs. Pipeline docstring
(`dagua/layout/ops/pipelines/drl.py`) confirms: "Full-suite parity depends on C++ float rounding and
density-grid boundary behavior." DrL/OpenOrd uses stochastic density-grid annealing with
implementation-defined float-rounding / grid-cell-boundary decisions.
  - **Best achievable tier: FLOOR — accepted DIVERGENT (4).** Genuine quality parity; remaining
    residual is C++ density-grid FP boundary chaos. Not fixable at sprint scale; legitimately at the
    floor (NOT a quality gap — dagua matches, the battery's absolute kNN margin is just too tight to
    certify). This is the honest "equal-quality but uncertifiable" bucket.

### NEATO (2 mode A) — both disconnected=True
- `parallel_cycles_4x5`: stress D=0.0176 vs R=0.0168 (near-equal), cross passes,
  **kNN D=0.833 < R=0.964 (dagua BETTER than ref baseline)** yet fails the absolute 0.02 margin.
- `random_dag_50`: D≈R on all (stress 0.32 vs 0.35, kNN 0.20 vs 0.17). Component packing variance.
  - **Best tier: 4, part of the disconnected systematic.** parallel_cycles_4x5 is genuinely
    equal-or-better quality (convention mismatch). Could flip if neato disconnected packing matches
    reference exactly. Honest tier today: 4-FLOOR.

### Tail summary
All 15 tail rows (umap 8 + drl 5 + neato 2) are at or near the irreducible FLOOR: dagua's quality
matches the reference (D≈R or D<R) but the strict absolute battery can't certify due to kNN-identity
sensitivity on degenerate (multi-edge / disconnected / small-dense) graphs. **None is a real quality
regression; none is sprint-fixable into a higher tier without a controls-validated metric change.**
The only structural lever shared by the tails is the disconnected/multi-edge systematic (Mission 1).

================================================================================
## BOTTOM LINE (ROI-ordered)
================================================================================
1. **[REAL FIX, HIGH ROI] Classical MDS disconnected-component packing.** 10 pathological rows
   (D/R up to 1.2M) + ~20 MDS disconnected total; bug, not floor. Source seam at
   classical_mds.py:355. Mirror igraph's component arrangement.
2. **[REAL FIX, MED ROI] Cross-family disconnected/multi-edge convention.** `random_dag_50` breaks 8
   families, `parallel_cycles_4x5` breaks 5; 152/574 (26%) of divergence touches these inputs. r73
   fixed some packers; MDS + multi-edge remain. 73 parity rows would tighten but mostly NOT flip
   battery (laundering bar).
3. **[3Q] ZERO legitimate promotions beyond 36.** 0/574 divergent rows pass even raw pre-BH
   battery_p_iut<0.05. The 142 "parity" rows are the r73 laundering trap (55% control pass). Clean
   negative — do NOT relax the gate.
4. **[TAILS] umap/drl/neato (15 rows) are at the FLOOR.** Quality matches reference; kNN-identity on
   degenerate graphs can't be certified by the absolute battery. Accept as DIVERGENT-FLOOR; only the
   Mission-1 disconnected/multi-edge work could mechanically improve a subset.

GUARDIAN NOTE: I am the guardian of guardrail #1 this round. I explicitly tested the most attractive
reclassification (D/R parity, 142 rows) against the 40 controls and it lands 55% — REJECTED. No
laundering proposed. The 3Q answer is honestly zero.
