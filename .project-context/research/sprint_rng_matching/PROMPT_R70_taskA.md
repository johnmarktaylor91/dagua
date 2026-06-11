<task>
Implement Task A (stats core) of the r70 definitive fidelity analysis for the dagua repo
(you are in /home/jtaylor/projects/dagua).

AUTHORITATIVE SPEC -- read it FIRST, in full, before writing any code:
  .project-context/research/sprint_rng_matching/SPEC_definitive_fidelity_analysis.md
It is version 6, APPROVED after 5 rounds of adversarial review (51 findings incorporated).
Implement it EXACTLY -- every threshold, guard, convention, and formula is pre-registered.
Where this prompt and the spec disagree, the SPEC wins.

CREATE EXACTLY TWO FILES (touch nothing else):
  1. dagua/eval/distributional_fidelity.py
  2. tests/test_distributional_fidelity.py

The module is the pure statistical core (spec sec. 10, Task A). A separate runner (Task B)
will call it later, so expose a clean public API:

- pairwise_procrustes_matrix(layouts, free_aspect=False) -> (m x m) float64 ndarray.
  layouts: sequence of [N,2] float64 arrays. Default path: complex Gram trick + hybrid
  exact-SVD fallback for entries < 1e-4, per spec sec. 3. Must match
  scripts/fast_fidelity_report.py:26 procrustes_rmsd EXACTLY (that function is the
  project-wide convention -- read it). free_aspect=True: symmetrized anisotropic distance
  d_sym(a,b) = 0.5*(d_aniso(a->b) + d_aniso(b->a)) where d_aniso follows
  dagua/eval/equivalence_metrics.py's anisotropic_procrustes convention (read it); exact
  computation, no Gram trick.
- analyze_mode_a(D_layouts, R_layouts, rng, free_aspect=False) -> dict with ALL spec sec. 4
  statistics and flags: E (diag-excluded U-statistic), e_rel, disp, m-out-of-n CI
  (sec. 4.1 exactly), p_diff (paired-swap, 10,000), split calibration
  (K=1,000, E_self/E_cross, dist_equivalent bool, equiv_percentile, fixed-m=15 sensitivity),
  p_track (100,000 vectorized), track_ratio, recovery_at_1, twin_rank median + deciles,
  near_deterministic / one_sided_degenerate routing flags, plain-Procrustes W means
  (guards ALWAYS read plain-Procrustes W even when free_aspect distances are used --
  spec secs. 3/5).
- analyze_mode_b(D_layouts, R_layout, rng, free_aspect=False) -> dict per spec sec. 5:
  near-det route first (trigger on plain W_D; d_R verdict value uses the registered
  distance), symmetric conformal typicality (score formulas EXACTLY as sec. 5, p_typ),
  informativeness guard mean(plain W_D)/sqrt(2) > 0.85, diagnostics.
- stress utilities (spec sec. 6): prepare_graph_distances(edges, n_nodes) -> BFS distance
  matrix; sample_pairs(dist_matrix, graph_name) -> pair index array (all finite pairs if
  <= 100,000 else 100,000 sampled with rng sha256("r70::stressP::<graph>"); cross-component
  (non-finite) pairs excluded); stress_per_layout(layout, pairs, dists) with closed-form
  optimal alpha.
- paired_tost(diffs, margin) and one_sample_tost(values, target, margin) -> dict with
  t-based p_tost, wilcoxon p_tost annotation, degenerate-sd direct-margin branch
  (sd < 1e-12), margin floor max(margin, 1e-6) applied by callers per spec.
- bh_fdr(pvals) -> q-values (standard Benjamini-Hochberg).
- assign_rung(record) -> (rung, annotations) implementing the spec sec. 7 ladder for both
  modes, given a record dict that already contains FDR-adjusted q-values where the ladder
  needs them (FDR is applied by the report stage, not inside per-combo analysis).
- run_oc_simulation(rng) -> dict implementing the operating-characteristics annex
  (spec sec. 4.3: synthetic 2D Gaussian-mixture clouds; shift/spread in {0,.25,.5,1.0};
  n in {30,60,100}; 500 reps/cell; q95-rule pass rates).

SEEDING: every stochastic step uses numpy default_rng seeded via
int.from_bytes(hashlib.sha256(purpose_string.encode()).digest()[:8], "little").
NEVER Python hash(). Callers pass rng; module functions must not create unseeded RNGs.

PERFORMANCE CONTRACT (spec sec. 10): per-pair / per-permutation pure-Python loops are
FORBIDDEN in hot paths. The Gram-trick matrix is two complex matmuls; permutations are
vectorized gathers/masks on the precomputed matrix; splits/bootstrap are index ops.
Include a test asserting pairwise_procrustes_matrix on 200 random layouts of N=2000
completes in < 10 seconds.

TESTS (tests/test_distributional_fidelity.py) -- the spec sec. 10 Task A list, verbatim:
- agreement vs procrustes_rmsd (import it from scripts/fast_fidelity_report.py or
  replicate it locally in the test as the oracle): atol 1e-10 for d > 1e-4 over >= 1000
  random pairs; atol 1e-12 after exact fallback on constructed near-identical pairs at
  d in {0, 1e-9, 1e-7, 1e-5}; degenerate cases (coincident points, collinear, N=2,
  mirrored pairs); float32-quantized inputs.
- synthetic same-distribution clouds -> dist_equivalent True; shifted/scaled clouds ->
  False; synthetic seed-tracking (R_s = D_s + small noise) -> tracking detected
  (low track_ratio, tiny p_track); permutation-p approximately uniform under a true null;
  U-statistic energy distance vs a hand-computed toy case (4 points);
  conformal p exactness on a toy case with the SYMMETRIC score (hand-computed);
  near-uniform cloud -> TYPICALITY_UNINFORMATIVE flagged; point-mass Mode B cloud ->
  near_deterministic route; Mode B with reference == one of the D draws -> REF_TYPICAL;
  ladder unit tests: TRACKING_BUT_SHIFTED annotation, one_sided_degenerate fall-through,
  Mode A near-det rung-1 and quality-fall-through branches;
  anisotropic d_sym symmetry (d_sym(a,b) == d_sym(b,a)) and agreement with the toolkit's
  directed residuals.
</task>

<completeness_contract>
Done means: both files exist; `python -m pytest tests/test_distributional_fidelity.py -x -q`
passes; `python -c "import dagua.eval.distributional_fidelity"` clean; ruff check on both
files clean (or only pre-existing-style issues); no other file in the repo modified.
Do NOT git commit -- leave the working tree for CC review (project convention).
</completeness_contract>

<verification_loop>
After writing: run the test file; fix failures; re-run until green or a genuine wall.
If a spec formula seems ambiguous, re-read the spec section -- 5 review rounds removed
the known ambiguities; prefer the literal reading. Record any residual interpretation
choice as a code comment prefixed "SPEC-INTERPRETATION:".
</verification_loop>

<action_safety>
Read-only with respect to everything except the two named files. Never invoke layout
engines or reference implementations (analysis reads stored positions only -- and Task A
touches no benchmark data at all, only synthetic test data). Do not install packages
(numpy/scipy/torch/pytest are present).
</action_safety>

<default_follow_through_policy>
Take the most reasonable low-risk interpretation and proceed; only stop for genuine
correctness walls. Do not expand scope to Tasks B/C.
</default_follow_through_policy>
