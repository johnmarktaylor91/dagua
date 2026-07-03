# SHARED CONTEXT -- dagua r75 fidelity endgame, research sweep

You are a research agent in dagua's r75 fidelity sprint. dagua (/home/jtaylor/projects/dagua,
branch develop @ 89ed3c3) is a PyTorch graph-layout engine that REIMPLEMENTS classic layout
algorithms (graphviz sfdp/dot, OGDF FMMM/stress, igraph mds/gem/drl, umap-learn, etc.) as
composable op pipelines under dagua/layout/ops/ and dagua/layout/ops/pipelines/. A 74-round
verification effort compares each of 118 classic_* variants against its canonical reference on
~95 graphs x 100 seeds, categorizing every (engine, graph) combo as bit-exact /
distributionally-equivalent / quality-identical / divergent.

North star (user directive): EVERY combo either bit-identical or statistically identical at the
distributional level; only genuine floating-point-rounding residuals are acceptable. This sprint
targets the remaining divergent combos.

A RIVAL research agent from the other AI lab is independently investigating the SAME bucket in
parallel; your report will be diffed against theirs and disagreements become signal. Be precise,
cite evidence, do not hand-wave.

THIS IS A RESEARCH-ONLY TASK:
- Do NOT modify or commit anything in the dagua repo. Scratch scripts go in /tmp. Read-only
  experiments are encouraged (python3 has torch 2.8 + igraph + the full dagua stack; run from
  /home/jtaylor/projects/dagua so `import dagua` resolves).

KEY DATA:
- Current divergent set: eval_output/fidelity_definitive/r74_phase2_rescore.jsonl -- one JSON row
  per (engine, graph) combo (409 rows; rows with quality_identical_raw=true were already
  reclassified and are NOT targets). Leg fields for battery_stress / cross / np:
  *_D_mean (dagua), *_R_mean (reference), *_margin, *_ref_self_spread, *_direct_equivalent.
- YOUR TARGET LIST (the divergent combos in your bucket, with per-leg numbers):
  .project-context/research/sprint_rng_matching/r75_findings/r75_targets_<BUCKET>.json
- Layout positions (per seed, torch .pt): eval_output/benchmark_100seed_r74_fixes/positions/
  named <graph>__<engine>__seed<NNN>.pt, indexed by results.json. Older benchmark_100seed_* dirs
  hold earlier runs (an overlay chain; freshest dir wins for engines it covers). Reference-engine
  positions use the reference's own engine name (e.g. graphviz_sfdp, ogdf_fmmm, igraph_mds).
- Analysis code: scripts/definitive_fidelity_analysis.py, dagua/eval/equivalence_metrics.py,
  dagua/eval/distributional_fidelity.py.
- Reference ADAPTERS (how references are invoked; CHECK PARAMS AND POSITION EXTRACTION HERE):
  dagua/eval/competitors/ (graphviz_competitor.py, ogdf_competitor.py, igraph_competitor.py, ...).
- Reference C/C++ source trees: /home/jtaylor/projects/_references/{graphviz,igraph,ogdf}.

CONTEXT ON METRICS (r74 corrected them; margins are now variance-tied):
- A divergent combo failed BOTH position-level distributional matching (Procrustes-family, with
  automorphism / per-component / anisotropic-for-sugiyama invariances) AND the 3-quality battery
  (scale-fitted normalized stress, edge crossings, k=10 neighborhood preservation) where
  equivalence margin = max(floor, reference self-spread across its own 100 seeds).
- IMPORTANT ASYMMETRY FINDING (from sprint lead): of 337 remaining divergent combos, 93 are
  dagua-BETTER on every failing leg and 89 mixed. Dagua-better-than-reference usually means a
  COMPARISON BUG (parameter mismatch, reference post-processing like overlap removal included in
  extracted positions, different iteration counts), NOT that our reimplementation is superior.
  Treat "dagua better" as a red flag to root-cause.

GUARDRAILS (violations wasted weeks in prior rounds):
- Any dagua-vs-reference experiment MUST use the benchmark path (dagua.eval.competitors
  get_competitor machinery / scripts/run_benchmark.py), NOT direct pipeline calls -- direct calls
  produce different positions and give FALSE readings.
- Params + seed must MATCH between the dagua variant and the reference run (never compare against
  reference defaults unless the variant explicitly targets defaults).
- Never propose having dagua import/invoke the reference at runtime. Ports must be real ports.
- A "floor / unfixable" claim requires FP-chaos evidence (1-ULP perturbation experiments,
  summation-order sensitivity, chaotic divergence growth), not an assertion.
- Prior-round fix attempts that FAILED (do not re-propose blindly): blanket per-component
  splitting for maxent (OGDF does NOT component-split; broke 25 bit-exact combos, reverted),
  TileToRows packing for classical_mds (igraph uses stochastic DLA merge; reverted).

OUTPUT CONTRACT:
Write your report to
.project-context/research/sprint_rng_matching/r75_findings/r75_<BUCKET>_<YOURLAB>.md
(markdown, ASCII only). Structure:
1. Executive summary (<=10 lines).
2. Findings ranked by expected combo-count impact. Label each CONFIRMED (exact command + output
   evidence included) or HYPOTHESIS (with the cheapest decisive experiment and est. runtime).
   Cite file:line for every code claim, on BOTH the dagua side and the reference source side.
3. Per root cause: fix sketch, expected impact (which combos), and RISK to existing bit-exact
   combos (the r74 failure mode: blanket fixes broke bit-exact combos and were reverted).
4. Explicit list of target combos you could NOT explain.
Keep total runtime of experiments bounded (<~45 min); prefer reading source over brute-force runs.
