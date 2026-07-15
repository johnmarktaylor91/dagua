# BUCKET: classical_mds (30 combos) + small tails (umap 7, gem 5, maxent 3, drl 3, neato 2)

Target lists: r75_targets_classical_mds.json and r75_targets_small_tails.json
(BUCKET=mds_tails for your report filename).

## Part 1: classical_mds (30 divergent combos, mostly disconnected graphs)
Reference: igraph layout_with_mds (seedable base igraph_mds; adapter
dagua/eval/competitors/igraph_competitor.py). Dagua: dagua/layout/ops/pipelines/classical_mds.py.
Known from r74 (codex Option-2 verdict): igraph SPLITS disconnected graphs, runs MDS per
component, then merges components via igraph_layout_merge_dla -- stochastic Diffusion-Limited
Aggregation (NOT grid packing; a naive TileToRows port was reverted as pure harm). A naive DLA
port attempt HUNG (unknown cause -- possibly the random-walk loop).
Your job:
1. PORT SPEC for igraph's merge_dla: read _references/igraph/src/layout/merge_dla.c (+ the
   grid/quadrant helpers it uses, and its RNG use igraph_rng_*). Document the exact algorithm
   (component circle sizes, placement order, random-walk step distribution, termination), what
   made a naive port hang (bounded? expected steps?), the RNG stream requirements for per-seed
   bit-exactness, and a bounded-time implementation strategy in torch/numpy. Estimate LOC/effort.
2. TRIAGE: which of the 30 combos are disconnected (target JSON `disconnected` field) and thus
   DLA-explained vs connected combos needing a DIFFERENT root cause? For connected divergent
   combos (some are hairline <=1%), diagnose separately (eigendecomposition details, landmark
   selection, double-centering, sign conventions?).
3. Are the igraph_fidelity variants vs default variants failing for the same reason?

## Part 2: small tails -- dispose of every combo with evidence
- umap (7): median stress excess -68% => dagua MASSIVELY better = almost certainly a comparison
  bug. Check the umap adapter params (n_epochs, min_dist, spread, init) vs the variant's params
  (dagua/eval/competitors/, variants.py). Are the reference runs using matched params per the
  parameter-matching rule? Is negative-sampling RNG the residual?
- gem (5): igraph gem. r71 "fixed" gem; what remains? All 5 are hairline (<=1% stress). Is this
  a genuine FP floor (then say so WITH evidence: 1-ULP perturbation experiment showing chaotic
  seed-level divergence) or fixable?
- maxent (3): after the r74 revert, what exactly still fails? (OGDF StressMinimization ref.)
- drl (3): igraph drl; r71 fixed the main issues; classify the residue.
- neato (2): all-worse, small gaps. graphviz neato ref. Check stress majorization details
  (init, termination) OR whether these are near-margin cases a calibrated crossings/stress
  margin would legitimately clear (do NOT propose margin changes yourself -- just flag).
For each tail combo: root cause (CONFIRMED/HYPOTHESIS + experiment), fix sketch or documented
floor-with-evidence, expected end-state (bit-exact / 3Q / aggregate-equivalence / floor).
