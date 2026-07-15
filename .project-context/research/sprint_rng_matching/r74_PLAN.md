# r74 -- Close All Fidelity Gaps: SYNTHESIZED PLAN (post dual-wave research, pre-adversarial-gate)

Baseline (r73): divergent 574, INSUFFICIENT 257, 3Q 36. Sources of truth: full findings in
`/tmp/r74_O{1..6}_findings.md` (Opus) + `/tmp/r74_CX{1..6}_findings.md` (Codex, redundant/blind).
Tier priority per JMT: bit-identical > statistically-identical-by-layout > quality-identical. Pure
last-decimal-FP residual = accepted floor (prove, don't chase). NEVER LAUNDER. NO RUNTIME DELEGATION.
Verify on BENCHMARK PATH, matched seed+params, re-bench only CHANGED algos, seed-count matched.

Agreement key: [BOTH]=Opus+Codex concur; [1-LAB]=one only; [DISPUTED]=labs disagree (note resolution).

## WAVE A -- cheap, high-confidence, both-labs-confirmed (implement first)
- **A1 sfdp p_neg2 force-law** [BOTH]. graphviz clamps repulsiveforce=-2 -> 0 (late_double min 0.0)
  then forces p=-1, i.e. runs pow(dist,2); dagua runs pow(dist,3) at `sfdp.py:539`. GATE: first verify
  p_neg2 reference rows are byte-identical to sfdp_default reference rows (if so, the variant is just
  default). Expected ~52 combos (sfdp_p_neg2 = 73 divergent vs ~21 for siblings). ONE-LINER. rung 1-2.
- **A2 umap_nn30 clamp** [BOTH]. n_neighbors(30) >= N on small graphs -> BrokenProcessPool both reimpl
  AND reference. Clamp `n_neighbors=min(30,N-1)` both sides. Recovers 10 INSUFFICIENT. Quick.
- **A3 sugiyama recursion-depth crash** [BOTH]. `small_world_2000::sugiyama_graphviz_fidelity` blows
  Python recursion in cycle-break (`ops/sugiyama.py:2150` region). Make cycle/feedback-arc iterative.
  1 crash combo (+ headroom for other big graphs).
- **A4 maxent_stress disconnected** [BOTH]. 3 divergent = disconnected random_dag_50. OGDF routes
  disconnected through ComponentSplitter + TileToRows (packer already ported in gem.py); dagua does one
  global majorization (`stress_majorization.py:589-612`). Deterministic -> rung 1.

## WAVE B -- medium ports, confirmed mechanism (implement after A)
- **B1 sfdp disconnected-component packing** [BOTH, with caveat]. dagua runs one shared force field;
  graphviz lays out each component independently (own srand) + packs. Reuse neato polyomino packer +
  per-component RNG reset. ~48 disconnected sfdp candidates. CAVEAT (CX6): stress direction is MIXED
  across disconnected combos (some have dagua already BETTER, D<R) -- packing graphviz-style may not
  flip those and could regress. Per-combo verify; do not assume all 48 flip.
- **B2 classical_mds disconnected-component packing** [BOTH]. igraph decomposes -> per-component MDS ->
  merge_dla; dagua fills cross-component inf with one global scalar -> single blob
  (`graph_utils.py:319-352`). Decompose + per-comp MDS + deterministic TileToRows pack -> rung 3
  (rung 1/2 needs a DLA port + seeded ref, out of scope). ~14-20 combos.
- **B3 fmmm fdp-perf vectorize** [BOTH]. dagua already ports graphviz's spatial grid but in Python
  loops (`fmmm.py:5189-5251`); torch-vectorize the exact grid-cell repulsion (NOT Barnes-Hut -- less
  faithful). Recovers ~9 fdp INSUFFICIENT timeouts (only ~2-4 likely score >floor).

## WAVE C -- bigger / uncertain ROI (GATED on adversarial verdict; attempt highest-leverage only)
- **C1 sugiyama igraph LP objective** [BOTH]. dagua uses a ZERO objective (`ops/sugiyama.py:379`);
  igraph minimizes sum(outdeg_i - indeg_i)*x_i AND gates the LP to directed graphs <=1000 nodes (else
  Eades), while dagua runs HiGHS unconditionally. Biggest single lever (25-60). Bounded code change.
  REGRESSION RISK: must not break currently-matching sugiyama combos -- verify no rung1->4 regressions.
- **C2 fmmm component min-area rotation** [DISPUTED count]. OGDF searches 10 angles/comp, keeps
  min-plain-area; dagua bakes a single-component aspect-ratio rotation + skips the search
  (`fmmm.py:1755/1331/1576`). Opus said 12-15; Codex (CX3) says rotation only touches the ~13
  DISCONNECTED (62/75 divergent are CONNECTED -> different unproven force residual). Cap expectation at
  the disconnected subset.
- **C3 sfdp gv_random RNG match** [1-LAB, Codex/CX2 -- NEW lead]. graphviz `gv_permutation` uses
  rejection-sampled `gv_random` (`random.c:15-32`); dagua uses raw modulo (`ops/sfdp.py:247-253`).
  Could systematically shift the connected-sfdp "floor" residual. Low-med effort; test before believing.
- **DEFER (document as future, do NOT attempt this sprint -- multi-day, JMT's FP-floor caveat applies
  to the residual):** sugiyama graphviz position.c network-simplex x-coord port (omega 1/2/4 per CX1,
  2-4 days); fmmm connected force-formula residual; sugiyama cluster structural metadata.

## DATA / RELABEL
- **D1 sgd2_multi_with_crossing (81 INSUFFICIENT "no_reference_rows")** [DISPUTED/PROVENANCE].
  O5+CX5 disagree on whether a real reference exists (siblings are 100% bit-exact; a ref row shows
  status running/incomplete) vs structural-by-design. MUST verify provenance BEFORE spending re-bench
  compute. If real+completable -> re-bench to completion (likely rung 1, biggest count win). If not ->
  relabel STRUCTURAL_NA. Do not blindly rerun.
- **D2 relabel ~52 COMPUTE_FRONTIER_NA** [BOTH]. Large-graph legit timeouts (FR/SFDP/DRL/sugiyama/MDS
  on N>=2000). Not recoverable; relabel so they stop counting as unexplained INSUFFICIENT.
- **D3 small-graph perf vectorize (davidson_harel, drl, stress_maj, neato)** [BOTH]. Pure-Python scalar
  inner loops time out on N<=300 (davidson ~9s on 42 nodes). Vectorize -> recover ~56-82 INSUFFICIENT.
  Bounded but spans several engines; sequence carefully.

## PROVE-FLOOR (no fix; run 1-ULP/perturbation experiment on benchmark path, DOCUMENT, then accept)
gem 22 (FP summation chaos; 269/309 already bit-exact) [BOTH]; classical_mds degenerate-eigenspace
14-16 (LAPACK dsyevr basis) [BOTH]; fmmm connected ~58 (libm chaos) [BOTH]; sfdp connected ~63
(adaptive-cooling REFUTED by CX2; test gv_random C3 first, then accept) ; drl/neato/umap tails [BOTH].

## 3Q -- NO PROMOTIONS [BOTH]. 0/574 pass strict gate even pre-BH; controls 0/40. Line held. Do not relax.

## ADVERSARIAL GATE -- questions the reviewers MUST answer before any code lands
1. DOUBLE-COUNTING: how much do A1 (p_neg2 ~52) and B1 (sfdp disconnected ~48) overlap? Net unique
   sfdp combos? (sum is misleading.)
2. REGRESSION RISK: for B1/B2 (disconnected packing) and C1 (sugiyama objective) -- can the fix push a
   currently-matching (rung1-3) or currently-dagua-better combo to WORSE? Enumerate the guard.
3. SGD2 PROVENANCE (D1): does a real, completable reference actually exist? Resolve before compute.
4. p_neg2 PREMISE (A1): confirm in source that graphviz truly clamps (not a dagua-side misread); predict
   actual flip count after removing disconnected/floor double-counts.
5. ORDER: correct implementation/verification order to avoid overlay-trap Frankensteining; which fixes
   share files and must serialize.

## EXECUTION CONTRACT
Implement in waves A -> B -> C(gated) via Codex, sequential or isolated worktrees (r73 wave-1 collided
in a shared tree). After each algo's fixes: re-bench ONLY that algo on the benchmark path, seed-count
matched (full 100 if base has 100), re-run seeded refs with --seed-refs for SEEDABLE_BASES, re-analyze
(overlay freshest-last), check no regressions, scorecard vs 574. Then report + supersede + commit +
file-for-review + TEXT JMT the final verdict table.
