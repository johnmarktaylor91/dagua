# r76 -- THE FINAL FIDELITY SPRINT (autonomous, JMT-authorized 2026-07-02)

**Directive (JMT, verbatim intent):** last sprint; close ALL remaining gaps "we can reasonably
close without doing violence"; structurally analogous to r75; LEAN CODEX OVER SONNET (preserve
Claude credits): codex for all research+impl; codex-vs-codex redundancy; claude agents only for
in-session-capability needs; Fable = architect/synthesis only. Memory:
project_r76_final_sprint_directive. Baseline: r75_RESULTS.md (163 divergent + 79
superior-distinct + big-graph tier unscored).

## Stop criteria (quantitative -- the sprint ends when EVERY combo has exactly one disposition)
identical (bit/dist/3Q, gate_5 0/40) | evidenced floor (perturbation/eigengap proof) |
no-canonical-reference | superior-distinct (fairness-audited) | aggregate-equivalent (population
tier). Zero rows left as bare "divergent" without a named cause + documented why-not-closed.
No regressions: bit-exact/identical counts never decrease. gate_5 0/40 on every rescore.

## Work items (from r75; ROI-ordered within tracks)
TRACK A -- sugiyama (121 divergent: the bulk):
  A1 mincross GD_nlist install order + representative-chain ED_xpenalty merge port (attempt-2
     analysis in r75_IMPL_mincross_NOTES.md; patches in r75_findings/r75_cx_mincross*.log;
     discriminator harness: dot -v ordering counts, already proven on 3/4 graphs).
  A2 stage-A x-simplex PERF (network simplex slow >=~100-node dense; unblocks 9 unmeasured
     graphviz_fidelity combos + ba_500 tail; iterative DFS already landed in transcript patches).
  A3 igraph family: GLPK-vs-HiGHS tie parity (LP objective now all-zero on feedback-free DAGs ->
     solver tie-break decides ranks), BK ordinal-edge conflict quirk, qsort ties (verdicts 19-21).
  A4 stages B-D (flat edges, edge labels, clusters) per position.c -- AFTER A1/A2 land.
TRACK B -- fmmm residual (33) + gem (7):
  B1 fmmm triage vs honest refs: which legs fail now? (references regenerated w/ fixed binary;
     probe showed RMSD ~1e-3 at steps10 -- suspect crossings-leg integer discreteness + steps100/200
     parity). Possibly bit-exact reachable via full RNG-stream match (r71 method).
  B2 gem re-port vs honest refs (r71 port matched STALE binary; dagua-vs-new 0.21 RMSD at matched
     rounds -- read foxglove-202510 gem source, first-divergence trace).
TRACK C -- mds (22) + tails (15):
  C1 mds disconnected residuals (8 flipped; which legs still fail on the rest? DLA placement-scale
     delta noted r=167 vs 3.3 -- check if crossings leg penalizes packing geometry).
  C2 mds connected = evidenced floor (eigenvalue ties, E2) -> formal floor disposition w/ data.
  C3 umap numba neg-sampling trace (7); drl/neato/maxent perturbation dispositions (8).
TRACK D -- infrastructure:
  D1 big-graph tier: hang-safe scoring (landmark APSP + sampled-crossings SE now plumbed +
     APPROX_UNRESOLVED semantics per metrics reports) -> rescore >300-node combos (~238 upper
     bound from r73 snapshot; ba_500/2000/5000 et al).
  D2 population-equivalence aggregate tier (S3 design, metadata-only, BH-corrected).
  D3 final ledger generation + official report + gates + r76_RESULTS + baton + memory.

## Wake-up routing
Case A: codex done + not verified -> verify (NOTES + gates + git log) -> merge or park (2-attempt
max per item) -> next item. Case B: bench/analysis done -> read counts -> next. Case C: codex
quota out -> agent-control status; fallback per global rules; note here. Case D: idle + items
open -> dispatch next per tracks. ALWAYS: check `ps -eo pid,args | grep -E "codex exec|
run_benchmark|definitive_fidelity"` before dispatching; never two writers on one results.json;
batch-scale --timeout (>= seeds x per-seed x 2); OMP_NUM_THREADS=1 on benches; verify git log
BEFORE any worktree remove.

## GUARDRAILS (unchanged from r75 + new lessons)
NEVER launder (gate_5 0/40). Benchmark path only. Params+seed matched. No runtime delegation.
Floor claims need FP-chaos/eigengap evidence. final_rung is a STRING. Adversarial-critique big
research before code (codex high-effort critic). Version-pin all source citations (graphviz `git
show 7.0.5:`; igraph = INSTALLED 1.0.0 runtime traces; OGDF = ~/tools/ogdf-src foxglove-202510).
Per-combo freshest-dir overlay is LAW (7fa972e). Resume counts errors as complete -- fresh dir
for retries. 2-attempt max per work item, then park documented.

## JMT BOUNDARY RULINGS (2026-07-02, final)
- OMIT: patched-graphviz reference build (theta/steps stay no-canonical) -- "recompiling the
  whole package" excluded.
- OMIT: LAPACK-degenerate eigenbasis vendoring (connected mds = floor; DO upgrade the label to
  "proven member of reference equivalence class" -- claim-strengthening only, no port).
- PUSH TO EXHAUSTION: (a) all "FP-chaos floor" candidates via first-divergence bisection --
  NO chaos-floor label until bisection stops finding op differences (sfdp connected residue,
  drl, any gem/fmmm leftovers); (b) umap scalar-faithful SGD port (tau_rand xorshift stream is
  portable; target distributional match, bit-exact if float32 order cooperates).
- OVERNIGHT MODE: JMT asleep; run autonomously to DONE. "Fix everything you can fix; leave
  nothing on the table except the truly absurd."

## Iteration log
- 2026-07-02: STATE created at r75 close (r75 official: divergent 574->352, 3Q 36->79,
  no-canonical 290, ZERO bit-exact regressions -- verified). WAVE 1 dispatched (all codex):
  A2 xns-perf (worktree dagua-xns-perf, branch r76/xns-perf), B1 fmmm triage probe,
  C1 mds+gem triage probe. PIPELINE: A2 done -> A1 mincross (recover attempt-2 patches from
  r75_findings/r75_cx_mincross*.log) -> A4 stages B-D; B1/C1 verdicts -> targeted fix codexes
  (bisection-first per JMT ruling); then umap scalar port; then superior-distinct fairness
  triage (79 rows: match-the-worse-reference where portable); then D1 big-graph hang-safe
  rescore + D2 population tier + D3 final ledger/report/baton.
