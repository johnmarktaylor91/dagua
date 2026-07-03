<task>
FMMM first-divergence bisection probe for dagua r75. Context: dagua's OGDF-FMMM fidelity port
diverges from the reference on SOME seeds only (grid_5x5 steps10: seeds 44/45 match the reference
to RMSD ~0.001-0.002, seeds 42/43/46 diverge to 0.04-0.14 -- see
.project-context/research/sprint_rng_matching/r75_findings/r75_PROBE_fmmm_RESULTS.md). A
seed-conditional divergence means the pipeline is nearly right and something occasional (an extra
RNG draw, a boundary event, a tie-break) desyncs it. Your job: find the FIRST divergent iteration
and the event that causes it. RESEARCH/PROBE ONLY for the dagua repo (no commits; you may write
scratch scripts to /tmp and the results file below).

Repo: /home/jtaylor/projects/dagua (develop, read-only except the results file).
OGDF source: /home/jtaylor/tools/ogdf-src (git checkout, tag foxglove-202510) with an existing
build used by scripts/rng_match/build_ogdf_runner.sh -> scripts/ogdf_runner. READ that build
script first to understand the build layout.

STEP 1 -- instrumented OGDF runner (SEPARATE binary, pristine restore contract):
- Before touching anything: save a baseline output: run the CURRENT scripts/ogdf_runner on
  grid_5x5 (see how dagua/eval/competitors/ogdf_competitor.py invokes it -- replicate one
  fixed_iterations=10 seed=42 invocation) and keep the JSON in /tmp/ogdf_baseline_grid5x5.json.
- Patch /home/jtaylor/tools/ogdf-src FMMM sources minimally (FMMMLayout.cpp) to, when an env var
  OGDF_FMMM_DUMP is set: (a) dump all node positions after each main force-loop iteration to a
  JSONL file (path from env), (b) log the count of randomNumber()/rand-family calls consumed per
  iteration if feasible (a static counter in the RNG wrapper or around the known call sites).
- Build to a SEPARATE binary /tmp/ogdf_runner_dump (do NOT overwrite scripts/ogdf_runner).
- RESTORE CONTRACT (hard requirement): after building the dump binary, `git -C
  /home/jtaylor/tools/ogdf-src checkout -- .` so the source tree is clean (`git status` clean),
  then rebuild the ORIGINAL runner via build_ogdf_runner.sh and verify it still reproduces
  /tmp/ogdf_baseline_grid5x5.json byte-identically. If the rebuild cannot reproduce baseline,
  STOP and report -- do not leave the tree in a mixed state.

STEP 2 -- dagua-side dumps: wrap _ogdf_fmmm_force_iteration in-memory (as the earlier probe did)
to dump positions per iteration for the same graph/seeds through the BENCHMARK PATH
(get_competitor('classic_fmmm'), variant_params={'steps':10,'fidelity_mode':True}).

STEP 3 -- bisect: for grid_5x5 seeds 42 (diverging) and 44 (matching), align the per-iteration
position series (they should start identical if initial placement matches -- confirm iteration-0
parity first). Report: first iteration k where max-abs position delta exceeds 1e-9; the delta
pattern at k (one node? all nodes? a contiguous cluster?); the OGDF RNG-consumption count at k vs
dagua's (if dagua consumes RNG that iteration, count its draws too); and any boundary events
(positions at the comp-box edge, coincident integer positions) at k-1/k. If iteration-0 already
differs on seed 42, the divergence is in initial placement -- bisect THAT (which RNG draw differs
first). Repeat for deep_chain_20 seed 42 if time allows (cheap once tooling exists).

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r75_PROBE_fmmm_bisect_RESULTS.md
-- commands, iteration-0 parity verdict, first-divergent iteration per seed, the divergence
signature, RNG-count comparison, your root-cause hypothesis ranked CONFIRMED/PLAUSIBLE, and the
minimal gated fix sketch. Plus the restore-contract verification evidence (git status clean +
baseline byte-match). ASCII only. Budget ~60-75 min including the OGDF partial rebuild.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation. If patching OGDF proves too slow to build, fall back to
LD_PRELOAD interposition on rand/random or a gdb-scripted dump -- document whichever path worked.
The restore contract is non-negotiable.
</default_follow_through_policy>
