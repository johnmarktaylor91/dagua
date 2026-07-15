<task>
Performance fixup for dagua's new classical_mds DLA merge path (landed today, commit ec24b05).
The 100-seed re-bench shows it TIMES OUT (>300s per layout) on many-component graphs:
random_dag_50 (52 weak components: 50 singletons + 45-node + 2-node) and random_dag_200
(~202 components). All other disconnected graphs (2-7 components) complete fine. A prior
research probe (r75_mds_tails_sonnet.md) showed a literal scalar Python port of igraph's DLA
completes random_dag_200 in ~17s -- so <300s is clearly achievable; target <10s/seed for
random_dag_200.

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-dla-perf (branch r75/dla-perf). Work ONLY
here. Conventional commit (perf(classical_mds): ...). No push, no merge. Pre-commit runs
ruff-format: if it reformats, re-add and re-commit until the commit lands.

CONSTRAINTS (hard):
- OUTPUT MUST BE UNCHANGED for graphs that already complete: the DLA path consumes
  random.Random(seed) draws in C-call order -- your optimization must NOT change the number,
  order, or values of RNG draws, nor any placement decision. Verify: for 3 seeds each on
  multi_component_80, parallel_cycles_4x5, disconnected_encoder_residual, positions pre/post
  optimization must be bit-identical (torch.equal). Write the comparison into a test or show
  the script + output in your notes.
- Do not relax the guardrail-raise semantics (caps may stay; they must still raise, not fallback).
- Connected-path code untouched.

WHERE TO LOOK (profile first, then fix -- do not guess):
- dagua/layout/ops/pipelines/classical_mds.py -- today's DLA implementation (_rng_unif walk loop,
  place_sphere / get_sphere collision check, grid rasterization).
- Likely hot spots: per-step collision checks scaling with number of placed spheres/occupied
  cells (202 components -> ~200 placed spheres by the end); Python-level per-step overhead in the
  random walk (startr grows with total area, walk step = startr/100); repeated allocation inside
  the loop. Profile ONE random_dag_200 seed (cProfile or time-per-phase prints) and report where
  the time actually goes BEFORE optimizing.
- Allowed techniques: precomputed numpy occupancy grids, incremental data structures, vectorizing
  the collision predicate for a single candidate against all occupied cells, caching -- anything
  that preserves exact draw-order semantics. NOT allowed: changing walk step distribution,
  early-exiting walks differently, torch JIT of the whole walk if it changes float semantics
  (python floats are C doubles -- keep scalar float math for RNG-derived values).

VERIFICATION LOOP:
a. Profile evidence: before/after per-seed wall time for random_dag_200 seed 100 and
   random_dag_50 seed 100 (target <10s and <5s respectively).
b. Bit-identity check on the 3 already-passing graphs (above).
c. pytest tests/test_pipeline_classical_mds.py -x -q green.
d. Benchmark-path probe (PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl): scripts/run_benchmark.py
   --workers 2 --timeout 300 --seeds 5 --seed-start 100 --variants
   --graphs random_dag_50,random_dag_200
   --engines classic_classical_mds_default,classic_classical_mds_igraph_fidelity
   --output-dir /tmp/r75_dla_perf_probe  -> expect 20/20 ok, 0 timeouts.
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_dla_perf_NOTES.md:
profile findings, what changed, before/after times, bit-identity evidence, commit sha.
</task>
<completeness_contract>
Done = profile-driven fix committed, random_dag_50/200 complete well under timeout, bit-identity
proven on already-passing graphs, tests green, notes written.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/dla-perf only. Touch only the DLA path + its tests.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
