<task>
Run the two decisive FMMM experiments ordered by dagua's r75 adversarial critique (verdicts 9-10
in .project-context/research/sprint_rng_matching/r75_findings/r75_ADVERSARIAL_VERDICTS.md; context
in r75_fmmm_codex.md and r75_fmmm_sonnet.md). RESEARCH/PROBE ONLY: no repo modifications, no
commits; scratch scripts in /tmp; you may monkeypatch dagua code IN MEMORY within probe scripts.

Repo: /home/jtaylor/projects/dagua (develop @ 89ed3c3, read-only). Reference OGDF source (the
version the benchmark runner is actually built from): /home/jtaylor/tools/ogdf-src
(tag foxglove-202510). Runner: scripts/ogdf_runner (built by scripts/rng_match/build_ogdf_runner.sh).

EXPERIMENT 1 -- coincident-node repulsion trigger census (verdict 9):
dagua zeroes repulsive force for exactly-coincident node pairs
(dagua/layout/ops/pipelines/fmmm.py:562-585); OGDF jitters coincident points, CONSUMING its
global RNG (ogdf-src .../fmmm/numexcept.cpp:48-70, :169-181). Question: do coincident pairs
actually OCCUR during failing runs?
- Instrument _ogdf_fmmm_tensor_repulsive_forces (in-memory monkeypatch: wrap it, count
  off-diagonal zero-distance pairs per iteration) and run the BENCHMARK PATH
  (dagua.eval.competitors get_competitor('classic_fmmm').layout_with_variant, fidelity_mode=True)
  for deep_chain_20, grid_5x5, weighted_chain_20, asymmetric_hourglass_hub at steps10, seeds
  42-46. ALSO check the integer-flooring claim: does the dagua fidelity loop floor positions to
  integers each iteration like OGDF (verify OGDF does: FMMMLayout.cpp restrict_force_to_comp_box /
  move semantics in ogdf-src) -- if dagua keeps float positions, coincidence probability differs
  fundamentally; report the actual position-quantization behavior on both sides.
- Deliverable: per graph/seed, trigger counts; verdict CONFIRMED-CAUSE (triggers observed in
  failing rows) or KILLED (no triggers).
EXPERIMENT 2 -- oscillation-damping angle formula (verdict 10):
dagua uses atan2(cross,dot) (pipelines/fmmm.py:168-196, tensor :748-752); OGDF uses
atan2(dy2,dx2)-atan2(dy1,dx1) (ogdf-src include/ogdf/basic/geometry.h:134-149), feeding
ceil(angle/0.52359878) sector buckets (FMMMLayout.cpp:1285-1299).
- In-memory swap dagua's formula for OGDF's (including its range behavior -- the subtraction can
  leave [-pi,pi]; check what OGDF does with negative/out-of-range angles before ceil) and rerun
  the same 4 graphs x 5 seeds through the benchmark path. Compare final positions + normalized
  stress vs the unswapped run and vs the ogdf_runner reference (scripts/ogdf_runner via
  get_competitor('ogdf_fmmm'), matched fixed_iterations).
- Deliverable: per graph/seed position RMSD swap-vs-unswapped, and whether the swap MOVES dagua
  toward the reference. Verdict: CONFIRMED-CAUSE / CONTRIBUTING / KILLED.
Also (cheap, while instrumented): report per-iteration first-divergence -- for one graph/seed
(deep_chain_20, seed 42) dump dagua positions at iterations 1,2,3,5,10 and ogdf_runner equivalents
if the runner supports iteration dumps (check scripts/ogdf_runner.cpp usage/flags; if it cannot
dump intermediates, note that and skip).

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r75_PROBE_fmmm_RESULTS.md --
per experiment: exact commands, raw numbers, verdict, and the recommended minimal gated fix (or
explicit kill). ASCII only. Runtime budget ~40 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation. If the benchmark-path instrumentation proves awkward,
document the obstacle and use the closest faithful harness, clearly labeled.
</default_follow_through_policy>
