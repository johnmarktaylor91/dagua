<task>
r76-B3: root-cause and fix dagua's GEM divergence vs the honest OGDF runner. Probe evidence
(.project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_mds_gem_triage.md -- READ
FIRST, "GEM First-Divergence Probe"): on grid_5x5 seed 100 the divergence is ALREADY LARGE at
round 20 (Procrustes 0.144) and grows only modestly by round 100 (0.163) -- an EARLY mismatch,
not chaos. Dagua's bbox is ~3x the runner's (~450 vs ~120 units) despite matched constants --
suspect init/RNG-distribution/scale. Probe anchors: runner seeds std::rand + OGDF, inits
positions rand()%1000/10.0 (scripts/ogdf_runner.cpp:309-317, :414-423); OGDF GEMLayout.cpp
component loop/permute/impulse (:150-186, :240-288, :291-340 in
/home/jtaylor/projects/_references/ogdf -- BUT verify against the RUNNER'S tree
/home/jtaylor/tools/ogdf-src which is the authoritative version); dagua side
dagua/layout/ops/gem.py (:272-286 seed bridge, :335-363 permutation, :415-453 runner-style init,
:830-978 scalar loop, :1049-1075 packing, :1291-1296 rounds translation).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-gem-trace (branch r76/gem-fix). Work ONLY
here; PYTHONPATH=$PWD. Conventional commit (fix(gem): ...); re-add/re-commit through ruff-format
until `git log` shows it. No push/merge.

METHOD (trace-first):
1. KNOWN SUSPECT #1 (check first, cheap): dagua has a custom minstd_rand +
   uniform_int_distribution clone for GEM (gem.py) -- OGDF uses std::mt19937 via randomNumber
   AND the runner ALSO uses std::rand (glibc) for initial positions. Write a /tmp C++ probe
   (like the fmmm triage did) emitting: (a) glibc rand() first 20 draws for the runner's seed
   path, (b) OGDF randomNumber draws, and diff against dagua's streams at the same call sites.
   A distribution-semantics mismatch (libstdc++ uniform_int_distribution vs dagua's clone for
   GEM's ranges) or a rand()-vs-minstd mismatch in INIT would explain immediate divergence.
2. KNOWN SUSPECT #2: the 3x bbox scale -- trace the first 5 node updates (desiredLength,
   barycenter, repulsion, attraction, impulse, position clamp) dagua-vs-OGDF. If needed build an
   instrumented runner to /tmp/ogdf_runner_gemdump from a separate cmake build dir
   (/tmp/ogdf-build-gem) with env-gated per-round dumps; PRISTINE-RESTORE contract on
   /home/jtaylor/tools/ogdf-src (git status clean at end); NEVER touch scripts/ogdf_runner.
3. Fix dagua's gem fidelity path to match; gate to fidelity_mode='ogdf'.
VERIFICATION:
a. grid_5x5 seed 100 rounds 20 + 100: Procrustes vs runner drops from 0.144/0.163 to <0.01
   (report; <1e-3 excellent). Repeat seeds 101-102.
b. triangular_lattice_36 + tl_resnet_2block + regular_4_40 (the other connected divergent rows),
   3 seeds each at iters100: RMSD before/after table -- material improvement on >=2 of 3.
c. Regression: 3 currently-PASSING gem combos (pick from r75_final where gem quality_identical
   or rung<=3 -- e.g. binary_tree/petersen if present) positions bit-identical pre/post
   (torch.equal) OR document why they legitimately change toward the reference; pytest tests/ -k
   gem -x -q green.
Write .project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_gem_NOTES.md:
first-divergence evidence, fix, before/after, commit sha. 2-attempt budget; honest documented
failure if it resists.
</task>
<completeness_contract>
Done = first divergence named w/ evidence; fix committed; ladder (a) <0.01 + (b) material on 2/3
+ (c) regression gate; notes written; ogdf-src git-clean; scripts/ogdf_runner untouched.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/gem-fix only. Never modify scripts/ogdf_runner, ~/tools/ogdf-src
(restore contract), eval code, or other engines.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical ambiguity.
</default_follow_through_policy>
