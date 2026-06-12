<task>
Implement P1a dispatch (i) of the r71 fidelity-completion plan for the dagua repo
(you are in /home/jtaylor/projects/dagua).

AUTHORITATIVE CONTEXT -- read FIRST:
  .project-context/research/sprint_rng_matching/PLAN_r71_fidelity_completion.md
  (v4, APPROVED after 3 adversarial rounds), sections 2 (P1) and Appendices A-B.
Where this prompt and the plan disagree, the PLAN wins.

GROUND TRUTH (adversarially verified, trust it): reference seed plumbing already exists
end-to-end -- graphviz adapter passes -Gseed/-Gstart (dagua/eval/competitors/
graphviz_competitor.py:413-415); scripts/ogdf_runner.cpp consumes seed via ogdf::setSeed/
srand (lines ~307-322, binary already built); run_benchmark.py forwards seeds to
competitor.layout(..., seed=...) (~line 1475). The ONLY blockers you are fixing:
(1) `_BASE_ENGINE_STOCHASTICITY` in dagua/eval/variants.py (~line 2110) marks the
graphviz/ogdf/igraph-deterministic reference families non-stochastic, so
`seeds_for_engine()` in scripts/run_benchmark.py returns [None] for them;
(2) IgraphSugiyama in dagua/eval/competitors/igraph_competitor.py (~line 200) lacks
`uses_igraph_rng=True`, silently DROPPING the seed (its siblings at lines ~215/247/267/
293/305 have it).

CHANGES (3 files MAX: scripts/run_benchmark.py, dagua/eval/competitors/
igraph_competitor.py, plus ONE test file tests/test_seed_refs_override.py):

1. run_benchmark.py: new CLI flag `--seed-refs <comma-separated engine names>`.
   Semantics: a RUN-SCOPED stochasticity override -- engines named (or whose
   original_engine resolves to a named engine; `__for__` synthetic variants route via
   original_engine, see dagua/eval/variants.py ~line 263 engine_is_stochastic) are
   treated as stochastic for THIS RUN ONLY: seeds_for_engine returns the run's seed list
   instead of [None]. The override MUST apply at BOTH call sites: job enumeration
   (~line 2158) AND position-recovery enumeration (~line 1115). The GLOBAL
   _BASE_ENGINE_STOCHASTICITY table is NOT modified -- flipping it would change record
   keys for every future run and break --resume on existing dirs (this constraint is
   load-bearing; the plan forbids it).
   Accept both base names (graphviz_sfdp) and synthetic names
   (graphviz_sfdp__for__classic_sfdp_default) in the flag; matching by base name covers
   all its __for__ variants.
2. igraph_competitor.py: add `uses_igraph_rng=True` to IgraphSugiyama so the seed reaches
   igraph's RNG. Audit the sibling adapters in the same file for any other missing
   uses_igraph_rng and fix those too (report which). Do NOT assume seeding changes
   sugiyama's output -- whether it varies is a later probe's question.
3. Tests (tests/test_seed_refs_override.py):
   - seeds_for_engine with override: named ref returns the seed list; unnamed ref still
     [None]; synthetic __for__ name routes via original_engine.
   - WITHOUT the flag: behavior byte-identical to today (regression lock: the function
     returns [None] for graphviz_sfdp et al.).
   - record-key shape: overridden engines produce `::seed{N}` keys, non-overridden keep
     `::deterministic` (use build_record_key / the key helper directly).
   - IgraphSugiyama now passes seed into the igraph RNG context (mock/spy on the RNG
     seeding helper `_igraph_rng_seed` or equivalent -- assert it is ENABLED for
     sugiyama).
   - seed=None path through an overridden engine is not required (the flag implies
     seeded runs) but must not crash.
   Keep tests hermetic -- no graphviz/ogdf binaries invoked (unit-level; mock layouts).
</task>

<completeness_contract>
Done means: `python -m pytest tests/test_seed_refs_override.py -x -q` green;
`python -m pytest tests/test_distributional_fidelity.py -q` still green (no collateral);
`python3 scripts/run_benchmark.py --help` shows the flag; ruff check clean on touched
files; ONLY the 3 named files modified. Do NOT git commit (CC commits after review).
</completeness_contract>

<verification_loop>
Iterate until green. If a line number drifted, find the construct by name -- the cited
functions/tables all exist (verified within the last day).
</verification_loop>

<action_safety>
Touch ONLY the 3 named files. Never modify _BASE_ENGINE_STOCHASTICITY. Never invoke
layout binaries in tests. No commits.
</action_safety>

<default_follow_through_policy>
Most reasonable low-risk interpretation; stop only for genuine correctness walls.
Do not expand into dispatch (ii) (provenance stamping) -- that is a separate task.
</default_follow_through_policy>
