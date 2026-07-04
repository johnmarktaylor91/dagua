<task>
r76-D4: ORACLE INVARIANTS + eval-infra hardening -- turn this sprint's instrument lessons
into permanent gates. Context: this campaign found FOUR oracle bugs (stale ogdf_runner
ignoring params; umap adapter randn fallback at n<=3 ignoring params; seed-era mismatches
silently halving statistical power; --max-nodes silently excluding graphs from "family"
benches) plus one scorer nondeterminism suspect. Encode the tripwires so none can recur
silently. Read .project-context/research/sprint_rng_matching/r76_final_sprint_STATE.md
iteration log (2026-07-03) for full context.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-ledger-infra (branch r76/ledger-infra,
fresh off develop). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

WORK ITEMS:
1. PARAM-SENSITIVITY TRIPWIRE in scripts/validate_benchmark_integrity.py: for every
   reference engine family with multiple __for__ param variants on the same graph, compare
   same-seed position tensors across variants; bit-identical across ALL variants = FAIL
   unless the (engine, graph-or-*) pair is in an explicit CLAMP_EQUIVALENT_WHITELIST
   (seed the whitelist with the PROVEN cases: umap n_neighbors clamping at tiny N
   [parallel_multiedge_bundle: default==nn5==nn30 legitimately]; graphviz_sfdp theta/
   steps/p_neg2 attrs ignored by installed 7.0.5 [ALL graphs]; document each entry with
   the evidence source). Sample seeds (3 per combo) for speed. Clear FAIL message naming
   the engine/graph/variants.
2. SEED-ERA GUARD in the same validator: for each combo, report the matched-seed count
   between dagua rows and reference rows across the given dirs; matched < 100 -> WARN
   with the two seed ranges printed (the 42-start vs 100-start era mismatch class).
3. __for__ ROW-COUNT ASSERTION: after any run_benchmark invocation with --seed-refs, the
   output results.json must contain >0 rows for EVERY requested __for__ reference variant;
   add a check function to the validator + a loud end-of-run summary line in
   scripts/run_benchmark.py listing __for__ row counts (and print EXCLUDED GRAPHS when
   --max-nodes filters any requested graph -- the silent >300-node exclusion bit three
   engine families this sprint).
4. OVERWRITE-OR-FAIL: scripts/definitive_fidelity_analysis.py --output must refuse to
   APPEND to an existing output file (the gem rescore silently doubled to 630 rows);
   default = fail if exists, --overwrite flag replaces atomically.
5. SCORER DETERMINISM CHECK: add a --self-check mode to definitive_fidelity_analysis.py
   (or a small script scripts/check_scoring_determinism.py) that scores a given combos
   file TWICE and field-diffs the verdict fields (quality_*, *_direct_equivalent, d_R,
   mode) -- exits nonzero listing any combo whose verdicts differ. RUN IT on
   /tmp/r76_maar_combos.txt (12 combos; the observed suspect:
   random_dag_50::classic_gem_iters100 flipped identical between two same-chain runs) with
   the dir chain from .project-context/research/sprint_rng_matching/r76_scratch/
   r76_gem_rescore.sh. If nondeterminism reproduces, ROOT-CAUSE it (suspects: worker
   scheduling affecting RNG consumption, unseeded bootstrap, hash ordering) and FIX it --
   verdicts must be reproducible.
6. FIX the two pre-existing test failures on develop:
   (a) tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest --
   scripts/bench_large.py::_load_hierarchy_checkpoint accepts an incomplete manifest the
   test expects rejected; decide which is right by reading the script's documented
   contract (the script docstring says it accepts incomplete hierarchies -- if so, fix the
   TEST to match the documented behavior and note it; if the docs say reject, fix the
   loader).
   (b) tests/test_classic_competitor.py::test_classic_competitor_names_match_expected_values
   -- 'classic_fcose' exists but is missing from the expected-names list; update the list
   (verify classic_fcose is a real registered engine first).

GATES: pytest tests/test_bench_large.py tests/test_classic_competitor.py -x -q green (the
two fixed tests now PASS); pytest tests/ -k "validate or benchmark_integrity" green if such
tests exist (add smoke tests for tripwires 1+4: synthetic param-identical refs -> FAIL;
append attempt -> FAIL); ruff clean; determinism check output documented (and green after
any fix). Validator run against the REAL eval_output dirs must complete and its findings
must be REPORTED in the notes (expected: it will flag the known param-noop families -> they
must be in the whitelist; anything NEW it flags, list prominently).

DELIVERABLES: .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_ledger_infra_NOTES.md (what was added, validator run output summary, determinism
verdict + root cause if any, test-fix rationale, commit shas). Conventional commits on
r76/ledger-infra; NO AI attribution; no push/merge. ASCII only.
</task>
<completeness_contract>
Done = all 6 items implemented/resolved with gates green and committed, OR precise notes on
which item is blocked and why. Item 5's determinism verdict (reproduces or not; root cause
if yes) is MANDATORY -- the final ledger cannot ship on a nondeterministic scorer.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/ledger-infra only. Do not modify engine/pipeline code or
scoring MATH -- only validation, logging, output-safety, determinism plumbing, and the two
tests. Never modify files outside the worktree except reading eval_output (read-only) and
/tmp scratch.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
