<task>
Implement P1a dispatch (ii) of the r71 fidelity-completion plan
(/home/jtaylor/projects/dagua). Read FIRST: .project-context/research/sprint_rng_matching/
PLAN_r71_fidelity_completion.md sec. 2a (git-SHA provenance bullet) + Appendix B item 2.
Dispatch (i) is already committed (--seed-refs override). This task is PROVENANCE ONLY.

CHANGES (3 files MAX):

1. scripts/run_benchmark.py: stamp `git_sha` (subprocess git rev-parse HEAD, once at
   startup, "unknown" fallback) into the run metadata block AND into every results.json
   row it writes (rows currently carry graph_name/engine_name/seed/status/...; add
   git_sha). Existing stores without the field must keep loading everywhere (additive,
   no schema break).
2. scripts/merge_benchmark_datasets.py: every merged row gains `source_dir` (the
   originating store's directory name); rows already carrying source_dir keep theirs
   (merge-of-merge safe).
3. scripts/definitive_fidelity_report.py: new hard assertion, same pattern as
   check_no_mixed_modes: given an OPTIONAL config file eval_output/fidelity_definitive/
   fixed_engines.json of {engine: {fixed_sha, pre_fix_dirs: [...]}} (absent file = check
   vacuously passes), assert that NO row consumed by the report for a listed engine has
   git_sha in/before the pre-fix era (row git_sha missing OR row source_dir in
   pre_fix_dirs => VIOLATION). Strict mode fails the build with a clear message;
   --no-strict records a warning. Plus a unit-style smoke in the report's existing
   self-checks if it has any; otherwise verify by running the report dry on
   /tmp/r70_smoke.jsonl with a synthetic fixed_engines.json (must FAIL) and without
   (must pass) -- print both outcomes.
</task>
<completeness_contract>
Done = the dry-run pair above behaves as specified; `python3 scripts/run_benchmark.py
--help` works; a 2-graph 1-engine micro-benchmark run (e.g. --engines classic_fr_steps50
--graphs grid_5x5 --seeds 1 --output-dir /tmp/r71_prov_smoke) produces rows carrying
git_sha; ruff clean on the 3 files; ONLY those files modified; no commits.
</completeness_contract>
<action_safety>Touch only the 3 named files + /tmp. No benchmark beyond the micro-smoke.</action_safety>
<default_follow_through_policy>Reasonable low-risk interpretation; this is mechanical.</default_follow_through_policy>
