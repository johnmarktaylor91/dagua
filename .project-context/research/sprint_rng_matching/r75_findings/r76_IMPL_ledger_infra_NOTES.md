# r76-D4 Ledger Infrastructure Notes

Date: 2026-07-03
Branch: r76/ledger-infra

## Changes

- `scripts/validate_benchmark_integrity.py`
  - Added overlay-aware `results.json` loading across repeated `--data-dir`.
  - Added param-sensitivity tripwire for synthetic `__for__` reference variants.
  - Added explicit clamp-equivalent whitelist entries:
    - `umap_graph` on `parallel_multiedge_bundle`: documented tiny-graph
      `n_neighbors` clamp equivalence from `r76_final_sprint_STATE.md`.
    - `graphviz_sfdp` on `*`: documented installed Graphviz 7.0.5 ignoring
      theta, maxiter/steps, and p_neg2/repulsiveforce attrs.
  - Added seed-era warnings for Dagua/reimplementation rows with fewer than 100
    matched reference seeds, including both seed ranges.
  - Added seeded-reference `__for__` row-count validation using `manifest.json`.

- `scripts/run_benchmark.py`
  - Prints explicit `EXCLUDED GRAPHS due to --max-nodes` for requested graphs
    filtered before scheduling.
  - Prints end-of-run `__for__` row counts when `--seed-refs` is active.
  - Returns nonzero if any selected `__for__` reference variant has zero rows.

- `scripts/definitive_fidelity_analysis.py`
  - Added fail-closed output handling: existing `--output` fails unless
    `--resume` or `--overwrite` is explicit.
  - Added atomic `--overwrite` replacement through a sibling temp file.
  - Added `--self-check` mode that scores requested payloads twice and diffs
    verdict fields: `quality_*`, `*_direct_equivalent`, `d_R`, and `mode`.

- Tests
  - Added `tests/test_benchmark_integrity_guards.py` smoke tests for
    param-identical reference failure and existing-output refusal.
  - Updated `tests/test_bench_large.py` hierarchy checkpoint test to match the
    documented loader contract: incomplete hierarchies are accepted so
    coarsening can continue from the last saved level.
  - Updated `tests/test_classic_competitor.py` registry-contract expectations:
    `classic_fcose` is registered; NeuLay and Graphviz tests now patch current
    adapter paths; TsNET includes `fidelity_mode`.
  - Updated `tests/test_config_defaults.py` for the documented
    `w_straightness=0.5` default after the 93-graph sweep.

## Validator Run

Command: `python scripts/validate_benchmark_integrity.py` over the same real
eval-output chain used by `r76_gem_rescore.sh`, with absolute paths under
`/home/jtaylor/projects/dagua/eval_output`.

Result: nonzero, as intended for existing stale artifacts.

Summary from `/tmp/r76_validator_real.log`:

- Param-sensitivity failures: 234
- Whitelist hits: 100
- Seed-era warnings: 241
- Failure families:
  - `ogdf_gem`: 103
  - `ogdf_fmmm`: 95
  - `igraph_mds`: 21
  - `ogdf_stress`: 15

Interpretation:

- `graphviz_sfdp` no-op variants were absorbed by the documented whitelist.
- Existing OGDF artifacts still contain the stale-param/no-op oracle class and
  are now loud failures instead of silent scorer inputs.
- `igraph_mds` and `ogdf_stress` also surfaced as param-identical families in
  existing artifacts and need triage before any future scoring trusts those
  reference variants.

## Determinism Verdict

Command: `scripts/definitive_fidelity_analysis.py --self-check --mode full`
on `/tmp/r76_maar_combos.txt` using the `r76_gem_rescore.sh` data-dir chain.

Result:

```text
overlay: 8779 combos resolved, 4639 would have era-mixed under union semantics
[self-check] deterministic verdicts for 12 combos
```

Verdict: scorer nondeterminism did not reproduce for the 12 MAAR combos,
including the observed suspect `random_dag_50::classic_gem_iters100`.

Root cause: none found; the added self-check confirms the current verdict path
is reproducible for the suspect set.

## Test Results

Passed:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_bench_large.py tests/test_classic_competitor.py tests/test_benchmark_integrity_guards.py -x -q
48 passed, 11 warnings

pytest tests/ -k "validate or benchmark_integrity" -q
13 passed, 3149 deselected, 34 warnings

pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
461 passed, 153 warnings

pytest tests/test_benchmark_integrity_guards.py -q
2 passed, 3 warnings
```

Final Tier 2:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

First run failed on stale `tests/test_config_defaults.py` expectation for
`w_straightness` (`expected 2.2`, actual documented default `0.5`). One fix
cycle updated the stale test.

Second run failed on unrelated pre-existing render smoke:

```text
tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
assert len(border_patches) >= 2
E assert 0 >= 2
```

Per the one-fix-cycle rule, this unrelated render failure is reported rather
than chased in this eval-infra task.

## Assumptions

- The param-sensitivity family key is `(graph_name, engine prefix before
  "__for__")`; variants are the full synthetic `__for__` engine names.
- Three lowest common seeds per family are enough for a fast tripwire, matching
  the task's sampling requirement.
- Row-count assertion means any row status counts as a row; success/failure is
  handled by normal benchmark status accounting.
- Existing stale eval artifacts should fail the new validator rather than be
  whitelisted unless there is documented evidence of legitimate equivalence.

## Controversial Choices

- Did not whitelist OGDF stale-param failures. Those are the oracle bug class
  this gate is meant to catch.
- Did not change scoring math or engine/pipeline behavior.
- Updated stale tests encountered by required gates where the current code or
  documented contract clearly disagreed with the old expectation.

## Concerns

- Real eval-output validator currently fails on `ogdf_gem`, `ogdf_fmmm`,
  `igraph_mds`, and `ogdf_stress` param-identical artifacts. Future scoring
  should regenerate or quarantine those refs before trusting affected combos.
- Final non-slow suite is still blocked by an unrelated double-border render
  smoke failure after the allowed one fix cycle.
- `TunableParam` metadata for `w_straightness` still advertises default `2.2`
  while `LayoutConfig` default is `0.5`; not changed here because config
  metadata edits were outside this task's scope.

## Knowledge

- `scripts/bench_large.py` intentionally accepts incomplete hierarchy
  manifests so interrupted coarsening can resume from the last complete saved
  level.
- `classic_fcose` is a real registered classic competitor.
- Graphviz competitor tests must patch
  `dagua.eval.competitors.graphviz_competitor._layout_with_graphviz_engine`,
  not the older `dagua.graphviz_utils.layout_with_graphviz` utility path.
- The scorer self-check can reuse normal combo/data-dir resolution and run
  entirely in-process without worker scheduling, isolating verdict
  nondeterminism from process-pool completion order.

## Commits

- Implementation commit: recorded in git after this notes file is committed.
