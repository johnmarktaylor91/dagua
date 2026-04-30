# Round 22 residual: sgd2_multi

Pairing: `classic_sgd2_multi` vs `sgd2_multi_ref`.

## Attempted bundle

The top-ranked Round 21 levers were reference-side infrastructure, not dagua
math:

- Restore/pin actual upstream `GD2` source or improve adapter diagnostics
  (`ROUND_21_DIFF_sgd2_multi.md:396-399`, recommended again at lines 440-441).
- Align RNG sources by seeding Python `random` in the reference adapter
  (`ROUND_21_DIFF_sgd2_multi.md:401-404`, recommended at line 441).
- Align sampler semantics only after source restoration, because the upstream
  loop is missing (`ROUND_21_DIFF_sgd2_multi.md:406-409`, line 444).

I implemented the small adapter-side subset in a local attempt: require both
`gd2.py` and `criteria.py` before import, return a clearer missing-source
diagnostic, and call `random.seed(seed)` beside `torch.manual_seed(seed)` and
`np.random.seed(seed)`. I also added a focused regression test for those two
behaviors.

## Result

The prescribed live compare uses cached `sgd2_multi_ref` target positions from
`eval_output/benchmark_full`; it does not run the reference adapter. Therefore
the adapter-only fix produced no change in this measurement.

- Baseline median: `0.10648379474878311`
- After median: `0.10648379474878311`
- Median delta: `0.0`
- Aggregate verdict: unchanged on the five-graph subset

Because the commit criterion was not met, I reverted the source and regression
test changes.

## Verification blockers

- `pytest tests/test_layout/ -x --tb=short -q -k "sgd2_multi"` failed during
  collection before reaching sgd2 tests because an unrelated untracked
  `tests/test_layout/test_fa2_fidelity.py` imports missing
  `_FA2_REFERENCE_PACKAGE_ORDER`.
- A focused run of the temporary sgd2_multi regression file passed:
  `2 passed in 0.11s`.
- A later focused pytest run was blocked by an unrelated in-progress
  `dagua/layout/ops/sugiyama.py` import error:
  `_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY` was undefined.
- `ruff check dagua/eval/competitors/sgd2_multi_competitor.py tests/test_layout/test_sgd2_multi_fidelity.py --fix`
  passed during the local attempt.
- `mypy --follow-imports=silent dagua/cli.py` passed.

## Next step

The next productive round should either vendor/pin the actual `GD2` source used
to create the cached `sgd2_multi_ref` positions, or regenerate the cached
reference outputs after the adapter reproducibility fix. Without that, the
highest-ranked fixes cannot affect `algo_fidelity_live_compare.py`.
