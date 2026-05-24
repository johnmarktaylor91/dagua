# Round 31 Infra Recovery Summary

## Changes

- Scoped `scripts/run_benchmark.py` watchdog handling to futures that can occupy worker slots. Expired active futures are recorded individually as `watchdog: future exceeded timeout`; queued peers are preserved instead of being marked as pool-wide stuck errors.
- Set `WATCHDOG_TIMEOUT` default to `600.0`.
- Added Round 31 timeout caps in `dagua/eval/variants.py`:
  - `classic_neulay`: `180`
  - `classic_sgd2_multi`: `120`
  - `classic_davidson_harel`: `180`
- Added per-variant `max_nodes` metadata and wired `VariantCompetitor` / benchmark variant construction to honor explicit caps:
  - NeuLay variants: `1500`
  - `classic_sgd2_multi_with_crossing`: `500`
  - Other SGD2 multi variants: `2000`
  - Davidson-Harel variants: `300`
- Verified R30 NeuLay `torch.enable_grad()` protection is active and added finite-value guards after NeuLay optimizer steps.
- Added explicit tracking comments for missing upstream NeuLay and SGD2 multi references so missing adapters remain visible as errors, not silent zero-pair rows.

## Commits

- `28e139c` `fix(bench): round 31 infra -- scoped watchdog`
- `ea9396f` `fix(bench): round 31 infra -- watchdog default`
- `bf2ca73` `fix(bench): round 31 infra -- timeout caps`
- `aa861f2` `fix(bench): round 31 infra -- max node caps`
- `c54795e` `fix(bench): round 31 infra -- neulay finite guard`
- `cd22ea7` `fix(bench): round 31 infra -- reference tracking`
- `86cbfab` `fix(bench): round 31 infra -- variant cap override`

## Smoke Test

Command:

```bash
python scripts/run_benchmark.py --seeds 5 --variants \
    --engines classic_neulay_default,classic_sgd2_multi_default,classic_davidson_harel_rounds50 \
    --graphs ba_500,small_world_500 \
    --output-dir /tmp/r31_infra_smoke \
    --timeout 300 --watchdog-timeout 1200
```

Result:

```text
[benchmark] Done: 30 total, 10 ok, 14 skipped, 6 errors, 0 timeouts
```

No `worker pool stuck` watchdog cascade occurred. Davidson-Harel skipped the 500-node graphs via `exceeds max_nodes=300`. NeuLay produced explicit `worker layout timeout exceeded` errors and then grouped seed skips after three consecutive errors.

## Verification

- `ruff check scripts/run_benchmark.py tests/test_scripts/test_run_benchmark.py --fix`: passed.
- `pytest tests/test_scripts/test_run_benchmark.py -x --tb=short -q`: `6 passed`.
- `ruff check dagua/eval/variants.py dagua/eval/competitors/classic_competitor.py scripts/run_benchmark.py tests/test_variant_registry.py --fix`: passed.
- `pytest tests/test_variant_registry.py::test_round31_infra_variant_max_node_caps tests/test_scripts/test_run_benchmark.py -q`: `7 passed`.
- `pytest tests/test_variant_registry.py::test_variant_competitor_honors_explicit_max_nodes tests/test_variant_registry.py::test_round31_infra_variant_max_node_caps -q`: `2 passed`.
- `pytest tests/test_layout/test_neulay_tsnet_grad.py -x --tb=short -q`: `3 passed`.
- `pytest tests/test_sgd2_multi_competitor.py -x --tb=short -q`: `6 passed, 6 skipped`.
- `ruff check .`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: `385 passed`.

Final Tier 2 command:

```bash
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

Stopped during collection with:

```text
ERROR collecting tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
```

This is outside the infra-recovery scope and overlaps the currently dirty DRL worktree changes.

## Notes

- Upstream reference availability in this environment:
  - `NeuLayReference().available()`: `False`
  - `SGD2MultiRef().available()`: `False`
- No dead code was intentionally introduced.
