# Round 20 Fix: NeuLay

## Scope

Applied the top three old-code fidelity fixes from
`ROUND_19_DIFF_neulay.md` without touching the competitor adapter:

1. Added opt-in `fidelity_mode="old_code"` defaults for the checked-in
   `old_code/NeuLay-2.py` target: `dim=3`, `gcn_steps=40000`,
   `fdl_steps=1000000`, and absolute `query_radius=4.0`.
2. Added explicit `fdl_steps` direct-refinement semantics so callers can keep
   GCN and direct budgets separate.
3. Added absolute `query_radius` configuration; when omitted, historical
   factor-scaled behavior is preserved unless old-code fidelity mode is active.

Historical defaults remain unchanged for existing callers: 2D output,
`gcn_steps=2000`, total-budget-derived direct steps, and
`query_radius_factor * radius`.

## Reference Installability

The cloned reference at `/home/jtaylor/projects/_references/NeuLay` is still
not directly installable.

`find` found no packaging or requirements metadata under max depth 2.

`python -m pip install -e /home/jtaylor/projects/_references/NeuLay --dry-run`
failed:

```text
ERROR: file:///home/jtaylor/projects/_references/NeuLay does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
```

## Baseline

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_neulay neulay \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/neulay/baseline
```

Output:

```text
Wrote 30 rows to eval_output/algo_fidelity/round_20/neulay/baseline/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_20/neulay/baseline/multi_seed_summary.json
graphs: 2
median: 0.122014
p25: 0.113229
p75: 0.130799
p95: 0.137827
worst: linear_3layer_mlp 0.139584
```

## After

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_neulay neulay \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/neulay/after
```

Output:

```text
Wrote 30 rows to eval_output/algo_fidelity/round_20/neulay/after/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_20/neulay/after/multi_seed_summary.json
graphs: 2
median: 0.122014
p25: 0.113229
p75: 0.130799
p95: 0.137827
worst: linear_3layer_mlp 0.139584
```

The median is unchanged because the old-code fidelity path is opt-in and the
round explicitly excluded changes to `dagua/eval/competitors/neulay_competitor.py`.
The commit criterion is still met through a clean fidelity-mode addition with
regression tests.

## Verification

Passed:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_layout/ -x --tb=short -q -k "neulay"
4 passed, 247 deselected in 1.05s
```

Blocked:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

The broader Tier 1 pytest command stalled for several minutes after partial
progress and exited with code `-1` without a traceback or failure output.

Failed outside scope:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR collecting tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
```
