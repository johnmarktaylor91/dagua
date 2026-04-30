# Round 22 Sugiyama Summary

## Changes

- Implemented the Round 22 staged scope from
  `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_sugiyama.md:266-268`.
- Filtered self-loops before cycle breaking/layering/expansion, carrying filtered edge
  weights with the filtered edge list.
- Added opt-in `fidelity_mode="igraph"` for Sugiyama. The mode enables stable-order
  early stop and igraph-style multiedge incidence barycenters.
- Added regression tests in `tests/test_layout/test_sugiyama_fidelity.py`.

## Measurement

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama igraph_sugiyama \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/sugiyama/baseline
```

Baseline result:

```text
graphs: 5
median: 0.000000
p25: 0.000000
p75: 0.000000
p95: 0.026252
worst: mixed_width_labels 0.032815
```

After command:

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama igraph_sugiyama \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/sugiyama/after
```

After result:

```text
graphs: 5
median: 0.000000
p25: 0.000000
p75: 0.000000
p95: 0.026252
worst: mixed_width_labels 0.032815
```

Median was already at the floor and stayed unchanged. Commit criterion is met by the
clean opt-in fidelity mode with regression tests.

## Verification

```text
ruff check dagua/layout/ops/sugiyama.py dagua/layout/ops/pipelines/sugiyama.py tests/test_layout/test_sugiyama_fidelity.py --fix
All checks passed!
```

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

```text
pytest tests/test_layout/ -x --tb=short -q -k "sugiyama"
3 passed, 293 deselected in 0.27s
```

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q -k "sugiyama or graph"
75 passed, 256 deselected, 1 warning in 280.55s (0:04:40)
```

Final Tier 2 was attempted and blocked by an unrelated collection error:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR collecting tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
```

## Residuals

- The default live-compare path does not pass `fidelity_mode="igraph"`, so the before/after
  subset remains numerically unchanged.
- Component packing and cyclic igraph layering remain separate larger-scope residuals.
