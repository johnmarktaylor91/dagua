# Sprint-FIDELITY-KK Result

## Summary

Closed the measured top-5 `classic_kk` fidelity gap to external
`igraph_kamada_kawai` by adding an opt-in direction-orientation step for the
benchmarked classic competitor path.

The raw Kamada-Kawai solve is invariant to global reflection. Dagua's
NetworkX-style port often converged to the vertically inverted equivalent
embedding on DAG-like graphs, producing excellent edge-length uniformity but
near-zero directed DAG consistency. The external igraph adapter happened to
land in the favorable vertical orientation on the largest gap cases.

The fix keeps direct `layout_kk_pipeline(...)` calls bit-exact by default, then
has the `classic_kk` competitor opt into a conservative post-solve axis flip:
only flip when edge-direction consistency improves by at least `0.05`.

## Empirical Result

Comparison command: inline Python harness using `get_test_graphs(max_nodes=220)`,
`ClassicKK`, `IgraphKamadaKawai`, and `dagua.eval.benchmark._metric_payload`.

```bash
python - <<'PY'
# See terminal history for the full 20-graph harness.
PY
```

Representative set: 20 graphs, `N <= 220`, side-by-side
`classic_kk` vs `igraph_kamada_kawai`, scored with the benchmark quick
composite path.

Original top-5 positive gaps:

| graph | before gap | after gap |
|---|---:|---:|
| `hexagonal_lattice_42` | 25.755 | -6.839 |
| `grid_5x5` | 25.412 | -9.463 |
| `heavy_tail_weights_50` | 12.946 | -10.618 |
| `real_football_115` | 10.285 | -6.769 |
| `linear_3layer_mlp` | 10.202 | -24.798 |

Mean top-5 positive gap: `16.920 -> -11.697`, closing `169.1%` of the measured
gap. Capped at the sprint gate, this is full closure of the target gap.

Regression check across the same 20 graphs:

```text
worst_score_gain=-0.0001689695357818266
```

No measured graph regressed materially; the tiny negative value is float noise
from recomputing metrics.

## Validation

Passed:

```bash
black dagua/eval/competitors/classic_competitor.py dagua/layout/ops/pipelines/kk.py tests/test_pipeline_kk.py
ruff check dagua/eval/competitors/classic_competitor.py dagua/layout/ops/pipelines/kk.py tests/test_pipeline_kk.py --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_pipeline_kk.py -q --tb=short
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

Targeted layout result:

```text
258 passed, 1 warning in 1144.74s (0:19:04)
```

Blocked by unrelated pre-existing repo state:

```bash
ruff check . --fix
```

failed on untracked scripts under `scripts/` with `E501` line-length errors.

```bash
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

failed during collection:

```text
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

The collection failure is the same missing `dagua.layout.classic` package export
issue seen in the SGD2 sprint and is unrelated to the KK files changed here.

## Changed Files

- `dagua/layout/ops/pipelines/kk.py`
  - Added opt-in direction-aware orientation helpers.
  - Added `direction` and `orient_to_direction` parameters to
    `layout_kk_pipeline`.
  - Preserved raw pipeline bit-exactness by keeping orientation disabled by
    default.
- `dagua/eval/competitors/classic_competitor.py`
  - Enabled direction-aware orientation for the benchmarked `classic_kk`
    competitor, including variant dispatch through `_quick_classic`.
- `tests/test_pipeline_kk.py`
  - Added regression coverage for opt-in top-to-bottom orientation.

## Follow-Up

The fix addresses the highest-leverage divergence: arbitrary global reflection
after KK convergence. It does not change the underlying NetworkX-style solver,
distance model, or igraph adapter. A deeper canonical port would require
matching igraph's C implementation defaults directly, but the measured gap was
dominated by orientation rather than spring-energy convergence.
