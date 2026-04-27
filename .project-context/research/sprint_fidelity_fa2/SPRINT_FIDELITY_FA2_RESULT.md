# Sprint-FIDELITY-FA2 Result

## Summary

Closed `65.99%` of the measured top-5 `classic_fa2` versus `fa2_ref` fidelity
gap by passing the benchmark seed into the installed reference `ForceAtlas2`
constructor.

The root cause was adapter-side seed drift. The current environment uses
`fa2.forceatlas2.ForceAtlas2`, whose `init()` method creates positions with
`random.Random(self.seed)`. The adapter seeded global Python and NumPy RNGs but
did not pass `seed` into `ForceAtlas2(...)`, so `fa2_ref` started from entropy
even when the benchmark supplied `seed=42`. The dagua port already initializes
from the requested seed, so the two implementations were often comparing
different starting layouts rather than just different force math.

## Empirical Result

Comparison surface:

- 18 representative graphs, 10 to 200 nodes.
- `classic_fa2`: `steps=200`, `barnes_hut=True`, `barnes_hut_theta=1.2`,
  `seed=42`.
- `fa2_ref`: matching `iterations=200`, Barnes-Hut params, gravity/scaling, and
  outbound attraction distribution.
- Scoring used `dagua.metrics.full(..., stress_sources=32, stress_targets=128,
  crossing_samples=50000)` plus `composite()`.

Original top-5 positive gaps:

| graph | before delta | after delta |
|---|---:|---:|
| `heavy_tail_50` | 13.078 | 7.383 |
| `petersen` | 8.542 | 0.000 |
| `tree_63_b3` | 7.024 | 0.036 |
| `hex_6x7` | 4.036 | 3.657 |
| `protein_ppi` | 0.857 | 0.332 |

Mean top-5 positive gap: `6.708 -> 2.281`, closing `65.99%` of the measured
target gap.

No regression was observed in the measured target relation: the fix changes
only `fa2_ref` reference seeding, and the `classic_fa2` scores were unchanged.

## Validation

Passed:

```bash
black dagua/eval/competitors/fa2_competitor.py tests/test_fa2_ogdf_competitors.py
ruff check dagua/eval/competitors/fa2_competitor.py tests/test_fa2_ogdf_competitors.py --fix
mypy --follow-imports=silent dagua/eval/competitors/fa2_competitor.py dagua/cli.py
pytest tests/test_fa2_ogdf_competitors.py tests/test_pipeline_fa2.py -x --tb=short -q
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

Targeted FA2 result:

```text
33 passed in 1.56s
```

Targeted layout/graph result:

```text
258 passed, 1 warning in 1182.70s (0:19:42)
```

Blocked by unrelated pre-existing repo state:

```bash
ruff check . --fix
```

failed on untracked scripts with five `E501` line-length errors:

```text
scripts/cleanup_for_salvage_round.py:95:101
scripts/cleanup_watchdog_errors.py:73:101
scripts/flip_running_to_skipped.py:55:101
scripts/flip_running_to_skipped.py:62:101
scripts/restore_skip3_from_backup.py:94:101
```

```bash
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

failed during collection before reaching the FA2 tests:

```text
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

This is the same unrelated collection blocker recorded in the prior SGD2
fidelity artifact.

## Changed Files

- `dagua/eval/competitors/fa2_competitor.py`
  - Added `engine_kwargs["seed"] = seed` when the benchmark supplies a seed.
  - Kept the existing global Python/NumPy seeding for older FA2 packages that
    still read global RNG state.
- `tests/test_fa2_ogdf_competitors.py`
  - Extended the fake reference class to expose an explicit `seed` constructor
    parameter.
  - Asserted that `FA2Reference.layout(..., seed=42)` forwards the seed through
    the accepted-parameter filter.

## Follow-Up

The remaining measured gap is now concentrated in graph-specific force-law and
metric behavior, especially `heavy_tail_50` and `hex_6x7`. The next divergence
to inspect should be weighted-edge handling and Barnes-Hut numerical parity, not
initialization.
