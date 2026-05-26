# R53 FDP tLayout Iter-1 Diff

## High-Precision Trace

Re-ran the instrumented Graphviz FDP path fixture with `%.17g` coordinate
logging and re-ran Dagua's pure-Python tLayout trace at the same precision.

Fixture: Round 39 eight-node path, seed 1.

- Iter 0 post-update positions matched exactly at 17 significant digits.
- Iter 1 attraction displacement matched exactly before repulsion.
- The first non-matching sub-step was tLayout grid repulsion.

Initial first diff before the port:

```text
first_diff ('after_rep', 0, 'n1')
Graphviz disp: 9.6582774793989508, -2.1315539716846246
Dagua disp:    9.6582774793989525, -2.1315539716846246
```

## Divergent Step

Graphviz `gridRepulse` applies all same-cell pairs for the current cell first,
then applies each of the eight neighboring cells. Dagua was applying same-cell
pairs and neighbor-cell checks inside each source-node loop. That preserved the
same force set but changed floating-point accumulation order.

The observed Graphviz order for the first cell was:

```text
CELL 0 -1 -1 n6 n0
REP n6 n0
REP n0 n6
REP n6 n5
...
```

Dagua's previous order was:

```text
REP n6 n0
REP n6 n5
...
REP n0 n6
```

## Port Applied

Updated both tLayout loops to match Graphviz's grid walk:

- Port-aware recursive tLayout: sorted cell traversal, same-cell pass, then
  neighbor passes at `dagua/layout/ops/pipelines/fmmm.py:1826`.
- Flat tLayout: sorted cell traversal, same-cell pass, then neighbor passes at
  `dagua/layout/ops/pipelines/fmmm.py:3240`.

After the port, the comparable path seed 1 `tlayout_gAdjust` trace matched
Graphviz bit-for-bit:

```text
first None
maxdiff 0.0
```

## Smoke RMSD

Command:

```bash
PATH=/tmp/graphviz_instr/bin:$PATH python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py
```

Result:

```text
one_cluster: 0.000016076, 0.000004781, 0.000014142 (mean=0.000011666, max=0.000016076)
path: 0.000006322, 0.000009034, 0.000010134 (mean=0.000008497, max=0.000010134)
clustered: 0.000006883, 0.000008769, 0.000003945 (mean=0.000006533, max=0.000008769)
multi_cluster: 0.000004591, 0.000009562, 0.000006960 (mean=0.000007038, max=0.000009562)
```

Path seed 1 is now below the `<1e-4` target.

## Verification

- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed,
  `Success: no issues found in 1 source file`.
- `pytest tests/test_pipeline_fmmm.py -x --tb=short -q`: passed,
  `20 passed, 2 warnings in 1.02s`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection before reaching FDP tests:
  `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.
