# Round 52 FDP tLayout Pure-Python Summary

## Implementation

Replaced the Graphviz FDP fidelity `tLayout` tensor mutation loop in
`dagua/layout/ops/pipelines/fmmm.py` with Python `float` list state for:

- flat component initial positions, displacements, repulsion, attraction, and
  `updatePos`;
- recursive component initial positions with ports, port-aware repulsion, and
  port boundary clamping;
- tensor conversion only at trace and phase boundaries.

The port keeps Graphviz's `gAdjust` sequencing: reset displacements and grid,
apply all outgoing attractive forces, walk grid repulsion in cell insertion
order with 8-neighbor checks, then update positions in node order.

## Before/After Smoke RMSD

Command:

```bash
PATH=/tmp/graphviz_instr/bin:$PATH python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py
```

Before:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.000016076 | 0.000004781 | 0.000040583 | 0.000020480 | 0.000040583 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.000006883 | 0.000008769 | 0.000003945 | 0.000006533 | 0.000008769 |
| multi_cluster | 0.000004591 | 0.000009562 | 0.000007970 | 0.000007374 | 0.000009562 |

After:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.000016076 | 0.000004781 | 0.000040583 | 0.000020480 | 0.000040583 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.000006883 | 0.000008769 | 0.000003945 | 0.000006533 | 0.000008769 |
| multi_cluster | 0.000004591 | 0.000009562 | 0.000007970 | 0.000007374 | 0.000009562 |

## Verification

- `PATH=/tmp/graphviz_instr/bin:$PATH python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py`:
  failed target; path seed 1 remains `0.009318386`.
- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed,
  `Success: no issues found in 1 source file`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: passed,
  `433 passed, 8 warnings in 1179.73s`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection before reaching FDP tests:
  `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.

## Verdict

Remaining residual. The real pure-Python scalar tLayout port is in place, but
the required smoke target is not met: path seed 1 remains above `1e-4`.

Trace comparison still shows the R51 residual at `tlayout_gAdjust` iteration 1
and propagation through the path seed 1 `xLayout` boundary. No subprocess
delegation or Graphviz binary shortcut was added.
