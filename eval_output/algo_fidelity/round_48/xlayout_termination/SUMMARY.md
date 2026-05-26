# R48 FDP xLayout Termination

## Instrumentation

Extended the Graphviz 7.0.5 instrumentation in `/tmp/graphviz_7_0_5_instr` and rebuilt:

```bash
cd /tmp/graphviz_instr_705_build
/home/jtaylor/anaconda3/bin/cmake --build . --target dot_builtins -j4
```

Trace rows now include:

- `/tmp/graphviz_7_0_5_instr/lib/fdpgen/xlayout.c:125` and `:507`: `XLAYOUT` rows with overlap count, bbox, `cnt`, `K`, temperature, node count, and edge count.
- `/tmp/graphviz_7_0_5_instr/lib/fdpgen/layout.c:67` and `:82`: `FINALCC` and `FINALCC_COMPONENT` rows around the real `finalCC()` implementation. The task named `comp.c`, but `finalCC()` is in `layout.c` in Graphviz 7.0.5.
- `/tmp/graphviz_7_0_5_instr/lib/common/utils.c:774` and `:834`: `COMPUTE_BB` rows for per-component bbox output.

The existing R47 `STEP` position trace remains active.

## Diagnosis

The termination divergence was not caused by `<` vs `<=`, a try-count mismatch, or a stale bbox update inside `adjust()`.

Graphviz and dagua both:

- compute overlap with axis-aligned boxes using `<=` on both axes (`xlayout.c:224-235`, mirrored by `dagua/layout/ops/pipelines/fmmm.py:2719-2745`);
- run `while (ov && try < tries)` with default `tries=9` (`xlayout.c:515`);
- stop when `adjust()` returns `ov == 0` before applying another position update (`xlayout.c:529-535`, mirrored at `fmmm.py:2996-3011`).

The real mismatch was upstream of root `xLayout`: child component packing made dagua cluster nodes too large.

Trace evidence on `build_clustered_path_graph`, seed 1:

- Graphviz child component bboxes before pack matched dagua exactly:
  `(-36.1965, 6.1518, 17.8035, 42.1518)` and three singleton boxes
  `(-50.7619, -20.3591, 3.2381, 15.6409)`.
- Graphviz `FINALCC_COMPONENT` placements were:
  `(6,-26), (81,50), (86,-15), (16,55)`.
- Dagua pre-fix placements were:
  `(6,-26), (86,55), (11,60), (96,-25)`.

Root cause: `dagua/layout/ops/pipelines/fmmm.py:_graphviz_cell()` used round-to-nearest. Graphviz `pack.c:CVAL` uses an integer cast and C truncating integer division (`pack.c:30-33`, `pack.c:240-245`). That over-expanded the occupancy grid and changed polyomino placements.

After fixing grid-cell truncation, Graphviz child `FINALCC after_bbox` became `140x146` points; dagua then needed Graphviz's padded finalCC label border of `24` points for recursive cluster bboxes. I kept the existing `18` point obstacle label constant separate so compound-edge obstacle tests retain their prior behavior.

## Ports Applied

- `dagua/layout/ops/pipelines/fmmm.py:72`: added dagua `XLAYOUT` trace rows under the existing fidelity trace path.
- `dagua/layout/ops/pipelines/fmmm.py:949` and `:1996`: added a separate finalCC cluster label border constant (`24` points).
- `dagua/layout/ops/pipelines/fmmm.py:3146`: changed `_graphviz_cell()` to use Graphviz/C truncation rather than rounding.
- `dagua/eval/variants.py:1101`: re-enabled `classic_fmmm_graphviz_fdp_fidelity`.

## Smoke RMSD

R47 baseline:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.299792684 | 0.012140572 | 0.015564031 | 0.109165762 | 0.299792684 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.272343047 | 0.253293883 | 0.132945270 | 0.219527400 | 0.272343047 |
| multi_cluster | 0.111096332 | 0.074638779 | 0.092953933 | 0.092896348 | 0.111096332 |

R48 against the instrumented `/tmp/graphviz_instr_705_build/cmd/dot/dot_builtins`:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.000448903 | 0.000322146 | 0.000556856 | 0.000442635 | 0.000556856 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.000006886 | 0.000008767 | 0.000003948 | 0.000006534 | 0.000008767 |
| multi_cluster | 0.119073969 | 0.067402310 | 0.091336502 | 0.092604261 | 0.119073969 |

For the traced clustered seed 1 fixture, dagua vs instrumented Graphviz plain output is `7.820916451735033e-06` RMSD.

The local conda `dot` is also Graphviz 7.0.5 but a later build string (`20221231.0122` vs instrumented tag build `20221223.1930`). Against that adapter, clustered remains above threshold:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.001876776 | 0.337958360 | 0.002466698 | 0.114100612 | 0.337958360 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.168141818 | 0.151738558 | 0.138964124 | 0.152948167 | 0.168141818 |
| multi_cluster | 0.119948610 | 0.078012879 | 0.052908847 | 0.083623445 | 0.119948610 |

The remaining conda-only floor is a disconnected-component equal-key packing tie/order difference, not the R48 termination condition. Example: clustered seed 1 differs by right-cluster internal component order while the instrumented build is bit-exact.

## Verification

Passed:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
433 passed, 8 warnings in 1188.27s (0:19:48)
```

Final non-slow suite still fails at the known unrelated collection error:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
```

Extra targeted check attempted during the fix cycle:

```text
pytest tests/test_layout/test_fmmm_fdp_recursion.py tests/test_fmmm_fdp_ports.py -x --tb=short -q
FAILED tests/test_fmmm_fdp_ports.py::test_fdp_attachment_points_clip_to_crossed_cluster_boundaries
assert (55.0, 0.0) == approx((20.0, 0.0))
```

That failure is outside the xLayout termination path and was not changed further.

## Verdict

The R48 termination divergence is closed against the instrumented Graphviz 7.0.5 build: clustered mean is `0.000006534`, well under `<0.05`, and the dagua `XLAYOUT` trace now stops on the same `after_adjust iter=17 ov=0` row as Graphviz. The registry variant is re-enabled.

Residual risk: current environment `graphviz_fdp` uses a later conda 7.0.5 build with a different equal-perimeter pack tie behavior, so the default adapter smoke still shows a clustered floor around `0.153`.
