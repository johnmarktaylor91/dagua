# R47 FDP Instrumented Build

## Instrumented Graphviz Build

The checkout at `/home/jtaylor/projects/_references/graphviz` was on main
(`14.1.6~dev`), not tag `7.0.5`. I fetched tag `7.0.5` and built a detached
instrumented worktree at `/tmp/graphviz_7_0_5_instr`.

Repro commands:

```bash
cd /home/jtaylor/projects/_references/graphviz
git fetch origin refs/tags/7.0.5:refs/tags/7.0.5
git worktree add --detach /tmp/graphviz_7_0_5_instr 7.0.5
cd /tmp/graphviz_7_0_5_instr
# flex was missing from the machine image; I installed it into the active conda env:
conda install -y -c conda-forge flex
/home/jtaylor/anaconda3/bin/cmake -S . -B /tmp/graphviz_instr_705_build \
  -DCMAKE_INSTALL_PREFIX=/tmp/graphviz_instr \
  -Denable_ltdl=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_PANGOCAIRO=TRUE \
  -DCMAKE_DISABLE_FIND_PACKAGE_CAIRO=TRUE
/home/jtaylor/anaconda3/bin/cmake --build /tmp/graphviz_instr_705_build --target dot_builtins -j4
```

The runnable wrapper is `/tmp/graphviz_instr/bin/dot`; it sets
`LD_LIBRARY_PATH=/tmp/graphviz_instr/lib` and executes the built
`dot_builtins.real`. Verification:

```text
dot_builtins.real - graphviz version 7.0.5 (20221223.1930)
```

Instrumentation was added in the 7.0.5 worktree:

- `/tmp/graphviz_7_0_5_instr/lib/fdpgen/tlayout.c:129` adds
  `dump_fdp_trace()`.
- `/tmp/graphviz_7_0_5_instr/lib/fdpgen/tlayout.c:436` dumps after
  `gAdjust()` calls `updatePos()`.
- `/tmp/graphviz_7_0_5_instr/lib/fdpgen/xlayout.c:113` adds
  `dump_fdp_trace()`.
- `/tmp/graphviz_7_0_5_instr/lib/fdpgen/xlayout.c:444` dumps after xLayout
  position updates.

Trace file: `/tmp/graphviz_fdp_trace.log`.

## Trace Findings

Fixture: `build_clustered_path_graph()`, seed `1`, rendered with:

```bash
rm -f /tmp/graphviz_fdp_trace.log
/tmp/graphviz_instr/bin/dot -Kfdp -Tplain -Gseed=1 -Gstart=1 /tmp/r47_clustered_seed1.dot
```

First divergence after R46 was immediately after the root sibling-cluster
`tLayout` pass. Graphviz child recursion for `cluster_left` emitted:

```text
STEP tlayout_gAdjust 0 n3 ...
STEP tlayout_gAdjust 0 _port_cluster_left_(3)_(4)_4 ...
```

Dagua initially emitted the whole internal path component
`port,n3,n2,n1,n0`. Source diagnosis: Graphviz `deriveGraph()` iterates only
edges present in the current Cgraph subgraph (`layout.c`), and the DOT fixture
declares the real path edges at root scope. Child cluster subgraphs therefore
receive generated boundary-port edges only.

After ports, Dagua and Graphviz matched all `3634` Graphviz-emitted trace rows
within `1e-6`. Dagua still emitted four extra root `xlayout_adjust` rows
(`17` and `18` for two root cluster nodes), so the remaining floor is after the
last Graphviz xLayout update and appears tied to residual cluster bbox/overlap
state rather than the force/update kernels.

## Ports Applied

- `dagua/layout/ops/pipelines/fmmm.py:40` adds Dagua trace output to
  `/tmp/dagua_fdp_trace.log`.
- `dagua/layout/ops/pipelines/fmmm.py:1165` scopes recursive real-edge
  grouping to the root level; non-root child levels now receive only generated
  port edges.
- `dagua/layout/ops/pipelines/fmmm.py:1186` names generated ports like
  Graphviz `portName()`, e.g. `_port_cluster_left_(3)_(4)_4`.
- `dagua/layout/ops/pipelines/fmmm.py:1267` orders connected components by
  derived-node order, matching Cgraph subgraph iteration.
- `dagua/layout/ops/pipelines/fmmm.py:1672` removes the recursive singleton
  shortcut so singleton child components still run Graphviz seeded
  `fdp_tLayout`.
- `dagua/layout/ops/pipelines/fmmm.py:1962` allows recursive packing to pass
  full component bboxes into the existing Graphviz tile packer.

`classic_fmmm_graphviz_fdp_fidelity` was not re-enabled because the clustered
mean stayed above `0.05`.

## Smoke RMSD

Before R47, from R46 after ports:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.011227314 | 0.012140536 | 0.015564053 | 0.012977301 | 0.015564053 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.271441490 | 0.231402623 | 0.191467755 | 0.231437289 | 0.271441490 |
| multi_cluster | 0.174154556 | 0.129153126 | 0.172191839 | 0.158499840 | 0.174154556 |

After R47:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.299792684 | 0.012140572 | 0.015564031 | 0.109165762 | 0.299792684 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.272343047 | 0.253293883 | 0.132945270 | 0.219527400 | 0.272343047 |
| multi_cluster | 0.111096332 | 0.074638779 | 0.092953933 | 0.092896348 | 0.111096332 |

## Verification

Passed:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_layout/test_fmmm_fdp_recursion.py tests/test_layout/ tests/test_graph.py -x --tb=short -q
439 passed, 8 warnings in 1180.01s (0:19:40)
```

Final required non-slow suite failed before running tests due an existing import
collection error outside this task's scope:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
```

## Verdict

Not acceptable yet. The instrumented binary exists and the first divergent
iteration was ported, but the clustered smoke floor remains `0.219527400` mean
RMSD. The current residual is after the last Graphviz-emitted xLayout step:
Dagua matches every Graphviz trace row and then continues two extra root
xLayout updates. Future work should instrument Graphviz component bounding
boxes and overlap counts around `finalCC()`, `compute_bb()`, and
`fdp_xLayout()` to close the remaining bbox/overlap termination mismatch.
