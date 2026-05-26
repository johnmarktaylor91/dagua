# R46 FDP Deep Close

## First Divergence

The first confirmed source-level divergence was in recursive child
`initPositions`, before child `tLayout` iteration 0. For a non-port child node
with exactly one positioned neighbor, Graphviz fdp sets `x = 0.98 * p.x` but
`y = 0.9 * p.y`; the Dagua port used `0.98` for both coordinates.

Trace harness: `eval_output/algo_fidelity/round_46/fdp_deep/trace_compare.py`
writes `trace_seed1.json` with Dagua recursive checkpoints and Graphviz
`dot -v -Tplain -Kfdp` phase/final-position output.

## Source Citations

- Graphviz 7.0.5 `lib/fdpgen/tlayout.c:initPositions`: recursive port
  initialization and the asymmetric single-neighbor coefficients.
  https://gitlab.com/graphviz/graphviz/-/blob/7.0.5/lib/fdpgen/tlayout.c
- Graphviz 7.0.5 `lib/fdpgen/grid.c:addGrid`: grid cell node lists prepend
  new nodes, so accumulation order differs from append-based Python lists.
  https://gitlab.com/graphviz/graphviz/-/blob/7.0.5/lib/fdpgen/grid.c
- Graphviz 7.0.5 `lib/fdpgen/xlayout.c`: default xLayout params and try-local
  `K` used by attraction after overlap-removal retries.
  https://gitlab.com/graphviz/graphviz/-/blob/7.0.5/lib/fdpgen/xlayout.c
- Graphviz 7.0.5 `lib/fdpgen/layout.c:layout`: recursion sequence:
  derive graph, `fdp_tLayout`, `expandCluster`, recursive layout, delete ports,
  `fdp_xLayout`, pack/finalize.
  https://gitlab.com/graphviz/graphviz/-/blob/7.0.5/lib/fdpgen/layout.c

## Ports Applied

- `dagua/layout/ops/pipelines/fmmm.py:1398`: match Graphviz's one-neighbor
  recursive port initialization: `0.98` on x and `0.90` on y.
- `dagua/layout/ops/pipelines/fmmm.py:1517` and `:2495`: prepend grid cell
  entries to match Graphviz `addGrid`.
- `dagua/layout/ops/pipelines/fmmm.py:2550`: include fdp's default additive
  xLayout node separation of 4 points per side.
- `dagua/layout/ops/pipelines/fmmm.py:2654`: pass the try-local xLayout `K`
  into attraction instead of reusing the default constant.

## Smoke RMSD

Before R46 ports:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.018305101 | 0.313030088 | 0.311203930 | 0.214179706 | 0.313030088 |
| path | 0.076886609 | 0.023582772 | 0.019508183 | 0.039992521 | 0.076886609 |
| clustered | 0.269335653 | 0.187179207 | 0.196027235 | 0.217514031 | 0.269335653 |
| multi_cluster | 0.169847873 | 0.149791494 | 0.139368773 | 0.153002714 | 0.169847873 |

After R46 ports:

| topology | seed 1 | seed 2 | seed 3 | mean | max |
|---|---:|---:|---:|---:|---:|
| one_cluster | 0.011227314 | 0.012140536 | 0.015564053 | 0.012977301 | 0.015564053 |
| path | 0.009318386 | 0.000009034 | 0.000010134 | 0.003112518 | 0.009318386 |
| clustered | 0.271441490 | 0.231402623 | 0.191467755 | 0.231437289 | 0.271441490 |
| multi_cluster | 0.174154556 | 0.129153126 | 0.172191839 | 0.158499840 | 0.174154556 |

## Verdict

Acceptable floor not reached. The one-cluster recursion and flat fdp kernels
improved substantially, but sibling clustered fixtures remain around
0.23 mean clustered RMSD. `classic_fmmm_graphviz_fdp_fidelity` remains disabled.

The remaining floor is not safe to guess through: public Graphviz `dot -v` only
exposes phase boundaries, not per-iteration positions, and this environment has
no installed private fdp headers for a libgraphviz trace helper. The next
required step is an instrumented Graphviz 7.0.5 build that dumps `ND_pos` after
each `gAdjust`/`updatePos` in `layout cluster_left` and `layout cluster_right`.
