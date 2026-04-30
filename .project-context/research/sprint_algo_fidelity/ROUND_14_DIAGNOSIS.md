# Round 14 Diagnosis -- DRL vs igraph

Status: RESIDUAL
Family: drl
Date: 2026-04-30

## Baseline

The requested live comparator name `classic_drl_default` is present in
`dagua/eval/variants.py`, but `scripts/algo_fidelity_live_compare.py` calls
the competitor registry directly and only accepts the base competitor
`classic_drl`. I used `classic_drl`, whose pipeline default is
`options="default"`, as the conservative equivalent.

Command:

```text
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,binary_tree,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_14/baseline_small
```

Result:

```text
graphs: 5
median: 0.206197
p25: 0.192176
p75: 0.263942
p95: 0.263942
worst: linear_3layer_mlp 0.263942
```

`binary_tree` was requested but not present in the live comparator graph set,
matching the Round 13 small-subset behavior.

## Findings

1. Default, coarsen, coarsest, and refine phase presets are aligned with igraph.
   Dagua defaults are in `dagua/layout/ops/drl.py:176`; igraph defaults are in
   `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:240`.

2. The `final` preset is not aligned. Dagua has
   `expansion=(50, 2000.0, 2.0, 1.0)` in `dagua/layout/ops/drl.py:213`, while
   igraph has `expansion_temperature=50`, `expansion_attraction=.1`, and
   `expansion_damping_mult=.25` in
   `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:374`.
   This is a high-confidence issue for `classic_drl_final`, but it does not
   explain the requested default-subset gap.

3. Dagua does use a density grid rather than plain O(N^2) repulsion. The
   implementation starts at `dagua/layout/ops/drl.py:376`; igraph's grid is in
   `/home/jtaylor/projects/_references/igraph/src/layout/drl/DensityGrid.cpp:90`.
   The exact boundary behavior differs: igraph returns a high density near grid
   boundaries, while dagua clamps positions into grid cells.

4. The default-path node acceptance rule diverges. igraph computes the analytic
   candidate and a random perturbation, then always accepts the lower-energy
   candidate between those two
   (`/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_graph.cpp:964`).
   Dagua also compares against the old coordinate and can reject both candidate
   moves (`dagua/layout/ops/drl.py:817`). This likely explains dagua's much
   lower within-seed variability in the baseline.

5. Edge cutting is structurally different. igraph erases the worst edge only
   from the current node's neighbor map
   (`/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_graph.cpp:1130`).
   Dagua removes it symmetrically from both endpoints in
   `dagua/layout/ops/drl.py:680`. This is a more invasive behavioral difference
   because it changes later directed neighbor traversals.

## Attempted Lever

I tested the smallest default-path divergence: change dagua to choose only
between the analytic and random candidates, matching igraph's node update rule.

Result:

```text
output: eval_output/algo_fidelity/round_14/post_fix
graphs: 5
median: 0.188797
p25: 0.175811
p75: 0.239951
p95: 0.239951
worst: linear_3layer_mlp 0.239951
```

Improvement was `0.206197 -> 0.188797`, or `0.017400`, below the `0.03`
commit threshold. The code change was reverted.
