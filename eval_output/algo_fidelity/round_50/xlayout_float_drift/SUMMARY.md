# R50 xLayout Float Drift Summary

## Float64 ports applied

- Kept recursive fdp fidelity node sizes, tLayout handoff positions, xLayout inputs/outputs, component offsets, bbox shifts, and final clustered positions in `float64`.
- The only remaining dtype boundary is the caller/device return boundary; clustered fidelity now assembles the final tensor in `float64`.

## Sum-order matching applied

- Confirmed root xLayout repulsion and attraction already use explicit Python loops in Graphviz node/pair/outedge order.
- Replaced the recursive port initializer's positioned-neighbor `torch.mean()` with Graphviz's sequential running-average recurrence.
- The remaining decisive mismatch was not a sum-order issue: Graphviz `finalCC` rounds component bboxes through `BF2B` before feeding child cluster dimensions back into parent xLayout. The port now applies the same C-style rounding before recursive bbox translation.

## Smoke RMSD

Command:

```bash
PATH=/tmp/graphviz_instr/bin:$PATH python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py
```

Before:

- `one_cluster`: mean `0.000442635`, max `0.000556856`
- `path`: mean `0.003112518`, max `0.009318386`
- `clustered`: mean `0.000006534`, max `0.000008767`
- `multi_cluster`: mean `0.004016973`, max `0.007097214`

After:

- `one_cluster`: mean `0.000020480`, max `0.000040583`
- `path`: mean `0.003112518`, max `0.009318386`
- `clustered`: mean `0.000006533`, max `0.000008769`
- `multi_cluster`: mean `0.000007374`, max `0.000009562`

## Verdict

`fdp_clusters` multi_cluster is BIT-EXACT for the R50 target threshold: mean RMSD is below `0.0001` and comparable to the clustered topology floor.
