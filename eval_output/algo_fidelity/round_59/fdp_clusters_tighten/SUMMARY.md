# R59 FDP Clusters Tighten

## Phase A: Residual Trace Source

Re-ran the instrumented Graphviz 7.0.5 binary at `/tmp/graphviz_instr/bin/dot`
with 17-significant-digit `STEP` traces and compared against Dagua's FDP trace
path at `/tmp/dagua_fdp_trace.log`.

First `> 1e-12` coordinate divergence by topology:

| topology | seed | first divergent row |
|---|---:|---|
| path | 1 | `xlayout_adjust 0 n0`, `dy=2.04e-12` |
| path | 2 | `xlayout_adjust 0 n2`, `dx=2.85e-11` |
| path | 3 | `xlayout_adjust 0 n0`, `dy=3.09e-10` |
| star | 1 | `xlayout_adjust 0 n0`, `dx=8.00e-11` |
| star | 2 | `xlayout_adjust 0 n0`, `dx=2.43e-11` |
| star | 3 | `xlayout_adjust 0 n0`, `dy=9.06e-11` |
| clustered | 1 | none across `3634` shared `STEP` rows |
| clustered | 2 | none across `3644` shared `STEP` rows |
| clustered | 3 | none across `3652` shared `STEP` rows |
| multi_cluster | 1 | `xlayout_adjust 19 cluster_beta`, `dx=1.70e-10` |
| multi_cluster | 2 | `xlayout_adjust 18 cluster_beta`, `dx=3.54e-11` |
| multi_cluster | 3 | `xlayout_adjust 21 cluster_alpha`, `dx=5.12e-11` |

The force/update traces were already within the expected machine-noise range.
The smoke floor was larger than these trace deltas, so the residual source was
the final Graphviz output boundary: `finalCC` translates from integer-point
bounding boxes, then the JSON/plain renderer exposes coordinates through
five-significant-digit text formatting.

## Port

- `dagua/layout/ops/pipelines/fmmm.py`: root component translation now uses
  C-style rounded lower-left bbox coordinates, matching Graphviz `BF2B`.
- `dagua/layout/ops/pipelines/fmmm.py`: fidelity-mode final coordinates are
  quantized through `%.5g` parsing to match the `graphviz_fdp` adapter's JSON
  coordinate precision.

## Smoke Verification

Command:

```bash
PATH=/tmp/graphviz_instr/bin:$PATH python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py
```

Result:

```text
one_cluster: 0.000000012, 0.000000008, 0.000000015 (mean=0.000000012, max=0.000000015)
path: 0.000000004, 0.000000000, 0.000000000 (mean=0.000000001, max=0.000000004)
clustered: 0.000000010, 0.000000015, 0.000000009 (mean=0.000000011, max=0.000000015)
multi_cluster: 0.000000012, 0.000000009, 0.000000016 (mean=0.000000012, max=0.000000016)
```

All `fdp_clusters` smoke maxes are below `1e-6`.
