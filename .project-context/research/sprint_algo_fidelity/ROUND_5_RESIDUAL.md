# Round 5 Residual: fmmm-vs-fdp

## Classification

`attempted_lever_no_signal: graphviz_fdp_force_model`

Advance to `sfdp` unless Round 6 explicitly elects to spend one final fdp
attempt on initialization alignment. The force-law-only lever did not improve
the fdp family median, despite matching the local Graphviz source formulas.

## Graphviz Source Confirmed

Read `lib/fdpgen/tlayout.c`, `grid.c`, `xlayout.c`, and `fdpinit.c` from the
local Graphviz checkout before editing. The relevant fdp defaults and formulas
are:

- `DFLT_K = 0.3`, in Graphviz inches.
- `DFLT_maxIters = 600`.
- `DFLT_smode = INIT_RANDOM`.
- Old repulsion: vector coefficient `K^2 / d^2`.
- New repulsion: vector coefficient `K^2 / d^3`.
- Old attraction: vector coefficient `weight * d / L_e`.
- New attraction: vector coefficient `weight * (d - L_e) / d`.
- Position updates are clamped by a linearly cooling temperature.
- Default grid cell is `0.0`, so the ungridded path performs full all-pairs
  repulsion.
- `xlayout.c` runs a separate overlap expansion phase after the initial point
  layout.

## Baseline

| Run | Median RMSD | Worst graph | Worst RMSD | Graphs |
|---|---:|---|---:|---:|
| baseline | 0.247474 | center_port_backedge_hub | 0.440077 | 21 |

## Lever Tried

Temporarily added `force_model="graphviz_fdp"` and
`force_model="graphviz_fdp_new"` to the FM^3 pipeline:

- Graphviz old repulsion scaled Dagua's exact/Barnes-Hut repulsion by `K^2`.
- Graphviz new repulsion scaled by `K^2 / d`.
- Graphviz old attraction used `delta * weight * d / L_e`.
- Graphviz new attraction used `delta * weight * (d - L_e) / d`.
- Base edge length for Graphviz modes was set to `0.3 * 72 = 21.6` points.
- `classic_fmmm` temporarily tried `graphviz_fdp` in its selector and used it
  for the non-selector path.

## Result

The lever regressed the family median and was reverted.

| Metric | Baseline | Attempted force model | Delta |
|---|---:|---:|---:|
| Median RMSD | 0.247474 | 0.257227 | +0.009753 |
| Worst RMSD | 0.440077 | 0.440077 | +0.000000 |
| p95 RMSD | 0.415475 | 0.413135 | -0.002340 |

Largest per-graph improvements occurred on `shape_and_routing_matrix`
(-0.097963), `moe_router_sparse` (-0.063684), and `edge_label_braid`
(-0.054453), but the median moved upward because `tl_mlp_3layer` regressed
from `0.244654` to `0.268921` and the unchanged middle-ranked graphs shifted
the median to `parallel_multiedge_bundle` at `0.257227`.

## Diagnosis

Force-law alignment alone is not enough inside Dagua's current FMMM framework.
The temporary implementation made the local force expressions closer to
Graphviz `tlayout.c`, but the surrounding solver still differs materially:

- Graphviz fdp starts from deterministic random rectangle/ellipse positions
  seeded by `start`/`DFLT_seed`; Dagua seeds through FR on the coarsest level.
- Graphviz fdp uses a 600-iteration linearly cooled point-node pass; Dagua uses
  a multilevel budget and exponential cooling.
- Graphviz fdp runs post-layout overlap expansion in `xlayout.c`; Dagua's FMMM
  finalization normalizes positions without that phase.

## Recommendation

Do not commit the force-law-only patch. If fdp gets one more attempt, try
Graphviz fdp random initialization alignment first. Otherwise classify fdp as a
principled residual requiring a fuller FR-style fdp solver and continue to
`sfdp`.
