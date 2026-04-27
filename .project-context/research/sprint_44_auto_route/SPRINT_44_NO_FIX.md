# Sprint-44 No-Fix Report

Date: 2026-04-26
Branch: `codex/sprint-31a-gate-refinement`

## Outcome

No auto-routing fix shipped.

The Sprint-44 gate required at least two current dagua loss graphs to flip to a
tie or win by dispatching `LayoutConfig(algorithm=None)` directly to a
structural specialist pipeline. The empirical probe found no callable internal
specialist that improved a current loss into the `>= -0.5` composite tie band.

## Current Loss Sweep

I reran dagua default layout live on the 93 benchmark graphs with
`num_nodes <= 500`, using `PYTHONHASHSEED=0` and `LayoutConfig(seed=42)`.
External competitor positions were loaded from
`eval_output/benchmark_full/positions/` and re-scored through current
`dagua.metrics.evaluate(..., tier="full")`.

Result:

- Wins: 78
- Ties: 9
- Losses: 6
- Elapsed: 467.9s
- Artifact: `/tmp/sprint44_current_loss_sweep.json`

Current losses:

| graph | dagua default | best external | delta | best external engine | classified family | topology tags |
|---|---:|---:|---:|---|---|---|
| `disconnected_encoder_residual` | 81.186 | 85.634 | -4.447 | `elk_layered` | `GENERAL` | `planar_dag` |
| `dependency_500` | 55.649 | 59.508 | -3.859 | `elk_layered` | `GENERAL` | |
| `dense_pair_50` | 72.713 | 75.378 | -2.665 | `graphviz_dot` | `GENERAL` | `dense_dag` |
| `clustered_medium_5x20` | 70.070 | 72.291 | -2.221 | `graphviz_dot` | `GENERAL` | |
| `parallel_cycles_4x5` | 60.650 | 62.732 | -2.081 | `graphviz_sfdp` | `HYBRID` | |
| `dependency_graph_100` | 57.961 | 59.518 | -1.557 | `elk_layered` | `GENERAL` | |

This differs slightly from the historical four-loss table because the current
deterministic external artifact set exposes two dependency graph losses and the
previously known `disconnected_encoder_residual` loss.

## Specialist Probe

For each current loss I tried the callable internal specialist pipelines:

`sugiyama`, `fr`, `kk`, `fa2`, `drl`, `fmmm`, `gem`, `sgd2_multi`,
`stress_majorization`, `stress_sgd`, `sfdp`, `lgl`, `linlog`,
`native_force_directed`, `native_planar`, and `dagua_flat`.

Artifact: `/tmp/sprint44_loss_specialist_probe.json`

Best internal result per loss:

| graph | best internal algorithm | best internal score | delta vs best external | tie-band pass? |
|---|---|---:|---:|---|
| `disconnected_encoder_residual` | default `dagua_native` | 81.186 | -4.447 | no |
| `dependency_500` | default `dagua_native` | 55.649 | -3.859 | no |
| `dense_pair_50` | default `dagua_native` | 72.713 | -2.665 | no |
| `clustered_medium_5x20` | default `dagua_native` | 70.070 | -2.221 | no |
| `parallel_cycles_4x5` | `native_planar` | 62.104 | -0.628 | no |
| `dependency_graph_100` | default `dagua_native` | 57.961 | -1.557 | no |

The closest case was `parallel_cycles_4x5` routed to `native_planar`, but it
still missed the requested `>= -0.5` tie band against `graphviz_sfdp`.

## Pipeline Availability Notes

Two registry entries are not directly callable through
`LayoutConfig(algorithm="<name>")` on this HEAD:

- `native_layered_dag` raises `AttributeError` because the module exports only
  `build_native_layered_dag_pipeline`, not `layout_native_layered_dag_pipeline`.
- `native_hybrid` raises the same wrapper-missing `AttributeError`.

I did not fix those registry wrappers because doing so would not satisfy the
Sprint-44 class-routing gate by itself, and the no-fix path should avoid
unrelated code churn.

## Decision

No structural class predicate has strong empirical support for Sprint-44:

- Dot-like external wins on `dense_pair_50` and `clustered_medium_5x20`, but
  internal `sugiyama` is 20-25 composite points below the external dot result
  and far below current default.
- SFDP wins externally on `parallel_cycles_4x5`, but internal `sfdp` scores
  38.603. The only near miss is `native_planar` at -0.628 vs external best.
- ELK-layered wins on the dependency losses, but internal layered-style
  specialists did not approach ELK or default.

Auto-routing would therefore be speculative topology-based routing rather than
evidence-backed specialist dispatch. That violates the Sprint-44 constraint.
