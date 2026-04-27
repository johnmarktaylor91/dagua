# sprint-27 small_world_100 polish research

## TL;DR

- Current live `dagua.layout(g)` for `small_world_100` scores **58.9995** under
  `dagua.metrics.full(...)` + `composite(...)`, matching the prompt's rounded
  **59.00**.
- The best simple chained polish I found is a centered vertical flip plus mild
  anisotropic scale: **`x *= 0.95`, `y *= -1.10`**, applied to the picker's
  running `pos`. It scores **59.3568**, a real **+0.3572** lift.
- This is jitter-stable under paired sigma `0.5` validation
  (`mean delta +0.3580`, `min +0.3500`, `max +0.3654`, `std 0.0046` over 12
  trials), so the lift is not a crossing-sampling artifact.
- It does **not** satisfy the strict sprint gate of `current + 0.5`
  (`59.4995` required), and it does not move the graph toward strong-win
  territory (`> 62.09` vs `igraph_sugiyama = 57.09`).
- Do **not ship** this candidate as a sprint-27 implementation. If the margin
  were ever relaxed, it must be protected by the existing composite picker and a
  narrow small-world/cyclic gate, because the raw transform badly regresses DAG
  protected wins.

## Per-metric diagnosis

Scoring used:

```text
graph = get_test_graphs(), name == "small_world_100"
pos = dagua.layout(graph, LayoutConfig(device="cpu"))
node_sizes = torch.tensor([[40.0, 20.0]] * N)
metrics = dagua.metrics.full(pos, graph.edge_index, node_sizes=node_sizes)
score = dagua.metrics.composite(metrics)
```

Baseline metrics:

| Metric | Value | Composite contribution | Headroom |
|---|---:|---:|---|
| `dag_consistency` | `0.5250` | `13.125` / 25 | Large, but cyclic orientation limits it |
| `edge_length_cv` | `0.0802` | `18.395` / 20 | Already excellent |
| `depth_spearman_rho` | `nan` | `0.000` / 15 | No useful topological depth signal here |
| `overlap_count` | `0` | `10.000` / 10 | Saturated |
| `edge_straightness_mean_deg` | `45.0338` | `0.000` / 10 | Main affine-polish opportunity |
| `crossing_rate` | `0.000208` | `9.979` / 10 | Saturated |
| `angular_res_mean_deg` | `54.1152` | `5.000` / 5 | Saturated |
| cluster neutral | n/a | `2.500` / 5 | No clusters |
| **Composite** |  | **58.9995** |  |

The graph is not losing to competitors because of crossings, overlap, angular
resolution, or CV. Those are already at or near the composite ceiling. The only
simple-transform headroom is straightness: edges are just over the 45-degree
cutoff, so the straightness term contributes zero. A vertical anisotropic
stretch can bring mean deviation below 45 degrees, but it immediately worsens
`edge_length_cv`. The best variants are therefore small, not dramatic.

Global orientation has only minor additional headroom. A vertical flip changes
`dag_consistency` from `0.5250` to `0.5300`, worth only `+0.125` composite. A
broader rotation sweep over 5-degree orientations did not find a projection
with enough forward-edge gain to offset the CV/straightness trade-off. This
makes sense for a directed Watts-Strogatz graph: the edge directions are cyclic
and locally rewired, so a single global up-axis cannot make many more edges
point forward.

The best candidate metrics:

| Metric | Baseline | Best candidate | Contribution delta |
|---|---:|---:|---:|
| `dag_consistency` | `0.5250` | `0.5300` | `+0.125` |
| `edge_length_cv` | `0.0802` | `0.0976` | `-0.348` |
| `edge_straightness_mean_deg` | `45.0338` | `42.3902` | `+0.587` |
| `crossing_rate` | `0.000208` | `0.000208` | `+0.000` |
| `angular_res_mean_deg` | `54.1152` | `50.9715` | `+0.000` because both cap at 5 |
| `overlap_count` | `0` | `0` | `+0.000` |
| **Composite** | **58.9995** | **59.3568** | **+0.3572** |

So the entire gain is the straightness term plus a tiny DAG flip, partly eaten
by worse edge-length uniformity. There is no evidence that a pure affine polish
can reach `+0.5`, let alone the `+3.09` needed to make this a strong win over
`igraph_sugiyama`.

## Algorithm sketch

This is the implementation shape I tested in `/tmp/sprint27_small_world_100_codex`.
It is intentionally written as a picker candidate, not as an always-on transform.

```python
from __future__ import annotations

from typing import Dict

import torch

from dagua.metrics import composite, full


def small_world_flip_stretch_candidate(pos: torch.Tensor) -> torch.Tensor:
    """Return the best researched small-world affine polish.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]`` from the current picker state.

    Returns
    -------
    torch.Tensor
        Candidate position tensor with shape ``[N, 2]``.
    """
    center = pos.mean(dim=0, keepdim=True)
    shifted = pos - center
    polished = torch.stack(
        [
            shifted[:, 0] * 0.95,
            shifted[:, 1] * -1.10,
        ],
        dim=1,
    )
    return polished + center


def score_positions(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> tuple[float, Dict[str, float]]:
    """Score positions on the sprint-27 full composite surface.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, Dict[str, float]]
        Composite score and raw metric dictionary.
    """
    metrics = full(pos, edge_index, node_sizes=node_sizes)
    return composite(metrics), metrics


def accept_small_world_flip_stretch(
    baseline_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> torch.Tensor:
    """Accept only if the existing picker gate proves a real improvement."""
    base_score, base_metrics = score_positions(baseline_pos, edge_index, node_sizes)
    candidate_pos = small_world_flip_stretch_candidate(baseline_pos)
    cand_score, cand_metrics = score_positions(candidate_pos, edge_index, node_sizes)
    no_new_overlaps = cand_metrics["overlap_count"] <= base_metrics["overlap_count"]
    if no_new_overlaps and cand_score >= base_score + 0.5:
        return candidate_pos
    return baseline_pos
```

As written, this returns the baseline for `small_world_100`, because the measured
lift is `+0.3572`, not `+0.5`.

## Empirical table

Target variants:

| Variant | Composite | Delta | `cv` | `dag` | `straight_deg` | `crossing_rate` | `angular_deg` | `overlaps` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `58.9995` | `+0.0000` | `0.0802` | `0.5250` | `45.0338` | `0.000208` | `54.1152` | `0` |
| `y *= 1.15` | `59.2315` | `+0.2319` | `0.0963` | `0.5250` | `42.5133` | `0.000208` | `51.1692` | `0` |
| `x *= 0.85` | `59.2295` | `+0.2299` | `0.1009` | `0.5250` | `42.1039` | `0.000208` | `50.5127` | `0` |
| `x *= 1.30, y *= 1.50` | `59.2317` | `+0.2322` | `0.0969` | `0.5250` | `42.4532` | `0.000208` | `51.0727` | `0` |
| vertical flip only | `59.1245` | `+0.1250` | `0.0802` | `0.5300` | `45.0338` | `0.000208` | `54.1152` | `0` |
| **`x *= 0.95, y *= -1.10`** | **`59.3568`** | **`+0.3572`** | `0.0976` | `0.5300` | `42.3902` | `0.000208` | `50.9715` | `0` |

Paired jitter validation for the best candidate, sigma `0.5`, 12 trials:

| Statistic | Delta |
|---|---:|
| mean | `+0.3580` |
| min | `+0.3500` |
| max | `+0.3654` |
| std | `0.0046` |

This passes artifact sanity, but it still fails the sprint success margin. An
independent-noise comparison was noisier (`mean +0.3269`, one negative trial),
which is expected because sigma `0.5` jitter is large enough to move several
near-cutoff straightness contributions independently. The paired check is the
more relevant test for whether the transform itself depends on exact colinear
coordinates.

Protected-win table using the raw transform plus the picker decision:

| Protected graph | N | E | Baseline | Raw candidate | Raw delta | Overlaps | DAG | Picker decision |
|---|---:|---:|---:|---:|---:|---|---|---|
| `recurrent_feedback_cell` | 5 | 6 | `74.889` | `48.712` | `-26.177` | `0 -> 0` | `0.833 -> 0.333` | reject/no-op |
| `parallel_cycles_4x5` | 20 | 20 | `65.356` | `65.356` | `-0.000` | `0 -> 0` | `1.000 -> 1.000` | reject/no-op |
| `hexagonal_lattice_42` | 42 | 53 | `92.067` | `55.128` | `-36.939` | `0 -> 0` | `1.000 -> 0.000` | reject/no-op |
| `triangular_lattice_36` | 36 | 85 | `87.058` | `46.777` | `-40.281` | `0 -> 0` | `1.000 -> 0.000` | reject/no-op |
| `disconnected_encoder_residual` | 9 | 8 | `86.186` | `46.186` | `-40.000` | `0 -> 0` | `1.000 -> 0.000` | reject/no-op |

The protected table is the key safety result. The raw vertical flip is
catastrophic for acyclic DAG wins because it reverses their orientation. This is
not a generally safe primitive. It is only tolerable inside a composite picker
that rejects regressions, and the current target itself is rejected by the
strict `+0.5` margin.

## Gate predicate

Recommended gate for sprint-27: **do not add this candidate**. The target
improvement is real but sub-threshold.

If someone later wants to keep it as a low-priority exploratory picker option,
the minimum safe predicate should be:

```text
candidate = centered transform: x' = 0.95*x, y' = -1.10*y
evaluate only after existing chained polish on the picker's running pos
accept iff:
  composite(candidate) >= composite(current_pos) + 0.5
  overlap_count(candidate) <= overlap_count(current_pos)
  dag_consistency(candidate) >= dag_consistency(current_pos) - 0.001
  graph is cyclic / small-world-like, or candidate is only considered by a
  best-of-polish picker that can no-op on every graph
```

The `dag_consistency` non-regression guard would reject the protected DAG
failures above. It would also reject this exact `small_world_100` candidate if
the graph were not cyclic and if the composite margin were ever relaxed.

## LOC estimate

- Candidate transform helper: about **8-12 LOC**.
- Picker wrapper/scoring call if added near existing chained polish candidates:
  about **25-35 LOC**.
- Regression tests for target rejection plus protected no-op behavior:
  about **35-55 LOC**.
- Total implementation envelope: **70-100 LOC**.

Because the strict gate is missed, my recommendation is **0 production LOC** for
sprint-27.

## Assumptions and concerns

- I used the prompt's default node sizes, `[[40.0, 20.0]] * N`, not measured text
  sizes.
- I treated `dagua.layout(g)` with `LayoutConfig(device="cpu")` as the current
  live state. It reproduced the prompt's rounded score, so this appears to be
  the right surface.
- I did not modify `dagua/`. Scratch work stayed under
  `/tmp/sprint27_small_world_100_codex`; this report is the only requested
  output.
- The candidate improves the target from a `+1.91` win over
  `igraph_sugiyama = 57.09` to about `+2.27`, still a modest win. Strong-win
  territory would require `> 62.09`, and simple affine transforms do not show a
  credible path there.

## Knowledge

`small_world_100` at sprint-27 HEAD is already metric-saturated on crossings,
overlap, angular resolution, and CV. The residual score ceiling for simple
chained polish is governed by a narrow trade-off: anisotropic verticalization
buys straightness, but only until edge-length CV consumes the gain. A global
orientation flip contributes just `+0.125` because the best available DAG
direction changes only one additional edge out of 200.
