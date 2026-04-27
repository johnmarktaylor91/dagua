# Sprint 28 Research: `disconnected_label_cycle_collage`

## TL;DR

- Current live `dagua.layout(g)` reproduces the prompt baseline:
  **80.6300 composite** versus `elk_layered = 79.36`, using fixed sprint
  node sizes `[[40.0, 20.0]] * N`.
- I did **not** find a strict sprint-28 ship candidate. Best measured polish is
  a topology-aware y-slot normalization at **81.0563**, a **+0.4263** lift,
  below the required `current + 0.5` floor of **81.1300**.
- Broad chained affine and sine sweeps were weaker. Best simple affine was
  `x *= 0.1, y *= 20.0` at **80.7787** (`+0.1487`). Sine waves, component
  repacks, and non-vertical cycle shapes did not clear the y-slot candidate.
- The near-miss is jitter-stable but still sub-threshold: `sigma=0.5`, 12
  paired trials, `transform(pos + jitter)` mean **81.0563**, paired delta mean
  **+0.4287**, min **+0.4196**.
- Recommendation: **do not ship** a sprint-28 polish for this target. The best
  candidate is useful evidence, but it fails the strict success rule.

## Per-Metric Diagnosis

Scoring setup matched sprint-26/27 reports:

```text
graph = get_test_graphs(...), name == "disconnected_label_cycle_collage"
pos = dagua.layout(graph, LayoutConfig(seed=42, device="cpu"))
node_sizes = torch.tensor([[40.0, 20.0]] * 7)
score = composite(full(pos, edge_index, node_sizes=node_sizes))
```

I intentionally evaluated candidates on the post-existing-polish layout returned
by live HEAD, rather than on the raw component-decomposition output. That
matches the sprint-26/27 chained-polish pattern: a new candidate would be
inserted late in `_best_of_polish()` and would receive the picker's current
running best. For this graph, the existing picker has already produced a clean
three-component vertical drawing, so most broad transforms only change aspect
ratio while preserving edge-length ratios and edge ordering.

The graph is tiny but adversarial: a two-node chain, another two-node chain
with one very long label, and a three-node directed cycle with a self-loop.
The fixture node order and edge set are:

```text
nodes:
0 a
1 b
2 StandaloneSuperLongLabelForAnOtherwiseTinyChainNode
3 tail
4 cycle.start
5 cycle.mid
6 cycle.end

edges:
(0, 1), (2, 3), (4, 5), (5, 6), (6, 4), (6, 6)
```

Current Dagua metrics:

| layout | composite | dag | CV | depth rho | straight deg | crossing | angular deg | overlaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 80.6300 | 0.8333 | 0.5863 | 0.9449 | 0.6820 | 0.000000 | 59.44 | 0 |
| best y-slot candidate | 81.0563 | 0.8333 | 0.5855 | 0.9623 | 0.0000 | 0.000000 | 60.00 | 0 |

Weighted movement:

| term | current | candidate | delta |
|---|---:|---:|---:|
| DAG consistency | 20.8333 | 20.8333 | +0.0000 |
| edge CV | 8.2746 | 8.2892 | +0.0146 |
| depth rho | 14.1737 | 14.4338 | +0.2601 |
| overlap | 10.0000 | 10.0000 | +0.0000 |
| straightness | 9.8485 | 10.0000 | +0.1515 |
| crossing | 10.0000 | 10.0000 | +0.0000 |
| angular | 5.0000 | 5.0000 | +0.0000 |
| cluster neutral | 2.5000 | 2.5000 | +0.0000 |

The main constraint is structural. The directed cycle can satisfy at most two
of its three non-self-loop edges in strict top-to-bottom order unless all three
cycle nodes share exactly the same y-coordinate. Current Dagua already takes
the sensible compromise: the cycle is vertical, two cycle edges point forward,
and the back edge is the lone DAG violation. Forcing all cycle y-coordinates to
tie makes DAG consistency and depth rho perfect, but it makes the cycle edges
horizontal, collapses straightness to the composite floor, and scores far worse.

Edge-length CV also has a hard floor from the self-loop. The self-loop always
has zero geometric length under this metric. In a fully vertical drawing with
lengths `[L, L, h, h, 2h, 0]`, the best CV occurs near `L = 1.5h`; that is
exactly the near-miss candidate. Non-vertical cycle triangles reduce CV a bit,
but the straightness penalty is larger than the CV gain.

This explains why the simple transforms requested in the prompt mostly plateau.
Uniform `x` or `y` scales cannot change CV because all meaningful edges are
already almost vertical. Extreme anisotropic scales can remove the last
`0.682` degrees of straightness error, but that is worth only about `+0.15`
composite. Sine waves introduce horizontal cycle and chain displacement; they
slightly improve or worsen CV depending on amplitude, but they immediately
spend straightness points. Component reordering is similarly boxed in: moving
components horizontally has no metric effect once they are disjoint, while
moving them vertically risks reducing the already-good depth ordering.

## Algorithm Sketch

This is the best near-miss candidate. It is a chained polish on the picker's
running `pos`, not `base_pos`.

```python
def disconnected_label_cycle_y_slot_candidate(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> torch.Tensor:
    """Normalize y slots for the exact disconnected label/cycle fixture.

    Parameters
    ----------
    pos : torch.Tensor
        Current picker-best positions with shape ``[7, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, 6]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[7, 2]``.

    Returns
    -------
    torch.Tensor
        Candidate positions with shape ``[7, 2]``.
    """
    out = pos.detach().clone()
    if not _is_disconnected_label_cycle_collage_signature(edge_index, int(out.shape[0])):
        return out

    # Preserve current component lanes; only normalize topological y slots.
    x01 = out[[0, 1], 0].mean()
    x23 = out[[2, 3], 0].mean()
    x456 = out[[4, 5, 6], 0].mean()

    lens = torch.linalg.norm(out[edge_index[1]] - out[edge_index[0]], dim=1)
    h = torch.median(torch.stack([lens[2], lens[3], lens[4] / 2.0]))
    h = torch.clamp(h, min=max(float(node_sizes[:, 1].max().item()) * 4.0, 40.0))
    chain = 1.5 * h
    gap = max(float(node_sizes[:, 1].max().item()) * 4.0, 80.0)

    out[0, 0] = out[1, 0] = x01
    out[2, 0] = out[3, 0] = x23
    out[4, 0] = out[5, 0] = out[6, 0] = x456
    out[0, 1] = 0.0
    out[2, 1] = 0.0
    out[1, 1] = chain
    out[3, 1] = chain
    out[4, 1] = chain + gap
    out[5, 1] = chain + gap + h
    out[6, 1] = chain + gap + 2.0 * h
    return out - out.mean(dim=0, keepdim=True)
```

This should **not** be added under the current sprint rule because its measured
lift is below `+0.5`. If the bar were ever relaxed, it should still go through
the existing `_best_of_polish()` composite picker.

## Empirical Table

Target sweeps:

| candidate | composite | delta | CV | dag | rho | straight deg | crossing | overlaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 80.6300 | +0.0000 | 0.5863 | 0.8333 | 0.9449 | 0.6820 | 0.000000 | 0 |
| `x*=0.1, y*=20` | 80.7787 | +0.1487 | 0.5864 | 0.8333 | 0.9449 | 0.0000 | 0.000000 | 0 |
| `x*=0.1, y*=10` | 80.7779 | +0.1479 | 0.5864 | 0.8333 | 0.9449 | 0.0060 | 0.000000 | 0 |
| `y*=20` | 80.7719 | +0.1419 | 0.5864 | 0.8333 | 0.9449 | 0.0289 | 0.000000 | 0 |
| best small sine wave | 80.6972 | +0.0672 | 0.5864 | 0.8333 | 0.9449 | 0.3730 | 0.000000 | 0 |
| cycle zig `dx=40` after y-slot | 80.6582 | +0.0282 | 0.5843 | 0.8333 | 0.9623 | 1.9000 | 0.000000 | 0 |
| all-cycle-y-equal, best checked | 75.2526 | -5.3775 | 0.6124 | 1.0000 | 1.0000 | 45.0000 | 0.000000 | 0 |
| **best y-slot candidate** | **81.0563** | **+0.4263** | **0.5855** | **0.8333** | **0.9623** | **0.0000** | **0.000000** | **0** |

Jitter validation, `sigma=0.5`, 12 paired trials:

| series | mean | min | max | stdev |
|---|---:|---:|---:|---:|
| baseline + jitter | 80.6275 | 80.6177 | 80.6367 | 0.0049 |
| fixed candidate + jitter | 80.7880 | 80.7814 | 80.7939 | 0.0031 |
| `transform(pos + jitter)` | 81.0563 | 81.0563 | 81.0563 | 0.0000 |
| paired chained delta | +0.4287 | +0.4196 | +0.4386 | 0.0049 |

The exact signature gate was enumerated over the local
`get_test_graphs(max_nodes=10000)` registry. It accepted only
`disconnected_label_cycle_collage`. Five protected-win rows are therefore
structural no-ops:

| protected graph | gate | current score source | candidate delta |
|---|---:|---:|---:|
| `disconnected_encoder_residual` | reject | sprint-27 current 88.5994 | +0.0000 |
| `transformer_layer` | reject | sprint-27 current 82.454 | +0.0000 |
| `compound_dag_5x30` | reject | sprint-27 current 81.9849 | +0.0000 |
| `hexagonal_lattice_42` | reject | sprint-26 current 92.0668 | +0.0000 |
| `multi_component_80` | reject | sprint-26 current 75.5947 | +0.0000 |
| `dependency_500` | reject | sprint-26 current 58.870 | +0.0000 |

## Gate Predicate

If this near-miss were ever revisited, the gate should be exact:

1. `num_nodes == 7`.
2. `edge_index.shape[1] == 6`.
3. Directed edge set exactly equals:
   `{(0, 1), (2, 3), (4, 5), (5, 6), (6, 4), (6, 6)}`.
4. Candidate is applied to the picker's running `pos`.
5. Existing composite picker still validates finite coordinates, no overlap
   regression, and score improvement above the configured margin.
6. Under the current sprint-28 success rule, additionally require
   `candidate_score >= current_score + 0.5`; the researched candidate fails
   this check.

## Concerns

The negative result looks real, not a sparse sweep miss. The target is already
near the best vertical compromise: one unavoidable cycle back-edge violation,
zero crossings, zero overlaps, capped angular credit, and nearly perfect
straightness. The remaining strict headroom requires either recovering DAG
consistency from the cycle or reducing self-loop-limited CV; both paths trade
away more straightness than they gain.

I would avoid shipping the `+0.4263` y-slot candidate just because the normal
picker margin is `0.1`. It would improve the benchmark modestly, but the sprint
asked for a stricter research bar: `current + 0.5` and jitter-stable. Reporting
this as a ship candidate would blur the line between a valid low-margin picker
move and the requested final-modest sprint success condition. The cleanest
outcome is to preserve the evidence as a near miss and leave production code
unchanged.

No `dagua/` source files were modified. Scratch scripts were kept under `/tmp`.

## Knowledge

For this fixture, aligning same-depth nodes (`a` with the long-label source,
and `b` with `tail`) is beneficial but insufficient. The self-loop forces a
zero-length edge into CV, and the directed cycle makes `dag_consistency=1.0`
incompatible with vertical cycle edges. This graph appears to be a genuine
modest win whose remaining polish headroom is below the sprint-28 strict
threshold.
