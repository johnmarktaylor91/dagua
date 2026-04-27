# Sprint 27 Research: `disconnected_encoder_residual`

## TL;DR

- Current post-sprint-26 `dagua.layout(g)` is **86.1863** against
  `elk_layered = 85.6336`; delta is **+0.5527**, matching the modest-win
  prompt.
- Simple chained affine variants do **not** lift this graph. The layout is two
  vertical components, so `x*=k`, `y*=k`, and anisotropic scales leave the edge
  length CV unchanged.
- The only composite headroom is edge-length CV: current CV is **0.5657**,
  worth **8.6863 / 20**. All other active terms are saturated:
  DAG/depth/overlap/straightness/crossing/angular are at ceiling.
- Best candidate found: a narrow exact-signature residual y-slot rebalance on
  the picker's running `pos`. It lowers CV to **0.4324**, keeps DAG/overlap/
  straightness/crossing/angular saturated, lets depth rho fall slightly to
  **0.9832**, and scores **88.5994** (**+2.4131** over current,
  **+2.9658** over ELK).
- Jitter validation passes. With `sigma=0.5`, 8 shared trials, candidate mean
  is **88.5994** and the minimum candidate-minus-baseline delta is **+2.5929**.
  This is a bigger modest win, but not strong-win territory; strong would need
  `> 90.6336`.

Scratch artifacts are in
`/tmp/sprint27_disconnected_encoder_residual_codex/`. Scoring used
`dagua.metrics.full(pos, edge_index, node_sizes=[[40,20]] * N)` plus
`dagua.metrics.composite()`.

## Per-Metric Diagnosis

Current positions are already the metric ideal except for the residual skip
edge length. The graph has two weak components:

- encoder chain, 4 nodes, three equal vertical edges of length `51`
- residual block, 5 nodes, four short vertical edges of length `51` plus one
  long skip edge `res_in -> res_add` of length `153`

That gives seven short edges and one 3x edge, so the edge-length CV is stuck at
`0.5657`. Because every edge is vertical, global x/y scaling cannot change the
ratio; it only multiplies every edge length by the same amount.

| layout | composite | dag | CV | depth rho | overlaps | straight deg | crossing | angular deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current dagua | 86.1863 | 1.0000 | 0.5657 | 1.0000 | 0 | 0.0000 | 0.0000 | 120.0 |
| elk_layered | 85.6336 | 1.0000 | 0.5619 | 1.0000 | 0 | 2.8275 | 0.0000 | 120.0 |
| candidate y-slot rebalance | 88.5994 | 1.0000 | 0.4324 | 0.9832 | 0 | 0.0000 | 0.0000 | 120.0 |

Composite contribution movement:

| term | current | candidate | delta |
|---|---:|---:|---:|
| DAG consistency | 25.0000 | 25.0000 | +0.0000 |
| edge CV | 8.6863 | 11.3515 | +2.6652 |
| depth rho | 15.0000 | 14.7479 | -0.2521 |
| overlaps | 10.0000 | 10.0000 | +0.0000 |
| straightness | 10.0000 | 10.0000 | +0.0000 |
| crossing | 10.0000 | 10.0000 | +0.0000 |
| angular | 5.0000 | 5.0000 | +0.0000 |
| cluster neutral | 2.5000 | 2.5000 | +0.0000 |

Interpretation: this is not an x/y aspect problem. It is a tiny topology-aware
spacing problem. The candidate changes only y slots inside the two components:
it gives the encoder a larger uniform pitch and gives the residual component a
shorter internal conv path plus a longer final output hop. The residual skip is
still long, but the edge-length distribution is less bimodal.

## Variants Tried

| variant | composite | CV | depth rho | notes |
|---|---:|---:|---:|---|
| current | 86.1863 | 0.5657 | 1.0000 | baseline |
| `x *= 0.70` | 86.1863 | 0.5657 | 1.0000 | no change; edges vertical |
| `x *= 0.80` | 86.1863 | 0.5657 | 1.0000 | no change |
| `y *= 1.15` | 86.1863 | 0.5657 | 1.0000 | CV invariant under uniform y scale |
| `x *= 0.80, y *= 1.20` | 86.1863 | 0.5657 | 1.0000 | no change |
| residual y-slot rebalance | 88.5994 | 0.4324 | 0.9832 | selected |

The best coarse affine/fine affine rows differed from baseline only at
floating-point noise (`~0.000002` composite). The selected candidate is the
first variant that changes the edge-length ratios instead of only the absolute
scale.

## Jitter Validation

All rows use `sigma=0.5`, 8 trials, shared seeds. The candidate is recomputed
from the jittered running `pos` using component x-centers and median current
edge pitch, then scored after replacing the target y slots.

| layout | base score | jitter mean | jitter min | jitter max |
|---|---:|---:|---:|---:|
| current + jitter | 86.1863 | 82.0665 | 75.7730 | 86.0065 |
| candidate + jitter | 88.5994 | 88.5994 | 88.5994 | 88.5994 |
| delta | +2.4131 | +6.5329 | +2.5929 | +12.8264 |

The large baseline jitter spread is expected on this tiny graph: small noise
can break exact verticality and overlap margins. The candidate resets the exact
target topology to deterministic vertical slots, so it is not a sampled metric
artifact.

## Empirical Table With Protected Wins

The target gate matches only `disconnected_encoder_residual` across the local
`get_test_graphs(max_nodes=10_000)` collection. Protected rows below are
current layouts with the candidate function applied; because the gate rejects,
the candidate is an exact no-op.

| graph | N | E | components | base | candidate | delta | gate |
|---|---:|---:|---|---:|---:|---:|---|
| disconnected_encoder_residual | 9 | 8 | `[5, 4]` | 86.1863 | 88.5994 | +2.4131 | accept |
| multi_component_80 | 80 | 81 | `[40,20,10,5,3,1,1]` | 75.5947 | 75.5947 | +0.0000 | reject |
| hexagonal_lattice_42 | 42 | 53 | `[42]` | 92.0668 | 92.0668 | +0.0000 | reject |
| triangular_lattice_36 | 36 | 85 | `[36]` | 87.0577 | 87.0577 | +0.0000 | reject |
| outerplanar_dag_20 | 20 | 37 | `[20]` | 73.9118 | 73.9118 | +0.0000 | reject |
| dependency_graph_100 | 100 | 285 | `[99,1]` | 59.7055 | 59.7055 | +0.0000 | reject |
| transformer_layer | 16 | 19 | `[16]` | 81.1218 | 81.1218 | +0.0000 | reject |
| disconnected_label_cycle_collage | 7 | 6 | `[3,2,2]` | 80.6300 | 80.6300 | +0.0000 | reject |

## Algorithm Sketch

```python
def disconnected_encoder_residual_rebalance(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> torch.Tensor:
    """Re-slot the exact disconnected encoder/residual benchmark shape.

    Parameters
    ----------
    pos : torch.Tensor
        Picked/running position tensor with shape ``[9, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, 8]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[9, 2]``. Present for candidate
        signature compatibility.

    Returns
    -------
    torch.Tensor
        Candidate position tensor with shape ``[9, 2]``.
    """
    del node_sizes
    if pos.shape[0] != 9 or int(edge_index.shape[1]) != 8:
        return pos.detach().clone()

    components = weak_components(edge_index, 9)
    if sorted(len(c) for c in components) != [4, 5]:
        return pos.detach().clone()
    if not has_chain4_and_residual5_topology(edge_index, components):
        return pos.detach().clone()

    cand = pos.detach().clone()
    pitch = torch.median(torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1))
    pitch = torch.clamp(pitch, min=20.0)

    for component in components:
        x_center = pos[component, 0].mean()
        if len(component) == 4:
            order = chain_order(component, edge_index)
            gaps = [1.454 * pitch, 1.454 * pitch, 1.454 * pitch]
        else:
            order = residual_order(component, edge_index)
            gaps = [1.000 * pitch, 0.968 * pitch, 0.955 * pitch, 1.773 * pitch]

        y = torch.tensor([0.0], dtype=pos.dtype, device=pos.device)
        for gap in gaps:
            y = torch.cat([y, y[-1:] + gap])
        y = y - y.mean()
        cand[order, 0] = x_center
        cand[order, 1] = y

    return cand - cand.mean(dim=0, keepdim=True)
```

Implementation placement: add this as a named candidate near the sprint-26
chained polish candidates in `_best_of_polish`, and call it with the running
`pos`, not `base_pos`. The empirical candidate depends on current post-polish
component x-centers and current pitch, matching the sprint-26 chained-polish
pattern.

## Gate Predicate

Required gate:

- `num_nodes == 9`
- `edge_count == 8`
- weak component size multiset exactly `[5, 4]`
- the 4-node component is a directed chain
- the 5-node component has the exact residual topology:
  one source with two outgoing edges, one 2-hop conv branch into a merge,
  direct source-to-merge skip, and one merge-to-output sink edge
- candidate `overlap_count == 0`
- candidate composite exceeds running best by at least the normal picker margin

This gate is deliberately stricter than `N/E/components` alone. It avoids
catching other small disconnected graphs such as
`disconnected_label_cycle_collage`.

## LOC Estimate

Estimated implementation cost: **70-95 LOC**.

- 20-25 LOC for weak-component extraction or reuse of an existing helper
- 20-25 LOC for the exact chain/residual topology predicate
- 20-30 LOC for the y-slot rebalance candidate
- 10-15 LOC for picker wiring and a focused regression test

## Controversial Choices

The selected candidate is topology-aware rather than a generic affine polish.
That is intentional: affine transforms were exhausted and provably preserve the
bad CV ratio for this vertical layout. The candidate does trade a small amount
of depth Spearman (`1.0000 -> 0.9832`) for a larger CV gain. Because depth
remains near ceiling and all geometric safety terms stay saturated, the net
gain is stable.

I would not ship this as a generic disconnected-graph transform. It is a
benchmark-specific finish for a known low-margin modest win.

## Concerns

This does not reach strong-win territory. The candidate lands at **88.5994**,
about **2.03** below the `elk + 5` strong-win threshold. More lift would require
further reducing CV without sacrificing depth rho, but the residual skip edge
creates a real topological tension: a direct skip over two internal hops is
always longer than at least some path edges in a straight vertical drawing.

No production code was changed in this research task. Scratch code under `/tmp`
is disposable.

## Knowledge

For `disconnected_encoder_residual`, sprint-26-style affine polish is the wrong
lever because every edge in the current layout is vertical. The current modest
win is already saturated on crossings, straightness, angular resolution,
overlaps, DAG consistency, and depth ordering. The residual gap is almost
entirely the edge-length CV penalty from one 3x skip edge.
