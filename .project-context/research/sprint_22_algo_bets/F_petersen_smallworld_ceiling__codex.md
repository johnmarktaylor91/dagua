# Area F -- petersen_10 + small_world_500 algorithm-ceiling investigation -- codex

## TL;DR

- `petersen_10` is still a loss at sprint-21b HEAD `c821eb6`: current dagua scores **74.6404** vs listed `igraph_sugiyama` **77.36** (`-2.7196`).
- The Petersen branch is not at ceiling: a **spectral-x / current-y candidate picker** scores **78.8288**, flipping Petersen to `+1.4688` over the listed best.
- `small_world_500` is also not at ceiling: a **directed-spine hybrid** scores **57.3368**, `+5.1466` over current dagua and `+3.1868` over listed `elk_layered`.
- Option A, longer graduated stress, is saturated for `small_world_500`; 90 steps gives only **+0.1360**. Option B, capped layers, does not close the gap; its best measured `small_world_500` candidate gives **+0.5085** but remains `-1.4513` vs ELK.
- Option C must be gated. It slightly regresses `small_world_100` (`57.0217`, `-0.1560` vs current dagua), while the current stress route already ties/wins that graph.
- Single implementation recommendation: add a scored rare-graph candidate picker for cyclic small-world graphs, and separately add the N<=12 Petersen spectral-x candidate to the existing polish picker.

## Petersen Status

Mandatory re-verification at HEAD `c821eb6fa027205e294b83b4de5d21d539089a59`:

| graph | candidate | score | vs current default | vs listed best | dag | rho | CV | straight deg | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| petersen_10 | current default | 74.6404 | 0.0000 | -2.7196 | 1.0000 | 0.9387 | 0.2129 | 25.39 | 0.108108 |
| petersen_10 | best Laplacian spectral-x + current-y | 78.8288 | +4.1884 | +1.4688 | 1.0000 | 0.9387 | 0.3734 | 18.65 | 0.040541 |

So the stale sprint-21 note was directionally right that Petersen is solvable, but not because current HEAD already wins. Current HEAD still loses under the canonical seeded scorer. The useful candidate keeps the current y coordinates, so it preserves DAG consistency and depth rank, and replaces only x with a scored Laplacian eigenvector. A pure 2D spectral layout was bad: best measured pure spectral was **61.6945**, mainly because it gives up too much directed structure (`dag=0.7333`, `straight=52.14`).

One important finding: the exact per-layer permutation proposal is a no-op against current default positions. The current Petersen layout has ten unique y values, one node per layer, so there is no within-layer permutation space. The idea is still valid for a Sugiyama-style candidate with multi-node layers, but the lower-risk implementation for this HEAD is spectral-x while retaining the current y.

## Algorithm Sketch

### Petersen N<=12 Spectral-X Picker

Complete working pseudocode:

```python
def petersen_spectral_x_candidates(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score: Callable[[torch.Tensor], float],
) -> torch.Tensor:
    """Return the best N<=12 spectral-x candidate, or the original positions."""
    n = int(pos.shape[0])
    if n > 12 or edge_index.shape[1] == 0:
        return pos

    adjacency = torch.zeros((n, n), dtype=torch.float64)
    for source, target in edge_index.t().tolist():
        if source == target:
            continue
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    degree = torch.diag(adjacency.sum(dim=1))
    eigenvalues, eigenvectors = torch.linalg.eigh(degree - adjacency)

    best = pos
    best_score = score(pos)
    y = pos[:, 1].detach().clone()
    # Skip eigenvector 0, the constant vector. Degenerate eigenspaces are fine:
    # the scorer picks the basis vector/sign that works with the current y.
    for eig_index in range(1, n):
        raw = eigenvectors[:, eig_index].to(dtype=pos.dtype)
        for sign in (-1.0, 1.0):
            x = raw * sign
            x = x - x.mean()
            std = x.std().clamp(min=1e-6)
            x = (x / std) * 80.0
            candidate = torch.stack((x, y), dim=1)
            candidate_score = score(candidate)
            if candidate_score > best_score + 0.5:
                best = candidate
                best_score = candidate_score
    return best
```

This is deliberately not a Petersen-name special case. The gate should be: `N <= 12`, connected, roughly regular (`max_degree - min_degree <= 1`), no clusters/pins/flex, and current candidate still below the best-of-polish score by the normal margin. The score callback should be the existing `composite(full(...))` path, as in `_best_of_polish`.

### Small-World Directed-Spine Hybrid

Complete working pseudocode:

```python
def cyclic_directed_spine_candidate(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    stress_pos: torch.Tensor,
    y_sep: float = 80.0,
    x_scale: float = 5.0,
) -> torch.Tensor:
    """Use FAS/longest-path y with stress-derived x jitter for cyclic graphs."""
    non_self = edge_index[:, edge_index[0] != edge_index[1]]
    acyclic_edges, _ = make_acyclic_robust(non_self, num_nodes)
    layers = longest_path_layering(acyclic_edges, num_nodes)
    if not isinstance(layers, torch.Tensor):
        layers = torch.tensor(layers, dtype=torch.float32)
    layers = layers.to(dtype=torch.float32)

    # Abort unless FAS exposes the degenerate one-node-per-rank structure
    # seen in small_world_500. This avoids hijacking ordinary cyclic graphs.
    if int(torch.unique(layers).numel()) < int(0.8 * num_nodes):
        return stress_pos

    x = stress_pos[:, 0].detach().to(dtype=torch.float32)
    x = x - x.mean()
    x = x / x.std().clamp(min=1e-6)

    candidate = torch.zeros((num_nodes, 2), dtype=torch.float32)
    candidate[:, 0] = x * x_scale
    candidate[:, 1] = layers * y_sep
    return candidate
```

The production version should not blindly replace the stress route. Build both current stress-route output and this directed-spine candidate, then pick by `composite(full(...))` with the established positive margin. If runtime cost is unacceptable, the heuristic fallback is `num_nodes >= 200`, tags/family small-world/cyclic, post-FAS unique-layer ratio >= 0.8, no clusters, no pins.

## Empirical Validation

All measurements came from temporary scripts under `/tmp`, especially `/tmp/sprint22_f_experiments.py`. I used the real graph registry, `LayoutConfig(seed=42)`, `torch.manual_seed(0)` before `full(...)`, and the canonical `composite(full(...))` path.

### Baselines and Pipeline Probes

| graph | candidate | score | vs best | dag | CV | straight | crossing | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_world_100 | current default | 57.1777 | +0.0877 | 0.5000 | 0.1423 | 44.80 | 0.000208 | 0 |
| small_world_100 | layered route (`route_flat_to_stress=False`) | 49.2010 | -7.8890 | 0.9850 | 2.6893 | 41.36 | 0.005729 | 0 |
| small_world_500 | current default | 52.1902 | -1.9598 | 0.5013 | 0.2275 | 45.00 | 0.000489 | 0 |
| small_world_500 | layered route (`route_flat_to_stress=False`) | 47.5018 | -6.6482 | 0.9960 | 2.7510 | 47.74 | 0.001727 | 0 |

The prompt assumption that `small_world_500` "currently uses layered_dag" is stale for this HEAD. Current default routes through the stress escape and is much better than forced layered. The remaining gap is a scorer tradeoff: stress wins CV/crossings; ELK wins directionality/straightness.

### Option A: Graduated Stress Route

| graph | candidate | score | vs current | vs best |
|---|---:|---:|---:|---:|
| small_world_100 | stress 30 eps .01 | 57.1777 | +0.0000 | +0.0877 |
| small_world_100 | stress 60 eps .01 | 57.4150 | +0.2373 | +0.3250 |
| small_world_100 | stress 90 eps .01 | 56.9963 | -0.1813 | -0.0937 |
| small_world_500 | stress 30 eps .01 | 52.1902 | +0.0000 | -1.9598 |
| small_world_500 | stress 60 eps .01 | 52.1902 | -0.0000 | -1.9598 |
| small_world_500 | stress 90 eps .01 | 52.3263 | +0.1360 | -1.8237 |
| small_world_500 | stress 60 eps .03 | 52.1791 | -0.0111 | -1.9709 |

Conclusion: useful only as a tiny polish for `small_world_100`; not enough for `small_world_500`.

### Option B: Per-Layer Cap

Best full-scored finalists after quick screening:

| graph | candidate | score | vs current | vs best |
|---|---:|---:|---:|---:|
| small_world_100 | cap 5, x 100, y 40 | 53.5997 | -3.5780 | -3.4903 |
| small_world_500 | cap 2, x 70, y 140 | 52.6987 | +0.5085 | -1.4513 |
| small_world_500 | cap 8, x 100, y 40 | 51.4009 | -0.7894 | -2.7491 |

Conclusion: cap-based layering is directionally correct for `small_world_500`, but it spends too many points on horizontal same-layer edges and CV. It is not the sprint-22 bet to implement first.

### Option C: Stress X, Layered Y

| graph | candidate | score | vs current | vs best | dag | CV | straight | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_world_100 | y80 x5 | 57.0217 | -0.1560 | -0.0683 | 0.9850 | 4.0036 | 0.25 | 0.000729 |
| small_world_500 | y80 x5 | 57.3368 | +5.1466 | +3.1868 | 0.9960 | 7.8635 | 0.06 | 0.000520 |
| small_world_500 | y40 x5 | 57.3255 | +5.1353 | +3.1755 | 0.9960 | 7.8635 | 0.12 | 0.000520 |

Conclusion: this is the single biggest call. The candidate intentionally sacrifices the 20-point CV term, but it recovers almost all 25 DAG points and all 10 straightness points. At N=500 that wins decisively. At N=100 the current stress route remains slightly better, so do not global-switch.

## Risk / Regression Analysis

The directed-spine hybrid is metric-driven and visually controversial: `small_world_500` becomes a very tall directed spine with small stress-x wiggle. That is close to what the directed benchmark scorer rewards, but it is not a general-purpose undirected small-world drawing. Keep it behind a rare topology gate and a scorer picker.

Protected wins to verify before implementation merge:

- `small_world_100`: must keep the current default, or optionally use stress 60 eps .01 if the picker confirms the +0.237 lift under final code.
- `parallel_cycles_4x5`: cyclic but not small-world; should not be captured by the unique-layer-ratio plus N>=200 gate.
- `recurrent_feedback_cell`: close/tie cyclic graph; should not receive directed-spine unless scored candidate wins.
- `petersen_10`: spectral-x picker helps, directed-spine should not run because N<200 and regular-small gate belongs to the Petersen picker.
- `dependency_500`: large DAG, not cyclic-small-world; should be excluded by cycle/back-edge requirement and family/tags.

For Petersen, the spectral-x picker can regress if applied to arbitrary small DAGs: it may preserve y but worsen CV and angular resolution. The normal +0.5 `composite(full(...))` acceptance margin is enough protection at N<=12 because scoring is cheap.

## Implementation Order

1. Implement the N<=12 spectral-x candidate inside the existing polish candidate set. Gate on small connected regular-ish graphs; score with `composite(full(...))`; accept only if it clears the current margin. First regression test: `petersen_10` should score above 77.36.
2. Add `cyclic_directed_spine_candidate` as a rare candidate in the flat/cyclic stress route, not as a replacement. Reuse existing `make_acyclic_robust`, `longest_path_layering`, and the already-computed stress output.
3. Pick between current stress output and directed-spine output by `composite(full(...))` for `N <= 2000`. If this is too slow, gate by `N >= 200`, cyclic, no clusters/pins, and post-FAS unique-layer ratio >= 0.8.
4. Add focused tests for `small_world_500`, `small_world_100`, and `petersen_10`. The expected acceptance behavior is: Petersen spectral-x selected, `small_world_500` directed-spine selected, `small_world_100` stress default retained unless the longer-stress variant is explicitly added and scored.
5. Run the sprint quality gates plus the final benchmark bucket script. The implementation should flip two current losses without touching the rest of the close-loss table.

## Assumptions

- I treated the listed best competitor scores as the sprint target: `petersen_10=77.36`, `small_world_100=57.09`, `small_world_500=54.15`.
- I used current HEAD behavior as authoritative where the prompt was stale. Specifically, `small_world_500` default is the stress escape route, while forced layered scores only 47.5018.
- I did not modify `dagua/`; all implementation experiments were temporary `/tmp` scripts.
