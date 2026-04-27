# Sprint 22 Area B: Conformal / Harmonic Embedding for True Planar Lattices

## TL;DR

- Biggest call: do **not** implement conformal/Tutte/harmonic embedding as a broad lattice finisher. It misses the primary hex target by a lot at current HEAD `c821eb6`.
- The prompt's June 2025 paper lead is useful but mischaracterized for this purpose. `arXiv:2506.20541` is about conformal rigidity via Laplacian eigenvalue extremality and edge-isometric spectral embeddings, not a directed Tutte or harmonic refinement algorithm.
- A depth-preserving spectral x-coordinate candidate helps `triangular_lattice_36` by `+1.49` composite, but regresses `hexagonal_lattice_42` by `-12.91`, `sierpinski_42` by `-15.06`, `planar_60` by `-9.47`, and `outerplanar_dag_20` by `-9.58`.
- The only large positive non-triangular result, `parallel_cycles_4x5 +5.39`, is a degenerate metric exploit: it creates `190` node overlaps. It must be rejected by any visual or guardrail gate.
- The viable implementation, if any, is a **picker-only candidate** named something like `spectral_x_depth_y` gated to triangular-lattice-like directed planar graphs with `overlap_count == 0`, `dag_consistency >= baseline`, and live composite improvement. It is not the solution for hex.
- For `hexagonal_lattice_42`, Area A's dot-position mimic is the better bet. Harmonic smoothing either collapses rows into overlaps or worsens edge-length CV; it does not discover dot's non-uniform row/column spacing.

## Sources Consulted

- Gouveia, Steinerberger, Thomas, "Conformal Rigidity and Spectral Embeddings of Graphs", arXiv:2506.20541, submitted June 25, 2025. The abstract defines the problem as conformal rigidity of graph Laplacians under edge-weight changes and states the paper's core tool is edge-isometric spectral embeddings: <https://arxiv.org/abs/2506.20541>.
- Steinerberger and Thomas, "Conformally rigid graphs", arXiv:2402.11758. This is the prior paper introducing the rigidity notion used by the 2025 paper: <https://arxiv.org/abs/2402.11758>.
- Tutte embedding theorem summary via "Tutte Embeddings of Tetrahedral Meshes", arXiv:2212.00452. This source restates the classical planar result: a 3-connected planar graph with a convex outer face and interior vertices as convex combinations of neighbors is embedded in the plane: <https://arxiv.org/abs/2212.00452>.
- Gortler, Gotsman, Thurston, "Discrete One-Forms on Meshes". Relevant for the harmonic/one-form view of Tutte embeddings and why convex-boundary harmonic systems are the right classical baseline, not the 2025 conformal-rigidity paper: <https://www.cs.harvard.edu/~sjg/papers/tutte.pdf>.

## Paper Findings

The June 2025 paper is mathematically relevant to "uniform edge length" but not operationally the way the sprint prompt hoped.

The paper defines a finite connected undirected graph as lower conformally rigid when uniform edge weights maximize the second eigenvalue of the weighted graph Laplacian under normalized nonnegative edge weights, and upper conformally rigid when uniform weights minimize the largest eigenvalue. It then connects those rigidity certificates to spectral embeddings where all graph edges have one fixed Euclidean length.

The key implementable idea for Dagua is therefore not "solve a Tutte system, then refine it conformally." It is:

1. Build an undirected Laplacian.
2. Compute a relevant eigenspace.
3. Look for a spectral embedding whose edge lengths are constant, or close to constant.
4. Use that embedding as geometry.

That clashes with Dagua's current composite on directed benchmark graphs:

- The spectral objective is undirected. Dagua's live score spends `25` points on `dag_consistency`, `15` on `depth_spearman_rho`, and `10` on vertical edge straightness for top-to-bottom layouts.
- A pure spectral or Tutte embedding can improve edge-length CV and crossings while throwing away the y-depth structure. On this score surface, that is usually a losing trade.
- The paper's conformal rigidity applies most cleanly to high-symmetry undirected graph families. Dagua's lattice fixtures are directed acyclic patches with node labels/sizes and a preferred y direction. They are not the closed, vertex-transitive, edge-transitive, or distance-regular objects where the rigidity theory is strongest.

I tested both plausible adaptations:

1. **Tutte-with-depth-anchor:** use harmonic/Tutte logic only for x, while y is the longest-path depth mapped to a monotone vertical coordinate.
2. **Spectral-with-depth-anchor:** use an undirected Laplacian eigenvector as x, while y remains depth.

I also tested an x-only variant that preserves Dagua's baseline y values, because that is the most conservative directed adaptation.

## Algorithm Sketch

The best candidate from the experiments is the spectral x-coordinate with depth y. It is not a broad recommendation, but it is the only non-degenerate positive result on a target graph.

Complete working pseudocode:

```python
def spectral_x_depth_y_layout(
    edge_index: Tensor,          # [2, E], directed Dagua edges
    baseline_pos: Tensor,        # [N, 2], current Dagua layout
    node_sizes: Tensor,          # [N, 2]
    graph_name: str,
) -> Tensor:
    """Return a depth-preserving spectral-x candidate.

    The candidate is intended for picker scoring, not direct replacement.
    """
    n = baseline_pos.shape[0]

    # 1. Directed y anchor: keep Dagua's preferred topological order.
    depth = longest_path_layering(edge_index, n)  # list[int] or Tensor[N]
    depth = as_float_array(depth)

    # Use baseline layer pitch instead of an arbitrary unit pitch so node sizes
    # remain roughly in the same coordinate system.
    baseline_y = baseline_pos[:, 1].cpu().numpy()
    layer_means = [mean(baseline_y[depth == d]) for d in sorted(unique(depth))]
    pitch = median(abs(diff(sorted(layer_means))))
    if pitch <= 1e-6:
        pitch = max(std(baseline_y), 1.0)
    y = depth * pitch

    # 2. Undirected Laplacian for spectral x.
    undirected_edges = set()
    for src, tgt in edge_index.T:
        if src != tgt:
            undirected_edges.add((min(src, tgt), max(src, tgt)))

    L = sparse_laplacian(n, undirected_edges)

    # 3. Compute low-frequency eigenvectors.
    # Skip the all-ones vector and choose the eigenvector least correlated
    # with y, so x carries lateral variation rather than duplicating depth.
    eigenvalues, eigenvectors = eigsh(L + 1e-9 * I, k=min(4, n - 1), which="SM")
    y_centered = y - mean(y)
    best_x = None
    best_decorrelation = -inf
    for idx in argsort(eigenvalues)[1:]:
        v = eigenvectors[:, idx]
        v = v - mean(v)
        corr = abs(dot(v, y_centered)) / max(norm(v) * norm(y_centered), 1e-9)
        decorrelation = 1.0 - corr
        if decorrelation > best_decorrelation:
            best_decorrelation = decorrelation
            best_x = v

    # 4. Normalize, then scan a small deterministic scale grid.
    x0 = (best_x - mean(best_x)) / max(std(best_x), 1e-9)
    y0 = (y - mean(y)) / max(std(y), 1e-9)

    best_pos = baseline_pos
    best_score = composite(full(baseline_pos, edge_index, node_sizes=node_sizes))
    baseline_metrics = full(baseline_pos, edge_index, node_sizes=node_sizes)

    for x_scale in [20, 35, 50, 70, 95, 130, 180]:
        for y_scale in [20, 35, 50, 70, 95, 130, 180]:
            pos = stack([x0 * x_scale, y0 * y_scale], axis=1)
            metrics = full(pos, edge_index, node_sizes=node_sizes)

            # Hard visual and directed guards. These reject the parallel-cycle
            # metric exploit and most harmonic collapses.
            if metrics["overlap_count"] != 0:
                continue
            if metrics["dag_consistency"] < baseline_metrics["dag_consistency"]:
                continue
            if metrics["crossing_rate"] > baseline_metrics["crossing_rate"] + 1e-9:
                continue
            if metrics["depth_spearman_rho"] < baseline_metrics["depth_spearman_rho"] - 0.01:
                continue

            score = composite(metrics)
            if score > best_score:
                best_score = score
                best_pos = pos

    return as_tensor(best_pos)
```

The harmonic x-coordinate variant is the same scaffold except step 3 is replaced with a sparse Dirichlet solve:

```python
def harmonic_x_with_outer_face(
    edge_index: Tensor,
    baseline_pos: Tensor,
    outer_face: list[int],
    ridge: float = 1e-4,
) -> ndarray:
    """Solve L_FF x_F = -L_FB x_B with baseline-ranked outer x anchors."""
    n = baseline_pos.shape[0]
    L = sparse_laplacian(n, undirected_edges(edge_index))

    x0 = baseline_pos[:, 0].cpu().numpy()
    x0 = (x0 - mean(x0)) / max(std(x0), 1e-9)

    boundary = sorted(set(outer_face))
    free = [v for v in range(n) if v not in boundary]
    if len(boundary) < 3 or not free:
        return x0

    L_FF = L[free, free] + ridge * I
    L_FB = L[free, boundary]
    rhs = -L_FB @ x0[boundary] + ridge * x0[free]

    x = copy(x0)
    x[free] = sparse_solve(L_FF, rhs)
    return x
```

Classical 2D Tutte was also tested by putting the NetworkX planar outer face on a regular polygon and solving the same Laplacian system for both coordinates. It was never the best safe candidate because it destroys directed y-order on these fixtures.

## Empirical Validation

All experiments were implemented in `/tmp/dagua_sprint22_b/conformal_experiment.py`. The script imports the current repo, runs Dagua baseline layouts at `LayoutConfig(seed=0)`, then scores all candidates using:

```python
torch.manual_seed(0)
score = composite(full(pos, graph.edge_index, node_sizes=graph.node_sizes))
```

No `dagua/` source files were modified. The JSON result artifact is `/tmp/dagua_sprint22_b/conformal_results.json`.

### Depth-Anchored Candidates

This is the main adaptation requested in the prompt: x from harmonic or spectral geometry, y from global directed depth.

| Graph | Baseline | Best depth-anchored candidate | Candidate score | Delta | Notes |
|---|---:|---|---:|---:|---|
| `hexagonal_lattice_42` | 88.3545 | `harmonic_x_depth_y[x=20,y=180]` | 75.4454 | -12.9091 | Regression. `overlap_count=48`, CV worsens `0.420 -> 0.563`. |
| `triangular_lattice_36` | 85.1774 | `spectral_x_depth_y[x=180,y=180]` | 86.6671 | +1.4897 | Real positive. Crossings drop to zero, CV improves `0.227 -> 0.188`, no overlaps. |
| `sierpinski_42` | 85.4255 | `harmonic_x_depth_y[x=35,y=50]` | 70.3659 | -15.0596 | Regression. `overlap_count=194`, CV worsens `0.206 -> 0.430`. |
| `parallel_cycles_4x5` | 62.1103 | `harmonic_x_depth_y[x=20,y=20]` | 67.5000 | +5.3897 | Degenerate. `overlap_count=190`; reject despite score gain. |
| `planar_60` | 80.0891 | `spectral_x_depth_y[x=180,y=130]` | 70.6159 | -9.4731 | Regression. Better CV but `overlap_count=27`, straightness collapses. |
| `outerplanar_dag_20` | 72.4174 | `harmonic_x_depth_y[x=20,y=180]` | 62.8398 | -9.5777 | Regression. `overlap_count=19`; CV remains above 1.0. |

### X-Only Conservative Candidates

This preserves baseline y order and tests whether the idea can act as a safer x-polish.

| Graph | Baseline | Best x-only candidate | Candidate score | Delta | Notes |
|---|---:|---|---:|---:|---|
| `hexagonal_lattice_42` | 88.3545 | `base_y_harmonic_x[x=20,y=180]` | 78.1401 | -10.2145 | Still bad. Slight CV gain is destroyed by `overlap_count=57`. |
| `triangular_lattice_36` | 85.1774 | `base_y_spectral_x[x=180,y=180]` | 86.2360 | +1.0586 | Safe but weaker than full depth-anchored spectral x. |
| `sierpinski_42` | 85.4255 | `base_y_blend_harm25[x=180,y=180]` | 84.3838 | -1.0418 | Near miss, but still a protected win regression. |
| `parallel_cycles_4x5` | 62.1103 | `base_y_harmonic_x[x=95,y=130]` | 62.1103 | +0.0000 | No meaningful effect when y is preserved. |
| `planar_60` | 80.0891 | `base_y_harmonic_x[x=20,y=35]` | 70.0891 | -10.0000 | Regression entirely from overlaps. |

### Metric Breakdown on Important Cases

`triangular_lattice_36`, best candidate:

- `dag_consistency`: `1.000 -> 1.000`
- `depth_spearman_rho`: `0.993 -> 1.000`
- `edge_length_cv`: `0.2265 -> 0.1879`
- `crossing_rate`: `0.00882 -> 0.00000`
- `edge_straightness_mean_deg`: `29.05 -> 31.84`
- `overlap_count`: `0 -> 0`
- Net: `+1.4897`

`hexagonal_lattice_42`, best depth-anchored candidate:

- `dag_consistency`: `1.000 -> 1.000`
- `depth_spearman_rho`: `0.995 -> 1.000`
- `edge_length_cv`: `0.4197 -> 0.5628`
- `crossing_rate`: `0.000 -> 0.000`
- `edge_straightness_mean_deg`: `3.07 -> 3.59`
- `overlap_count`: `0 -> 48`
- Net: `-12.9091`

This is decisive for the main target. The method does not even attack the intended hex gap; it worsens CV and creates overlaps.

## Risk / Regression Analysis

The regression surface is specific and severe.

1. **Overlap collapse.** Harmonic x coordinates are smooth by construction. On grid/lattice patches with many same-depth nodes, that smoothing pulls interiors toward a narrow set of x values. The composite gives only a binary 10-point overlap penalty, so a visually invalid layout can still look competitive if CV/dag improve. `parallel_cycles_4x5` is the warning case: `+5.39` composite with `190` overlaps.

2. **Directed metric conflict.** Pure Tutte and spectral embeddings are undirected. They can reduce edge-length CV and crossings, but Dagua scores top-to-bottom hierarchy heavily. The only safe adaptation is to hard-preserve or reconstruct y-depth. That preservation removes much of the conformal/Tutte value and leaves x as a weak one-dimensional smoothing problem.

3. **Hex mismatch.** `hexagonal_lattice_42` needs graphviz-dot-like non-uniform row spacing. Harmonic averaging does the opposite: it regularizes local coordinates and loses the row/column alternation that dot's layer assignment produces.

4. **Protected wins.** `sierpinski_42` is already a Dagua win. Both depth-anchored and x-only variants regress it. Any gate based only on `is_planar && lattice_like` would harm a protected graph.

5. **Planar false positives.** `planar_60` and `outerplanar_dag_20` regress hard. General planarity is not a useful gate for this candidate.

Specific protected graphs to verify if this is ever implemented:

- `hexagonal_lattice_42`: must not run unless the live picker proves improvement; current measured delta is `-12.91`.
- `sierpinski_42`: protected win, current best x-only delta is `-1.04`, depth-anchored delta is `-15.06`.
- `planar_60`: current Dagua score `80.09`; candidate loses roughly `9.5-10.0`.
- `outerplanar_dag_20`: close-loss graph but candidate loses `9.58`; do not gate on outerplanarity.
- `parallel_cycles_4x5`: reject candidates with any overlap; otherwise the metric can accept a collapsed drawing.

## Recommended Gate

Do not add this as an unconditional pipeline stage.

If implementation proceeds, add it only as a candidate inside the existing picker, with hard guards:

```python
def allow_spectral_x_depth_candidate(structure, graph, baseline_metrics, candidate_metrics):
    if graph.num_nodes < 12:
        return False
    if "lattice" not in graph.tags and "lattice_like" not in structure.topology_tags:
        return False
    if not structure.is_planar:
        return False

    # The empirical positive case is triangular, not honeycomb or Sierpinski.
    if not looks_like_directed_triangular_lattice(graph):
        return False

    if candidate_metrics["overlap_count"] != 0:
        return False
    if candidate_metrics["dag_consistency"] < baseline_metrics["dag_consistency"]:
        return False
    if candidate_metrics["depth_spearman_rho"] < baseline_metrics["depth_spearman_rho"] - 0.01:
        return False
    if candidate_metrics["crossing_rate"] > baseline_metrics["crossing_rate"] + 1e-9:
        return False

    return composite(candidate_metrics) > composite(baseline_metrics) + 0.25
```

The missing piece is `looks_like_directed_triangular_lattice`. A conservative first version:

- planar
- `N >= 25`
- average undirected degree between `3.8` and `5.5`
- many directed edges are one of three local rank moves: same row, next row same column, next row next column, inferred from integer-ish depth plus within-layer x order
- no clusters
- baseline `edge_length_cv < 0.35` so the graph is already regular enough that x spectral finishing is plausible

This gate intentionally excludes `hexagonal_lattice_42` (average degree too low / honeycomb degree-3) and `sierpinski_42` (fractal/holes and protected existing win).

## Implementation Order

1. **Prioritize Area A dot-lattice mimic for `hexagonal_lattice_42`.** Area B does not close hex. Do not spend implementation time here before the dot mimic is evaluated.
2. **If triangular remains a loss after Area A, add `spectral_x_depth_y` as a picker candidate only.** Keep it in a small function near existing polish candidate generation, not as a new primary pipeline.
3. **Add strict guards before scoring:** `overlap_count == 0`, no crossing-rate regression, no dag/depth regression beyond tiny tolerance.
4. **Run targeted verification:** `triangular_lattice_36`, `hexagonal_lattice_42`, `sierpinski_42`, `parallel_cycles_4x5`, `planar_60`, `outerplanar_dag_20`.
5. **Run final benchmark picker evaluation.** Accept only if triangular gains without losing any protected graph. Based on the measurements here, expected suite impact is tiny: roughly `+1.0 to +1.5` on one graph if safely gated.

## Final Recommendation

Area B is not the big algorithmic bet that closes the remaining lattice bucket. The 2025 conformal-rigidity paper is a spectral rigidity/certificate paper, not a directed planar layout recipe. The adaptation that respects Dagua's score can recover `+1.49` on `triangular_lattice_36`, but it fails exactly where we needed it most: `hexagonal_lattice_42`.

Treat this as a narrow triangular-lattice candidate behind the picker, or skip it until Area A is exhausted. Do not ship a general conformal/Tutte/harmonic planar finisher.
