# Sprint 23 Area D: spectral-x + depth-y

## TL;DR

- **Do not ship** `spectral_x_depth_y` as a Sprint 23 finisher. Against live HEAD `d27fced`, the target `small_world_500` baseline scores **57.40** and the best fixed spectral-depth candidate scores **54.94** (`-2.46`).
- The apparent win in the first fast pass was caused by stale saved Dagua positions under `eval_output/*/positions`. Those tensors scored `small_world_500` at `49.32`; a live native layout at HEAD scores `57.40`.
- Current HEAD has already absorbed the useful cyclic idea through the sprint-22a back-edge relayer / stress route. Spectral-depth mostly recreates that vertical ordering with worse edge-length CV and/or worse straightness.
- The candidate is not safe as a broad picker candidate: it regresses `small_world_100` (`-9.00`), `parallel_cycles_4x5` (`-3.25`), `hexagonal_lattice_42` (`-18.89`), `planar_60` (`-7.37`), `deep_chain_20` (`-6.51`), and `clustered_medium_5x20` (`-9.03`).
- The only artifact-level positive outside stale small-world scores was `dependency_500 +6.44`, but that candidate has **1797 node overlaps**, so any visual/overlap guard must reject it. I did not use that as ship evidence.

Scratch artifacts:

- `/tmp/sprint23_d_codex/spectral_x_depth_y_experiment.py`
- `/tmp/sprint23_d_codex/spectral_x_depth_y_results.json`
- `/tmp/sprint23_d_codex/current_baselines.py`
- `/tmp/sprint23_d_codex/current_baselines.json`

## Algorithm sketch

The probed candidate is intentionally close to the prompt: build an undirected Laplacian, use a low-frequency eigenvector for x, and use directed longest-path depth for y. I measured three layering modes:

- `raw`: `longest_path_layering(edge_index, N)`
- `dfs_forward`: remove DFS back-edges and self-loops, then layer
- `robust_forward`: use `dagua.layout.cycle.make_acyclic_robust`, then layer

The measured fixed candidate used the prompt scale: `x_span = pitch * sqrt(N)`, `y = layer * pitch`, with `pitch = 2 * mean(max(node_width, node_height))`.

```python
def spectral_x_depth_y(edge_index, node_sizes, mode):
    n = node_sizes.shape[0]

    if mode == "raw":
        layered_edges = edge_index
    elif mode == "dfs_forward":
        back = detect_dfs_back_edges(edge_index, n)
        layered_edges = edge_index[:, ~back]
    elif mode == "robust_forward":
        non_self = edge_index[0] != edge_index[1]
        layered_edges, _ = make_acyclic_robust(edge_index[:, non_self], n)

    layers = longest_path_layering(layered_edges, n)
    layers = np.asarray(layers, dtype=float)

    # Symmetric unweighted Laplacian from the original graph.
    pairs = {
        (min(s, t), max(s, t))
        for s, t in edge_index.T.tolist()
        if s != t
    }
    degree = np.zeros(n)
    rows, cols, data = [], [], []
    for s, t in pairs:
        rows += [s, t]
        cols += [t, s]
        data += [-1.0, -1.0]
        degree[s] += 1.0
        degree[t] += 1.0
    rows += list(range(n))
    cols += list(range(n))
    data += degree.tolist()
    L = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Compute low-frequency eigenvectors. The code used eigsh(L + eps*I,
    # k=min(6, n-1), which="SM") and fell back to dense eigh for tiny cases.
    values, vectors = eigsh(L + 1e-9 * scipy.sparse.eye(n), k=min(6, n - 1), which="SM")

    # Pick the first non-constant low-frequency vector, preferring low
    # correlation with y so x does not duplicate depth.
    y0 = layers - layers.mean()
    best_x = None
    best_key = (-1.0, -1.0)
    for idx in np.argsort(values)[1:]:
        v = vectors[:, idx] - vectors[:, idx].mean()
        if np.linalg.norm(v) <= 1e-12:
            continue
        corr = abs(np.dot(v, y0)) / max(np.linalg.norm(v) * np.linalg.norm(y0), 1e-12)
        key = (1.0 - corr, np.var(v))
        if key > best_key:
            best_key = key
            best_x = v

    if best_x is None:
        best_x = np.linspace(-0.5, 0.5, n)

    x_unit = (best_x - best_x.mean()) / max(best_x.max() - best_x.min(), 1e-12)
    pitch = max(2.0 * node_sizes.max(dim=1).values.mean().item(), 1.0)
    x = x_unit * pitch * math.sqrt(n)
    y = (layers - layers.mean()) * pitch
    pos = np.stack([x, y], axis=1)
    return torch.tensor(pos - pos.mean(axis=0, keepdims=True), dtype=torch.float32)
```

I also ran a small scale scan in the scratch script, but the final table below uses the fixed prompt-scale candidate because that is the implementation being evaluated. The scan did not change the ship decision: positives were either stale-baseline artifacts or overlap failures.

## Empirical validation

All candidate scores use `dagua.metrics.full(..., crossing_samples=1_000_000)` and `dagua.metrics.composite`. For current baselines, I ran `dagua.layout.engine.layout(graph, LayoutConfig(seed=0, device="cpu"))` on live HEAD. `dependency_500` exceeded the available local time while current-layout measuring, so the table marks it as artifact-only and treats it as non-evidence because the candidate overlaps are catastrophic.

`rho` is `depth_spearman_rho`; `cross` is sampled crossing rate.

| graph | baseline source | base | best spectral | delta | mode | CV base -> cand | straight base -> cand | dag base -> cand | rho base -> cand | cross base -> cand | overlaps cand |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `small_world_500` | live | 57.40 | 54.94 | -2.46 | robust_forward | 7.863 -> 7.787 | 0.00 -> 10.06 | 0.996 -> 0.996 | NaN -> NaN | 0.0000 -> 0.0024 | 0 |
| `small_world_100` | live | 59.27 | 50.28 | -9.00 | robust_forward | 0.081 -> 3.637 | 44.89 -> 29.36 | 0.535 -> 0.985 | NaN -> NaN | 0.0000 -> 0.0057 | 0 |
| `recurrent_feedback_cell` | live | 74.65 | 72.73 | -1.92 | raw | 0.615 -> 0.530 | 21.65 -> 46.06 | 0.833 -> 0.833 | 0.894 -> 1.000 | 0.0000 -> 0.0000 | 0 |
| `parallel_cycles_4x5` | live | 65.36 | 62.11 | -3.25 | dfs_forward | 0.357 -> 0.769 | 90.00 -> 0.00 | 1.000 -> 0.800 | NaN -> NaN | 0.0000 -> 0.0000 | 0 |
| `hexagonal_lattice_42` | live | 88.19 | 69.30 | -18.89 | raw | 0.166 -> 0.448 | 14.99 -> 32.32 | 1.000 -> 1.000 | 0.823 -> 1.000 | 0.0000 -> 0.0205 | 13 |
| `triangular_lattice_36` | live | 86.61 | 86.45 | -0.16 | raw | 0.189 -> 0.267 | 32.04 -> 23.00 | 1.000 -> 1.000 | 1.000 -> 1.000 | 0.0000 -> 0.0000 | 0 |
| `planar_60` | live | 80.09 | 72.72 | -7.37 | raw | 0.464 -> 0.654 | 0.00 -> 16.36 | 0.923 -> 1.000 | 0.919 -> 1.000 | 0.0000 -> 0.0342 | 0 |
| `deep_chain_20` | live | 97.50 | 90.99 | -6.51 | robust_forward | 0.000 -> 0.065 | 0.00 -> 23.44 | 1.000 -> 1.000 | 1.000 -> 1.000 | 0.0000 -> 0.0000 | 0 |
| `clustered_medium_5x20` | live | 70.05 | 61.02 | -9.03 | dfs_forward | 1.312 -> 1.534 | 25.57 -> 15.41 | 1.000 -> 1.000 | 1.000 -> 1.000 | 0.0168 -> 0.0210 | 8 |
| `dependency_500` | artifact only | 48.21 | 54.65 | +6.44 | raw | 0.789 -> 0.755 | 55.23 -> 21.96 | 1.000 -> 1.000 | 0.994 -> 1.000 | 0.1250 -> 0.0879 | 1797 |

### Interpretation

The target result is clear. `small_world_500` is no longer a close-loss for this candidate to fix at live HEAD; the live native pipeline produces an almost perfectly DAG-consistent, zero-crossing layout under the metric. Spectral-depth cannot improve that surface. It preserves `dag_consistency` but introduces crossings and straightness loss, while edge-length CV remains so high that the CV term contributes no useful points.

`small_world_100` is an even stronger rejection. Robust forward layering raises `dag_consistency` from `0.535` to `0.985`, but it destroys edge-length uniformity (`0.081 -> 3.637`). The composite loss is nearly nine points. This means the candidate optimizes the wrong axis for small-world graphs now that native already has a metric-balanced stress/back-edge route.

The cyclic micrographs reject the idea as a replacement for sprint-22a. `recurrent_feedback_cell` is close but still loses by `1.92`; `parallel_cycles_4x5` loses by `3.25`. The prior Sprint 22 note warned that depth-anchored spectral/harmonic candidates could exploit metrics on parallel cycles; at live HEAD the protected baseline is already better.

For lattices, the old conclusion holds. `triangular_lattice_36` is near neutral (`-0.16`) but no longer a meaningful win. `hexagonal_lattice_42` is a large regression with overlaps. The Fiedler coordinate does not reproduce the integer-grid / network-simplex spacing that dot-like lattice polish needs.

`dependency_500` is the only positive row, but it is not shippable evidence. The candidate has 1797 node overlaps, which would fail any visual guard and lose the binary no-overlap term. I attempted to measure the live current baseline, but the native layout run exceeded 11 minutes under the shared sprint workload before I stopped it. Since the candidate is overlap-invalid regardless of baseline, this does not affect the shipping decision.

## Picker decision

**Do not ship.**

I would not add this as a narrow picker candidate for `small_world_500`, cyclic graphs, lattices, or dense DAGs.

Required safety gates would be so restrictive that no useful target remains:

- weakly connected only, because disconnected spectra are degenerate;
- `overlap_count == 0`;
- `dag_consistency >= baseline`;
- `depth_spearman_rho >= baseline - 0.01`, ignoring NaN cyclic cases carefully;
- `crossing_rate <= baseline + epsilon`;
- live composite improvement over the current native result by a normal picker margin.

Applying those rules rejects `small_world_500` by composite, rejects `small_world_100` by composite/CV, rejects `parallel_cycles_4x5` by DAG regression, rejects `hexagonal_lattice_42` and `clustered_medium_5x20` by overlaps, and rejects `dependency_500` by overlaps.

## Implementation

If this were implemented anyway, the code would be small: roughly 100-140 LOC for Laplacian construction, eigsh fallback, back-edge-filtered layering, scale normalization, and picker scoring. The natural home would be a polish candidate beside `_best_of_polish` in `dagua/layout/ops/pipelines/dagua_native.py`, gated to connected graphs with `N <= 2000` to keep `eigsh` and full picker scoring cheap.

I do not recommend spending that LOC. The live HEAD measurements say the opportunity has been consumed by existing sprint-22 cyclic polish, and the remaining apparent wins are either stale-position artifacts or overlap-invalid layouts.
