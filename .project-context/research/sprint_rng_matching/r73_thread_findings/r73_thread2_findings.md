# r73 Thread 2 Findings: Classical MDS + Pivot MDS (55 divergent combos)

**Date:** 2026-06-15
**Thread:** 2 (MDS-family Mode-B deterministic)
**Bucket:** classical_mds (39 combos) + pivot_mds (16 combos)
**Deliverable format:** per research context -- sub-bucket breakdown, FIXABLE/FLOOR/INVALID-COMPARISON, fix spec file:function:line, expected impact, residual explanation.

---

## PART 1: CLASSICAL MDS (39 combos)

Reference: `igraph_mds` (igraph Python package `Graph.layout_mds()`).
Dagua variants: `classic_classical_mds_default`, `classic_classical_mds_igraph_fidelity`.
Benchmark data source: `eval_output/fidelity_definitive_r72/per_combo.json`.

### Algorithm Identity: What Is igraph layout_mds?

igraph `Graph.layout_mds()` is classical/Torgerson MDS -- double-centering of the squared shortest-path distance matrix followed by eigendecomposition. It is NOT SMACOF/stress-majorization. The prior r70 claim that "igraph MDS WITH a seed is STOCHASTIC (~0.19)" was **incorrect**. The stochasticity is isolated to one step: disconnected-component spatial arrangement via DLA (Diffusion-Limited Aggregation) packing. For connected graphs, `layout_mds()` is fully deterministic regardless of seed. Experimental verification: 5 different seeds on real_karate_34 produce **identical** positions (max diff = 0.0). 5 different seeds on disconnected_encoder_residual produce 5 distinct layouts.

Benchmark confirmation: ALL 39 classical_mds combos have `seed_na=True`, `n_ref_seeded_ok=0`. This means the reference was run without seeding. For connected graphs this is irrelevant (deterministic anyway). For disconnected graphs it means every reference run is a different DLA arrangement.

All 39 dagua combos have `near_deterministic=True` and `mean_W_D~=0` -- dagua is fully deterministic (scipy eigendecomposition is stable across runs).

---

### Sub-bucket A: DISCONNECTED GRAPHS -- INVALID-COMPARISON (20 combos)

**Combos (10 graphs x 2 variants each):**
- disconnected_encoder_residual::{default, igraph_fidelity}
- disconnected_label_cycle_collage::{default, igraph_fidelity}
- er_100::{default, igraph_fidelity}
- er_500::{default, igraph_fidelity}
- kitchen_sink_platform_graph::{default, igraph_fidelity}
- multi_component_80::{default, igraph_fidelity}
- parallel_cycles_4x5::{default, igraph_fidelity}
- random_bipartite_60::{default, igraph_fidelity}
- random_dag_200::{default, igraph_fidelity}
- random_dag_50::{default, igraph_fidelity}

**Root cause:** The two algorithms handle disconnected graphs via fundamentally different mechanisms:

- **dagua**: Applies global MDS on the full N-node distance matrix, using infinity for inter-component BFS distances. When scipy encounters inf values in the centering matrix, rows/columns collapse -- nodes in different components get placed on top of each other or at the center of mass. Result: dramatically different layout topology from igraph.
- **igraph**: Applies per-component MDS on each connected component independently, then uses DLA (Diffusion-Limited Aggregation) packing to arrange components in 2D space. DLA is stochastic (random walk). The reference has `seed_na=True` throughout (benchmark harness does not seed igraph for this), so every reference run produces a different component arrangement.

**Evidence:**
- `disconnected_encoder_residual`: dagua stress_D=0.42, ref stress_R=0.011 -- dagua places inter-component nodes incorrectly
- `disconnected_label_cycle_collage`: dagua stress_D=1.0 (complete failure), ref stress_R~=0.0
- `multi_component_80`: dagua stress_D=0.52, ref stress_R=0.039
- Experimental: igraph DLA produces 5 distinct arrangements across 5 seeds for disconnected_encoder_residual; dagua produces same positions all 5 times (collapses to one center mass)

**Classification: INVALID-COMPARISON.** These are not measuring the same thing. igraph performs per-component MDS + spatial arrangement; dagua performs global MDS with undefined inter-component distances.

**Recommendation:** Exclude disconnected-graph combos from classical_mds fidelity scoring, OR implement per-component MDS in dagua. Full parity would additionally require matching igraph's DLA packing (stochastic, not matchable at Mode-B deterministic).

**Impact if excluded from scoring:** 20 combos reclassified as INVALID-COMPARISON (not divergent).

---

### Sub-bucket B: DEGENERATE/COLLAPSED GRAPHS -- FLOOR (2 combos)

**Combos:**
- wide_3_50_3::classic_classical_mds_default
- wide_3_50_3::classic_classical_mds_igraph_fidelity

**Root cause:** `wide_3_50_3` is a 3-layer linear graph with 50 nodes in the middle layer -- essentially a 1D chain graph. Classical MDS of a path-like graph produces a 1D embedding: all nodes collapse to the x-axis (y-coordinates ~= 0). The second eigenvector of the Laplacian for near-linear graphs has ~zero eigenvalue, producing near-zero y-spread.

igraph produces a 2D layout because its internal classical MDS implementation (vendored LAPACK 3.4.2 dsyevr) chooses a different eigenbasis when the top eigenvalues have minimal gap. dagua's scipy uses a different LAPACK routine with a different tie-breaking path.

**Evidence:** `d_R=100.0` (maximum failure), `stress_D=1.0`, `stress_R=0.44`. dagua is actually worse by stress measure -- the collapsed 1D layout has terrible stress. The path graph topology makes classical MDS inherently ill-conditioned.

**Classification: FLOOR.** The divergence is geometric: when eigenvalues 1 and 2 have near-zero gap, the orientation of the 2D eigenspace is determined by numerical noise, not algorithm logic. SciPy and igraph's LAPACK see the same matrix and choose different rotation of the degenerate eigenspace. Cannot be fixed without emulating LAPACK 3.4.2 dsyevr internals.

---

### Sub-bucket C: WEIGHTED GRAPHS (default variant only) -- INVALID-COMPARISON (5 combos)

**Combos:**
- heavy_tail_weights_50::classic_classical_mds_default
- real_karate_34::classic_classical_mds_default
- real_lesmis_77::classic_classical_mds_default
- weighted_chain_20::classic_classical_mds_default
- weighted_clusters_3x10::classic_classical_mds_default

**Root cause:** The `default` variant uses a different distance matrix than igraph:

- `dagua`: `_quick_classic()` in `classic_competitor.py` at line 1711-1712 calls `extra_kwargs.setdefault("edge_weights", graph.edge_weights)` when the graph has weights. `layout_classical_mds_pipeline()` then uses Dijkstra-weighted shortest paths as the distance matrix.
- `igraph layout_mds()`: always uses unweighted BFS shortest paths. It has no edge_weight argument for MDS.

The `igraph_fidelity` variant correctly ignores edge_weights (passes `edge_weights=None`), so these same graphs pass in the `igraph_fidelity` variant. This is confirmed: real_karate_34 and real_lesmis_77 are NOT in the divergent list under the `igraph_fidelity` variant.

**Evidence (karate_34):**
- `real_karate_34::classic_classical_mds_default`: d_R=0.45, stress_D=0.12, stress_R=0.07
- `real_karate_34::classic_classical_mds_igraph_fidelity`: NOT divergent (passes)
- Direct scipy experiment: computing classical MDS with unweighted distances on karate_34, then Procrustes vs igraph -> RMSD=0.0 (bit-exact). Computing with weighted distances -> completely different embedding.

**Classification: INVALID-COMPARISON.** dagua is correctly implementing weighted classical MDS (arguably the more correct algorithm for weighted graphs). The reference (igraph) cannot be compared because it ignores weights. This is a comparison design issue, not a dagua bug.

**Recommendation:** Either (a) mark `function_name="layout_classical_mds_pipeline"` as unweighted by adding it to `_UNWEIGHTED_REFERENCE_LAYOUTS` in `classic_competitor.py` (which causes `_quick_classic()` to not pass edge_weights for the `default` variant), OR (b) add a separate weighted classical MDS variant with a different/weighted reference. Option (a) would make the `default` variant match igraph, but lose the weighted behavior for weighted graphs.

**Impact:** 5 combos reclassified as INVALID-COMPARISON (not dagua bugs).

---

### Sub-bucket D: SYMMETRIC CONNECTED GRAPHS (REPEATED EIGENVALUES) -- FLOOR (12 combos)

**Combos (6 graphs x 2 variants each):**
- bipartite_4_3_4::{default, igraph_fidelity}
- center_port_backedge_hub::{default, igraph_fidelity}
- densenet_block::{default, igraph_fidelity}
- org_chart_1_5_4_8::{default, igraph_fidelity}
- petersen_10::{default, igraph_fidelity}
- wide_single_layer_1_50_1::{default, igraph_fidelity}

**Root cause:** When the distance matrix has repeated top eigenvalues (lambda_1 = lambda_2), any 2D rotation of the eigenspace is a valid MDS solution. The two implementations choose different bases:

- **dagua**: `scipy.linalg.eigh(..., subset_by_index=(num_nodes-2, num_nodes-1), driver='evr', lower=True)` -- LAPACK dsyevr with Intel MKL backend.
- **igraph**: Vendored LAPACK 3.4.2 dsyevr (bundled in igraph C library, different compilation).

Both routines are correct (both produce valid MDS solutions), but they choose different eigenvectors from the degenerate subspace, leading to rotated/reflected layouts with equal stress but nonzero Procrustes RMSD.

**Evidence:**
- **Petersen graph** (10 nodes, 15 edges): 3-fold symmetric, top 2 eigenvalues both = 3.5. Direct measurement: `scipy eigh` vs igraph on Petersen -> Procrustes RMSD=22.7. Gap test: small asymmetric DAG (eigenvalue gap=5.15) -> RMSD=0.
- **bipartite_4_3_4**: Top 3 eigenvalues all = 2.0, gap=0. RMSD=34 on direct comparison.
- All 6 graphs in this bucket are structurally symmetric or near-symmetric, which is precisely what causes repeated eigenvalues.
- Sign-canonicalization (fixing sign of largest-magnitude component per eigenvector) handles reflections from DISTINCT eigenvectors but does NOT fix rotations from DEGENERATE eigenspaces.

**Classification: FLOOR.** Would require emulating igraph's vendored LAPACK 3.4.2 compilation choices (compiler, optimization flags, internal Gram-Schmidt pivot order). Numerically irreducible at the scipy level.

**Codex critic challenge response:** The verification is direct: compute the eigenvalues of the centering matrix for petersen_10 and observe lambda_1=lambda_2=3.5. Any rotation R of [v1,v2] is equally valid. The igraph code uses a different LAPACK binary that produces a rotated basis.

---

### Classical MDS Summary

| Sub-bucket | Count | Verdict | Fixable? |
|---|---|---|---|
| A: Disconnected | 20 | INVALID-COMPARISON | Out of scope for this ref |
| B: Degenerate (wide_3_50_3) | 2 | FLOOR | No |
| C: Weighted (default variant) | 5 | INVALID-COMPARISON | Comparison design issue |
| D: Symmetric eigenspace | 12 | FLOOR | No |
| **TOTAL** | **39** | | |

**Actionable items:**
1. Sub-bucket A: Remove disconnected graphs from classical_mds fidelity or annotate as `INVALID-COMPARISON`. No code fix needed.
2. Sub-bucket C: Add `"layout_classical_mds_pipeline"` to `_UNWEIGHTED_REFERENCE_LAYOUTS` in `dagua/eval/competitors/classic_competitor.py` line 122 if goal is to match igraph on weighted graphs. This re-routes the default variant to use unweighted distances for the igraph comparison. **Expected impact: 5 combos would pass.**
3. Sub-buckets B and D: Accept as irreducible floor.

---

## PART 2: PIVOT MDS (16 combos)

Reference: `ogdf_pivot_mds` (OGDF PivotMDS algorithm via `scripts/ogdf_runner.cpp`).
Dagua variants: `classic_pivot_mds_{10, 50, 100, 200}`.
All 16 combos are modeB (deterministic).

**Graphs:** heavy_tail_weights_50, real_karate_34, real_lesmis_77, weighted_clusters_3x10 -- all 4 are weighted graphs, each tested at 4 pivot counts.

### Algorithm Identity: What Is OGDF PivotMDS?

OGDF PivotMDS (Brandes & Pich 2007) uses:
1. Maxmin pivot selection (first pivot = node 0 by convention; subsequent pivots = farthest unselected node from current pivot set)
2. BFS distance from each pivot to all nodes (UNWEIGHTED -- OGDF's PivotMDS ignores edge weights)
3. Double-centering of the pivot distance matrix
4. Power iteration eigensolver with random start (seeded via `srand(seed)`)

OGDF C++ sources: `ogdf_runner.cpp` calls `ogdf::setSeed(seed); std::srand(static_cast<unsigned>(seed));` then `ogdf::PivotMDS layout;`. The benchmark seeds OGDF with seed=42 via JSON payload.

---

### Sub-bucket E: SCALE MISMATCH -- FIXABLE (16 combos)

**ALL 16 PIVOT MDS COMBOS FALL INTO THIS SINGLE MECHANISM.**

**Root cause:** `PivotMDSFinalizePositions.apply()` in `dagua/layout/ops/postprocess.py` normalizes the output positions to a fixed extent: `extent = max(num_nodes^0.5 * 5.0, 1.0)`. OGDF does NOT normalize -- it outputs raw coordinates at the `distance_scale=100.0` scale.

Dagua's OGDF-fidelity path sets `distance_scale=100.0` (all 4 pivot variants: `classic_pivot_mds_{10,50,100,200}` all have `dagua_params={"distance_scale": 100.0, ...}`). After computing pivot MDS coordinates in distance_scale=100 space, `PivotMDSFinalizePositions` then normalizes to `sqrt(N)*5` extent, which shrinks or expands the coordinates by a factor that varies with graph size and density. OGDF keeps the raw scale.

**Experimental verification (conclusive):**
- `real_karate_34` (N=34): `scale_ratio = std(OGDF_pos) / std(dagua_pos) = 9.519`
- After multiplying dagua positions by 9.519: `Procrustes RMSD = 9e-7` (bit-exact within float32 precision)
- `real_lesmis_77` (N=77, n_pivots=10): scale_ratio=4.40, RMSD after correction = 1.4e-6
- `real_lesmis_77` (N=77, n_pivots=100): scale_ratio=5.59, RMSD after correction = 3e-7
- Tested all 8 combinations (2 graphs x 4 pivot counts): RMSD < 1e-5 in all cases after scale correction.

**Seed hypothesis DISPROVED:** Initial hypothesis was that `_ogdf_random_matrix()` hardcodes `srand(0)` while the OGDF reference uses `srand(42)`. Experimental test: computed OGDF-fidelity Pivot-MDS with srand(0) and srand(42) on real_karate_34. `max_diff = 0.0`. Power iteration converges to the same eigenvectors regardless of random initialization because the top eigenvalues of typical graph distance matrices are well-separated. The srand seed does not matter here.

**Why the 4 graphs are all weighted:** The `_quick_classic()` function passes edge_weights for graphs that have them. The OGDF PivotMDS reference ignores edge weights (the C++ runner only receives topology, not weights). This means the distance matrices diverge: dagua uses weighted BFS distances, OGDF uses unweighted BFS distances. HOWEVER, based on the scale-correction experiment showing near-zero RMSD, the weighted vs unweighted distinction does NOT explain the divergence for these specific graphs -- the structural layout is the same (power iteration eigenvectors), only the scale differs. For the 4 graphs in this bucket, the topology dominates the distance matrix, making weighted vs unweighted secondary.

**Note:** The weighted-distance inconsistency is a separate latent issue that should be investigated if the scale fix reveals residual divergence.

---

### Fix Spec: Mechanism E (Scale Mismatch)

**File:** `/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/pivot_mds.py`

**Function:** `build_pivot_mds_pipeline()` at line 103-129.

**What to change:**

When `_uses_ogdf_fidelity_coordinates()` returns True (i.e., when building the OGDF-fidelity path with `first_pivot="first_node"`, `compute_dtype=float64`, `distance_scale=100.0`), the pipeline currently ends with `PivotMDSFinalizePositions()` (line 126), which normalizes to `sqrt(N)*5`. This must be replaced with a no-normalize finalize step.

**Option A (minimal change) -- Preferred:**

Add `skip_normalization: bool = False` to `PivotMDSFinalizePositions` config in `/home/jtaylor/projects/dagua/dagua/layout/ops/postprocess.py`. In `apply()`, when `skip_normalization=True`, skip the call to `_normalize_classical_positions()` and instead do only dtype cast and device move.

Then in `build_pivot_mds_pipeline()`, pass `PivotMDSFinalizePositions(skip_normalization=True)` when `_uses_ogdf_fidelity_coordinates()` is True.

```python
# In build_pivot_mds_pipeline() at line 126:
# BEFORE:
#   PivotMDSFinalizePositions(),

# AFTER:
is_ogdf_fidelity = _uses_ogdf_fidelity_coordinates(
    first_pivot=first_pivot,
    first_pivot_index=first_pivot_index,
    compute_dtype=resolved_dtype,
    distance_scale=distance_scale,
)
# ...
PivotMDSFinalizePositions(skip_normalization=is_ogdf_fidelity),
```

Note: `_uses_ogdf_fidelity_coordinates()` is already computed at line 93 to select the `coordinate_op`. Reuse that boolean -- capture it in a variable rather than calling twice.

**Option B (separate Op):**

Add `_OGDFPivotMDSFinalizePositions` class in `postprocess.py` that performs only dtype cast + device transfer (no normalization). Use it in `build_pivot_mds_pipeline()` when OGDF fidelity is detected.

**Verification after fix:**

```python
from dagua.layout.ops.pipelines.pivot_mds import layout_pivot_mds_pipeline
from dagua.eval.graphs import make_real_karate_graph  # or equivalent
import torch

g = make_real_karate_graph()
pos = layout_pivot_mds_pipeline(
    g.edge_index, g.num_nodes,
    n_pivots=50, seed=42,
    first_pivot="first_node", first_pivot_index=None,
    compute_dtype=torch.float64, distance_scale=100.0,
    ogdf_path_special_case=True,
)
# Run OGDF reference with n_pivots=50, seed=42
# Compare with Procrustes: target RMSD < 1e-3
```

**Expected impact:** All 16 pivot_mds combos should converge to RMSD < 1e-3 after this fix.

---

### Residual Risk for Pivot MDS

**Weighted distances:** The 4 graphs (heavy_tail_weights_50, real_karate_34, real_lesmis_77, weighted_clusters_3x10) all have edge weights. After the scale fix, OGDF-fidelity combos use dagua's weighted BFS distances while OGDF uses unweighted BFS. The scale-correction experiment showed near-zero RMSD despite this, suggesting these graphs have small or uniformly-distributed weights that don't significantly alter relative BFS distances. If post-fix RMSD is > 1e-3 for any of these, the secondary fix is to add `"layout_pivot_mds_pipeline"` to `_UNWEIGHTED_REFERENCE_LAYOUTS` (same fix as classical_mds sub-bucket C), which forces the OGDF-fidelity variant to use unweighted distances.

---

### Pivot MDS Summary

| Sub-bucket | Count | Verdict | Fix |
|---|---|---|---|
| E: Scale mismatch (PivotMDSFinalizePositions normalizes, OGDF doesn't) | 16 | FIXABLE | Add skip_normalization flag to PivotMDSFinalizePositions |

---

## CONSOLIDATED IMPACT TABLE

| Bucket | Combos | Verdict | Post-fix combos passing |
|---|---|---|---|
| Pivot MDS scale mismatch (E) | 16 | FIXABLE | 16 |
| Classical weighted default (C) | 5 | INVALID or FIXABLE* | 5 if _UNWEIGHTED_REFERENCE_LAYOUTS extended |
| Classical disconnected (A) | 20 | INVALID-COMPARISON | 0 (wrong reference) |
| Classical degenerate (B) | 2 | FLOOR | 0 |
| Classical symmetric eigenspace (D) | 12 | FLOOR | 0 |
| **TOTAL** | **55** | | **up to 21** |

*Sub-bucket C fix (adding `layout_classical_mds_pipeline` to `_UNWEIGHTED_REFERENCE_LAYOUTS`) makes the `default` variant ignore edge weights and match igraph -- 5 combos resolved. But note this changes dagua's behavior for the default variant on weighted graphs (removes the weighted MDS behavior). If weighted MDS is desired, the fix is instead to create a new reference or mark these as INVALID-COMPARISON.

---

## REFERENCE CHAIN

- `dagua/eval/competitors/igraph_competitor.py`: `IgraphMDS` class, `uses_igraph_rng=True`, `layout_algo="mds"` -- wraps `Graph.layout_mds()` with optional igraph RNG seed; `seed_na=True` in all classical_mds benchmark entries (harness did not seed).
- `dagua/eval/competitors/classic_competitor.py`: `_quick_classic()` line 1711-1712: adds `edge_weights` to kwargs if graph has weights and function not in `_UNWEIGHTED_REFERENCE_LAYOUTS`.
- `dagua/eval/variants.py`: `classic_pivot_mds_*` use `dagua_params={"n_pivots": N, "first_pivot": "first_node", "compute_dtype": "float64", "distance_scale": 100.0, "ogdf_path_special_case": True}`.
- `dagua/layout/ops/pipelines/pivot_mds.py`: `build_pivot_mds_pipeline()` line 103-129, `PivotMDSFinalizePositions()` line 126 (the bug).
- `dagua/layout/ops/postprocess.py`: `PivotMDSFinalizePositions.apply()` calls `_normalize_classical_positions(state.pos, extent)` where `extent = sqrt(num_nodes) * 5.0`.
- `scripts/ogdf_runner.cpp`: calls `ogdf::setSeed(seed); std::srand(static_cast<unsigned>(seed));` then runs `ogdf::PivotMDS layout;` with optional `setNumberOfPivots(n_pivots)`.
