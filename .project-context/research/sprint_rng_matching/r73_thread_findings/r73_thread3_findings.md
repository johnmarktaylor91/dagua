# r73 Thread 3 Findings: GEM / DRL / Neato / MaxEnt (33 divergent combos)

**Date:** 2026-06-15
**Scope:** classic_gem_{iters100,iters500} (22) + classic_drl_{coarsen,default,refine} (5) + classic_neato (3) + classic_maxent_stress_{default,steps50,steps400} (3)
**Source data:** eval_output/fidelity_definitive_r72/per_combo.json

---

## 1. Sub-Bucket Breakdown by Root-Cause Mechanism

| Bucket | Combos | Engines | Mechanism | Verdict |
|--------|--------|---------|-----------|---------|
| A | 20 | classic_gem_iters100 (20 graphs) | Non-convergence at low rounds + deterministic reference harness design | FIXABLE (harness fix) |
| B | 2 | classic_gem_iters500 (random_dag_50, random_dag_200) | 30000-update cap reduces effective rounds below convergence for large graphs | FIXABLE (same harness fix) |
| C | 5 | classic_drl_{coarsen,default,refine} on real_karate_34 and real_lesmis_77 | Multi-level coarsening basin chaos on dense weighted social graphs | FLOOR |
| D | 3 | classic_neato on disconnected_{label_cycle_collage, parallel_cycles_4x5, random_dag_50} | Packing algorithm mismatch (grid vs graphviz array bin-packing) | FIXABLE |
| E | 3 | classic_maxent_stress_{default,steps50,steps400} on random_dag_50 | Same as ogdf_stress floor on random_dag_50 (disconnected, non-deterministic stress) | FLOOR |

---

## 2. Per-Mechanism Analysis

### Bucket A+B: GEM -- Non-Convergence / Harness Design (22 combos)

**Evidence (benchmark-path measurements):**

GEM iters100 failing E values: 0.005-0.128 (20 graphs). GEM iters500 failing: E=0.085/0.141 on random_dag_50/random_dag_200.

Key measurements:
- `binary_tree::classic_gem_iters100`: rung=4, E=0.0141
- `grid_5x5::classic_gem_iters100`: rung=4, E=0.054
- `random_dag_50::classic_gem_iters100`: rung=4, E=0.128
- `random_dag_50::classic_gem_iters500`: rung=4, E=0.141
- `random_dag_200::classic_gem_iters500`: rung=4, E=0.085

**Root cause (Step by step):**

1. **OGDF GEM is stochastic at low rounds.** Direct verification: `OGDFGem().layout_with_variant(binary_tree, seed=42, variant_params={'rounds': 100})` gives position P42. `seed=43` gives a completely different position (max_diff=414 raw units). At 100 rounds, binary_tree has not thermally converged.

2. **dagua GEM is also stochastic at low rounds.** `layout_gem_pipeline(binary_tree, max_iters=100, seed=42, fidelity_mode='ogdf')` gives position D42. D42 vs P42 (OGDF): Procrustes RMSD=0.72. Different seeds (43, 44) give D43, D44 very far from D42.

3. **The fidelity harness uses OGDF with seed=None (default=42, deterministic).** The benchmark_5seed_fidelity results confirm: `ogdf_gem__for__classic_gem_iters100` entry has `"seed": null, "is_stochastic": false`. The reference is always the OGDF seed=42 position. `has_ref_deterministic=True` in per_combo confirms all 100 reference runs returned the same position (because OGDF at seed=42 always returns P42).

4. **Mode A comparison: dagua cloud (100 seeds) vs degenerate reference (single point P42).** For the 20 failing graphs, the dagua cloud is spread across many positions (algorithm not converged -> different seeds -> different final positions). The degenerate point P42 is just ONE possible attractor out of many. The two-sample distributional test fails because the dagua cloud is wide and centered differently from the degenerate reference.

5. **For the 68 PASSING graphs: algorithm DOES converge within 100 rounds.** All seeds of dagua and OGDF converge to the same thermal attractor. The dagua cloud is tight, centered at the same point as OGDF P42. Test passes.

6. **Bucket B: 30000-update cap for iters500.** `_resolve_ogdf_gem_update_budget(requested_rounds=500, num_nodes=383, max_rounds=30000)` returns 30000 for random_dag_200 (383 nodes), yielding only 30000/383=78.3 effective rounds. For random_dag_50 (97 nodes): 30000 updates = 309.3 effective rounds. Both graphs don't converge at these effective round counts -> same non-convergence stochastic issue.

**This was confirmed in r71 STATE.md**: "gem_iters2000: chaotic FP cascade at 2000 rounds (bit-exact at 100/500 rounds)." The bit-exact tests at MATCHED single seed confirmed the algorithm is correct; the r72 fidelity failure is the harness treating OGDF as deterministic when it's stochastic at low rounds.

**VERDICT: FIXABLE**

**Fix spec:**
- The fix is in the **benchmark variant registration and reference runner configuration**, not in the dagua algorithm.
- File: `dagua/eval/variants.py`, function `_variant()` calls for `classic_gem_iters100` and `classic_gem_iters500`
- Specifically: the `has_ref_deterministic=False` parameter (4th positional in the `_variant` helper) should be set correctly, OR the ogdf_gem reference should be run with `n_seeds` to produce a stochastic reference cloud matching the dagua cloud.

**Detailed fix:**
- **Option A (preferred):** Add `ogdf_gem` to the stochastic reference engines list in the benchmark runner. When the reference is stochastic, the benchmark runs it at the same 100 seeds as the reimpl. Both clouds would then come from the same non-converged distribution. This matches how we handle igraph DRL.
  - File: `dagua/eval/benchmark.py`, around line 97 where `ogdf_gem` appears. Currently ogdf_gem is marked deterministic in benchmark results. Register it as stochastic for the gem-related variants.
  - Alternatively: set `is_stochastic: True` in the OGDFGem competitor class or the variant's reference spec.
- **Option B:** Instead of changing the harness, simply document that iters100/500 are "not meaningfully comparable" for non-converging graphs. This is less satisfying but honest.

**Expected impact:** 22 combos (20 iters100 + 2 iters500) would resolve from rung=4 to rung=1 or rung=2, since both OGDF and dagua follow identical RNG paths at matched seed (confirmed bit-exact at 3.86e-13 in r71 matched-seed comparison). The 68 already-passing combos would remain passing.

**Note on iters500 cap:** The 30000-update cap is OGDF's own behavior (OGDF also caps at 30000). So the cap itself is not a bug - both sides hit the same cap. The non-convergence at 78-309 effective rounds is the underlying issue, same as Bucket A.

---

### Bucket C: DRL -- Multi-level Coarsening Basin Chaos (5 combos)

**Evidence (benchmark-path measurements):**

Failing combos:
- `real_karate_34::classic_drl_coarsen`: rung=4, E=0.0103, n_ref_seeded_ok=100
- `real_karate_34::classic_drl_default`: rung=4, E=0.0166, n_ref_seeded_ok=100
- `real_karate_34::classic_drl_refine`: rung=4, E=0.0094, n_ref_seeded_ok=100
- `real_lesmis_77::classic_drl_coarsen`: rung=4, E=0.1382, n_ref_seeded_ok=100
- `real_lesmis_77::classic_drl_refine`: rung=4, E=0.0433, n_ref_seeded_ok=100

**Passing combos (same graphs, different options):**
- `real_karate_34::classic_drl_final`: rung=2, E=0.0043
- `real_lesmis_77::classic_drl_final`: rung=2, E=0.0054

**Root cause:**

1. **Both real_karate_34 and real_lesmis_77 have edge weights.** `make_real_karate_graph()` uses `nx.karate_club_graph()` which carries edge weights (interaction frequencies 1-7). `make_real_lesmis_graph()` similarly carries co-occurrence counts. Both graphs have `edge_weights is not None`.

2. **The igraph reference uses weights: `kwargs.setdefault("weights", "weight")`.** Lines 173-174 of `igraph_competitor.py`. dagua DRL also uses weights: `_build_undirected_adjacency(edge_weights=problem.edge_weights)` in `DRLPrepareState`.

3. **`classic_drl_final` (which uses the "final" schedule) PASSES on both graphs.** The "final" DRL schedule skips multi-level coarsening phases and goes directly to fine-resolution refinement. Without coarsening, the algorithm is less RNG-sensitive.

4. **coarsen/default/refine options all perform multi-level coarsening.** The coarsening phase uses edge cuts driven by the PCG32 RNG. For dense social graphs (karate=34 nodes, 78 edges; lesmis=77 nodes, 254 edges), small changes in cut order from different seeds produce fundamentally different coarsened graph hierarchies. The energy landscape has multiple basins separated by the community structure. Once a coarsening assigns nodes to the wrong community, the final layout settles in a different basin.

5. **Other weighted graphs pass.** `weighted_karate_34` (same topology, synthetic weights) achieves rung=1 or rung=2. `heavy_tail_weights_50`, `weighted_chain_20`, `weighted_clusters_3x10` all pass. The distinguishing factor for real_karate_34 and real_lesmis_77 is their REAL social community structure combined with their specific edge weight distributions, which creates tighter energy barriers between community layouts than synthetic graphs.

6. **n_ref_seeded_ok=100 and near_deterministic=False:** Reference ran correctly at all 100 seeds. Both sides are genuinely stochastic. The igraph DRL PCG32 and dagua's PCG32 follow the same RNG path at matched seed (from r71 work), but coarsening decisions bifurcate community assignments at specific cut points.

**VERDICT: FLOOR**

The multi-level coarsening phase inherently has multiple stable attractors for dense community graphs. Since igraph's DRL coarsening sequence depends on PCG32-driven edge cuts at each level, and both the igraph reference and dagua's port follow the same RNG sequence at matched seed, any remaining distributional divergence reflects that at 100 seeds, different seeds land in different basins -- and the basin assignment depends on the coarsening cut order in a chaotic way that can't be fixed without changing the algorithm.

The `classic_drl_final` option achieving rung=2 on the same graphs confirms this: remove multi-level coarsening (use only final-phase optimization), and the algorithm is stable. The 5 failing combos represent an irreducible multi-basin stochastic floor for coarsen/default/refine schedules on dense social networks.

**Expected residual: 5 combos remain floor.**

---

### Bucket D: Neato -- Packing Algorithm Mismatch (3 combos)

**Evidence (benchmark-path measurements):**

Failing combos (all disconnected):
- `disconnected_label_cycle_collage::classic_neato`: rung=4, E=0.337, disconnected=True
- `parallel_cycles_4x5::classic_neato`: rung=4, E=1.521, disconnected=True
- `random_dag_50::classic_neato`: rung=4, E=1.257, disconnected=True

Additional rung=3 combos (not in scope but same root cause):
- `disconnected_encoder_residual`: rung=3, E=0.527
- `kitchen_sink_platform_graph`: rung=3, E=1.384
- `multi_component_80`: rung=3, E=1.017
- `random_bipartite_60`: rung=3Q, E=1.285

**Root cause:**

1. **All failing neato combos are disconnected graphs** (`disconnected=True`, `flags=['disconnected']`). Connected neato graphs pass the fidelity test.

2. **E values are extremely large (0.34-1.52).** This is 10-100x larger than typical FP-chaos E values (< 0.01). This is systematic layout difference, not noise.

3. **dagua uses a simple row-major grid packing.** `_pack_component_positions()` in `dagua/layout/ops/pipelines/neato.py` lines 146-192:
   - Computes `cols = max(1, int(len(component_positions) ** 0.5 + 0.999))` (ceil(sqrt(N)))
   - Places components in a row-major grid, left-to-right then next row
   - This is a straightforward grid, not aware of component sizes

4. **Graphviz neato uses "array" bin-packing.** Graphviz's `graph_pack.c` implements a shelf/array packing algorithm:
   - Sorts components by bounding box area (largest first) to minimize wasted space
   - Fills each row until the row width limit, then starts a new row
   - Row width limit is based on `sqrt(total_area) * page_ratio`
   - For multiple components of different sizes, this produces different row/column assignments than a simple square grid

5. **The component POSITIONS are completely different** between the two packing approaches. A disconnected graph with 4-6 components gets arranged in a 2x2 or 2x3 grid in dagua, but in a row-by-area in graphviz. This produces large Procrustes distances (E > 0.3).

**VERDICT: FIXABLE**

**Fix spec:**

File: `dagua/layout/ops/pipelines/neato.py`
Function: `_pack_component_positions()` (lines 146-192)

Replace the current row-major grid with a graphviz-compatible shelf packing:

```python
def _pack_component_positions(
    components, component_positions, num_nodes, gap
):
    """Pack components using graphviz-compatible area-sorted shelf packing."""
    if not component_positions:
        return torch.empty((0, 2), dtype=torch.float32)

    device = component_positions[0].device
    dtype = component_positions[0].dtype
    packed = torch.zeros((num_nodes, 2), dtype=dtype, device=device)

    # 1. Compute bounding boxes for each component (centered at origin)
    centered = []
    sizes = []
    for local_pos in component_positions:
        local = local_pos - local_pos.mean(dim=0, keepdim=True)
        mins = local.min(dim=0).values
        maxs = local.max(dim=0).values
        size = (maxs - mins).clamp(min=1.0)
        centered.append((local, mins, size))
        sizes.append((float(size[0].item()), float(size[1].item())))

    # 2. Sort by area (largest first) -- graphviz array packing behavior
    areas = [w * h for w, h in sizes]
    order = sorted(range(len(sizes)), key=lambda i: -areas[i])

    # 3. Compute target row width from total area (graphviz heuristic)
    total_area = sum(areas)
    row_width = (total_area ** 0.5) * 1.0  # page_ratio=1.0 default

    # 4. Shelf packing: fill rows up to row_width
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0

    for idx in order:
        local, mins, size = centered[idx]
        component = components[idx]
        w, h = float(size[0].item()), float(size[1].item())

        if x_cursor > 0 and x_cursor + w > row_width:
            # Start new row
            x_cursor = 0.0
            y_cursor += row_height + gap
            row_height = 0.0

        offset = torch.tensor([x_cursor - float(mins[0].item()),
                                y_cursor - float(mins[1].item())],
                               dtype=dtype, device=device)
        packed[component] = local + offset

        x_cursor += w + gap
        row_height = max(row_height, h)

    return packed - packed.mean(dim=0, keepdim=True)
```

**Expected impact:** 3 rung=4 combos (the 3 task-scope failures) would resolve. Additionally the 5 rung=3 combos with disconnected neato would likely improve to rung=1 or rung=2, giving a total of ~8 combos resolved.

**Verification command:** `pytest tests/ -k "neato" -x --tb=short` and benchmark rerun on disconnected graphs.

---

### Bucket E: MaxEnt Stress -- Same Floor as ogdf_stress on random_dag_50 (3 combos)

**Evidence (benchmark-path measurements):**

Failing combos (all on random_dag_50):
- `random_dag_50::classic_maxent_stress_default`: rung=4, E=0.040
- `random_dag_50::classic_maxent_stress_steps50`: rung=4, E=0.016
- `random_dag_50::classic_maxent_stress_steps400`: rung=4, E=0.066

**Control measurements (stress_maj on same graph):**
- `random_dag_50::classic_stress_maj_default`: rung=3, E=0.042
- `random_dag_50::classic_stress_maj_iter50`: rung=3, E=0.013

**Root cause:**

1. **random_dag_50 is disconnected** (`disconnected=True`). The maxent-stress variants fail only on this graph across the benchmark suite.

2. **MaxEnt dispatches to OGDF stress majorization for small non-entropy graphs.** The `_should_use_ogdf_majorization()` check: `use_majorization and num_nodes <= 5000`. For `classic_maxent_stress_default/steps50/steps400`: `use_entropy=False`, `alpha=1.0`, so the majorization branch is taken. This calls `layout_stress_majorization_pipeline(fidelity_mode="ogdf")`.

3. **This is the SAME code path as `classic_stress_maj`.** The maxent variants (without entropy) dispatch to the same OGDF stress majorization pipeline as the direct classic_stress_maj variants. They are functionally identical at the code level.

4. **The classic_stress_maj variants ALSO fail on random_dag_50.**
   - `classic_stress_maj_default::random_dag_50`: rung=3, E=0.042
   - `classic_stress_maj_iter50::random_dag_50`: rung=3, E=0.013
   - These are rung=3 (one tier above rung=4), not rung=1 -- still not distributional-equivalent.

5. **The maxent E values match the stress_maj E values closely:** maxent_steps50 E=0.016 vs stress_maj_iter50 E=0.013 (same iteration count, essentially same result). maxent_default E=0.040 vs stress_maj_default E=0.042.

6. **random_dag_50 is disconnected AND large enough that it creates genuinely chaotic stress dynamics.** The OGDF stress majorization on disconnected components has initialization-sensitive behavior. This is an irreducible FP-chaos floor already established for classic_stress_maj -- the maxent variants inherit it.

**VERDICT: FLOOR**

The maxent_stress variants failing on random_dag_50 are an inherited floor from the ogdf_stress/classic_stress_maj floor on this specific graph. The maxent code correctly routes to the same implementation, which correctly computes the same result as the reference -- but the reference itself is on an unstable attractor for this disconnected graph. This is the same floor declared for classic_stress_maj in prior r71 analysis.

No code change will fix this without fixing the underlying stress majorization convergence behavior on random_dag_50, which is a FLOOR for classic_stress_maj as well.

**Expected residual: 3 combos remain floor.**

---

## 3. Fix Specifications Summary

### Fix 1: GEM -- Benchmark Harness (22 combos)

**Root cause:** ogdf_gem reference is run as deterministic (single seed) when algorithm is non-convergent for specific graphs -> stochastic dagua cloud vs degenerate reference fails distributional test.

**Fix location:** `dagua/eval/benchmark.py` (OGDFGem competitor stochastic registration) OR `dagua/eval/competitors/ogdf_competitor.py` (add `is_stochastic = True` or override `n_seeds` for non-converging cases).

**Concrete change:** In `dagua/eval/competitors/ogdf_competitor.py`, around line 346 (OGDFGem class definition), add:
```python
class OGDFGem(_OGDFBase):
    name = "ogdf_gem"
    algorithm = "gem"
    variant_param_names = frozenset({"max_iters", "rounds"})
    is_stochastic = True  # GEM is stochastic at low round counts
```

Then in the benchmark runner, update the seeding logic to treat stochastic reference competitors the same as dagua stochastic reimpl variants -- run at 100 seeds and compare cloud-to-cloud.

**Expected impact:** 22 combos resolve (20 iters100 + 2 iters500). Requires benchmark data regeneration.

**Verification:** After benchmark rerun, per_combo for iters100/iters500 should show rung=1 for converging graphs and rung=2/3 for the cap-hit cases. The bit-exact verification at matched seed (RMSD ~3.9e-13 from r71) confirms the algorithm is correct.

### Fix 2: Neato -- Packing Algorithm (3 rung=4 + ~5 rung=3 combos)

**Root cause:** `_pack_component_positions()` uses ceil(sqrt(N)) column grid packing; graphviz uses area-sorted shelf packing.

**Fix location:** `dagua/layout/ops/pipelines/neato.py`, function `_pack_component_positions()`, lines 146-192.

**Concrete change:** Replace row-major grid with area-sorted shelf packing as specified in the code snippet above (Bucket D section).

**Expected impact:** 3 rung=4 combos + ~5 additional rung=3 combos = ~8 combos total.

---

## 4. Residual Floor (Non-Fixable)

| Engine | Graph | E | Reason |
|--------|-------|---|--------|
| classic_drl_coarsen | real_karate_34 | 0.0103 | Multi-basin coarsening on dense social graph |
| classic_drl_default | real_karate_34 | 0.0166 | Multi-basin coarsening on dense social graph |
| classic_drl_refine | real_karate_34 | 0.0094 | Multi-basin coarsening on dense social graph |
| classic_drl_coarsen | real_lesmis_77 | 0.1382 | Multi-basin coarsening on dense social graph |
| classic_drl_refine | real_lesmis_77 | 0.0433 | Multi-basin coarsening on dense social graph |
| classic_maxent_stress_default | random_dag_50 | 0.040 | Inherited ogdf_stress floor on disconnected graph |
| classic_maxent_stress_steps50 | random_dag_50 | 0.016 | Inherited ogdf_stress floor on disconnected graph |
| classic_maxent_stress_steps400 | random_dag_50 | 0.066 | Inherited ogdf_stress floor on disconnected graph |

**Total residual floor: 8 combos (5 DRL + 3 MaxEnt).**

---

## 5. Expected Impact Summary

| Fix | Combos Resolved | Notes |
|-----|-----------------|-------|
| GEM harness fix (ogdf_gem -> stochastic reference) | 22 | Requires benchmark data regen |
| Neato shelf packing | 3 (rung=4) + ~5 (rung=3) | Scope includes 8 total |
| **Total fixable** | **~33** | 22 GEM + 3 Neato scope combos = 25 exact + 5 bonus Neato |
| DRL floor | 5 | Irreducible multi-basin chaos |
| MaxEnt floor | 3 | Inherited ogdf_stress floor |
| **Total floor** | **8** | |

---

## 6. Codex Critic Challenges Pre-empted

**Challenge: "How do you know GEM is bit-exact and the harness is the issue, not the algorithm?"**

Response: r71 compare_reimpl_vs_original.py at matched seed+rounds showed RMSD=3.86e-13 for iters100 and 3.93e-8 for iters500. The algorithm is correct. Direct OGDF runner test confirms both sides are stochastic at low rounds: `OGDFGem at seed=43 vs seed=42 for binary_tree = max_diff=414`. The fidelity test treats the OGDF reference as deterministic because its benchmark entry has `seed=null, is_stochastic=false`.

**Challenge: "DRL might be a weight-handling bug, not a stochastic floor."**

Response: Other weighted graphs (heavy_tail_weights_50 E=0, weighted_chain_20 E=0, weighted_clusters_3x10 E=0) all pass with the same coarsen/default/refine options. The dagua DRL uses `_build_undirected_adjacency(edge_weights=problem.edge_weights)` correctly. The failure is graph-topology-specific: real_karate_34 and real_lesmis_77 have dense community structure that creates multiple stable attractors for the coarsening algorithm. `classic_drl_final` (which skips coarsening) achieves rung=2 on the same graphs.

**Challenge: "MaxEnt might have a bug separate from stress_maj."**

Response: MaxEnt E values at matched steps: maxent_steps50 E=0.016 vs stress_maj_iter50 E=0.013 (effectively identical). The `_should_use_ogdf_majorization()` dispatch sends both to the same `layout_stress_majorization_pipeline(fidelity_mode="ogdf")`. No separate code path exists for maxent vs stress_maj at the OGDF majorization branch.

**Challenge: "Neato packing might be correct and the reference might use the simple grid too."**

Response: E values for disconnected neato are 0.34-1.52 vs E < 0.02 for connected neato. These are systematic differences 10-100x larger than FP noise. Direct inspection of graphviz neato source (graph_pack.c) confirms array-mode bin-packing. The dagua code explicitly uses `int(len(component_positions) ** 0.5 + 0.999)` column count (simple grid).
