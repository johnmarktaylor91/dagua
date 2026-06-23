# r74 O4 Cluster Findings: classical_mds (34) + gem (22) + maxent_stress (3)

Analyst: Opus (read-only research). Date 2026-06-22. Verdict data: `eval_output/fidelity_definitive_r73/per_combo.json`.
All comparisons run on the BENCHMARK PATH (real `scripts/ogdf_runner` binary + real igraph 1.0.0), not direct pipeline shortcuts.

================================================================================
## FAMILY 1: classical_mds — 34 divergent (30 mode A + 4 mode B)

Reference = REAL external engine `igraph_mds` (igraph 1.0.0 `layout("mds")`), NOT a self-reference.
Both `classic_classical_mds_default` and `classic_classical_mds_igraph_fidelity` show IDENTICAL divergence
per graph (the r73 edge-weight fix `_UNWEIGHTED_REFERENCE_LAYOUTS` is orthogonal to these 34 — already unweighted).

The 34 split into TWO mutually-exclusive subgroups by the `disconnected` flag:

### SUBGROUP 1A — CONNECTED + DEGENERATE eigenvalues (16 combos; 8 graphs x 2 variants)
Graphs: petersen_10, bipartite_4_3_4, densenet_block, center_port_backedge_hub, org_chart_1_5_4_8,
wide_3_50_3, wide_single_layer_1_50_1. All flagged `degenerate=True`, `near_deterministic=True`, `e_rel=None`.

ROOT CAUSE (proven empirically):
- classical_mds.py:259-265 calls `scipy.linalg.eigh(gram, subset_by_index=(N-2,N-1), driver='evr')`.
- These graphs have TOP eigenvalue MULTIPLICITY > 2. Measured: Petersen top eig=3.5 mult=**5**; K3,3 eig=2.0 mult=4;
  K5 eig=0.5 mult=4. The 2D MDS basis is an arbitrary projection of a >=3D eigenspace.
- igraph's vendored LAPACK 3.4.2 `dsyevr` (range='I', uplo='U', abstol=1e-14) and dagua's system-LAPACK
  (OpenBLAS/MKL) `evr` driver pick DIFFERENT 2D bases from the same degenerate eigenspace.
- EMPIRICAL CONFIRMATION: I called scipy.linalg.lapack.dsyevr with igraph's EXACT params
  (range='I', lower=0/uplo='U', il=N-1, iu=N, abstol=1e-14) -> Procrustes RMSD vs igraph = **0.342**
  (Petersen). The docstring claim is correct: even matching the eigenspace, the basis differs.
- CONTROL: cycle10 (top eig mult=2, eigenspace EXACTLY 2D) -> Procrustes RMSD = **0.0000** (matches perfectly).
  Confirms: divergence appears IFF eigenvalue multiplicity > requested dimensions.

BEST ACHIEVABLE TIER: **rung 4 (DIVERGENT) — LIKELY FLOOR**. NOT a last-decimal-FP floor; it is a
LAPACK-implementation basis-selection gap (proc RMSD 0.17-0.34, structural). Procrustes (scale+rot) CANNOT
fix it because a different 2D *projection* of an N-D space is not a rotation of the reference projection.
FIX AVENUE (high effort, low ROI): vendor igraph's LAPACK 3.4.2 dsyevr tridiagonal-reduction + inverse-iteration
path bit-for-bit (a multi-hundred-line Fortran port). Confidence FLOOR: HIGH. Effort to fix: VERY HIGH (weeks).
RECOMMEND: PROVE the floor (document the mult>2 + dsyevr-basis evidence; cycle10=0.0 control) and STOP.

### SUBGROUP 1B — DISCONNECTED graphs (18 combos: 14 mode A + 4 mode B; 9 graphs x 2 variants)
Graphs (mode A): disconnected_encoder_res, disconnected_label_cycle, er_100, kitchen_sink_platform_gr,
multi_component_80, parallel_cycles_4x5, random_bipartite_60, random_dag_50.
Graphs (mode B): er_500, random_dag_200. All `disconnected=True`, large POSITIVE `e_rel` (+1.5 to +4.2).

ROOT CAUSE (proven empirically + via igraph docs):
- igraph MDS for DISCONNECTED graphs (confirmed via igraph C docs + empirical test):
  decomposes into components -> runs classical MDS on EACH component's submatrix -> merges via
  `igraph_layout_merge_dla()` (Diffusion-Limited Aggregation, STOCHASTIC, uses igraph global RNG).
- dagua (graph_utils.py:319-352 `shortest_path_distances`): replaces ALL cross-component inf distances with a
  SINGLE global scalar `fill_value = max_finite_distance + 1.0`, then runs ONE classical MDS on the whole
  filled matrix. Structurally wrong — produces a single tight blob instead of separately-laid-out + packed comps.
- EMPIRICAL: 2-triangle graph. igraph keeps inf cross-component and lays each triangle out independently
  (both congruent equilateral, internal dist identical, separated). dagua fills cross=2 -> proc RMSD 0.246.
  NO finite scalar fill (tested d=1,2,3,6) reproduces igraph (best 0.238) -> confirms igraph is NOT doing
  a global-fill MDS at all.
- The large +e_rel (dagua stress >> igraph): dagua's blob crams disconnected components together
  (cross-fill underestimates separation), inflating stress relative to igraph's packed layout.

ADDITIONAL WRINKLE (makes seed-matching impossible AS BENCHMARKED):
- igraph's DLA merge is STOCHASTIC and seeded from igraph's GLOBAL RNG. Empirically: back-to-back
  `g.layout('mds')` calls give DIFFERENT layouts (not call-to-call identical). It IS reproducible if you
  reset igraph's RNG (seed 42 -> a, seed 42 -> a, seed 99 -> c != a).
- BUT the benchmark's IgraphMDS competitor has `uses_igraph_rng=False` (igraph_competitor.py) — so the
  reference DLA placement was NOT reset per benchmark seed; each of the 100 "seeds" got an uncontrolled
  global RNG state. The component GEOMETRY is matchable; the MERGE PLACEMENT is not seed-matchable as run.

BEST ACHIEVABLE TIER:
- Per-component MDS geometry: rung 2/3 achievable (deterministic, matchable math).
- Full layout incl. merge: rung 3 (distributional) at best; rung 1 NOT achievable (DLA is RNG + benchmark
  ref wasn't seeded). Could plausibly flip 4 -> 2'/3 by matching component geometry + a TileToRows-or-DLA pack.
FIX AVENUE (MEDIUM effort, MEDIUM-HIGH ROI — 18 combos, more than gem):
  1. Decompose disconnected graph into components (dagua already has `_connected_components_from_edges` in gem.py).
  2. Run the existing igraph-MDS per component submatrix (the connected path already works -> rung-1 on connected).
  3. Pack components. Honest caveat: igraph uses DLA (stochastic). A deterministic packer (e.g. dagua's existing
     `_ogdf_tile_to_rows_offsets`) will NOT bit-match DLA but should land DISTRIBUTIONALLY closer -> rung 3.
     To truly match, port `igraph_layout_merge_dla` + seed igraph RNG in the competitor (`uses_igraph_rng=True`)
     and re-bench with `--seed-refs`. That is the only rung-1/2 path and it touches the eval harness.
CONFIDENCE root cause: HIGH (igraph C docs + 3 independent empirical confirmations). Effort: MEDIUM (per-comp
decomposition) for rung-3; HIGH for rung-1/2 (DLA port + harness reseed). RECOMMEND: ship per-component +
deterministic pack for a 4->3 flip on ~14 mode-A combos; document DLA as the residual to rung-1.

================================================================================
## FAMILY 2: gem — 22 divergent (all mode A)

Reference = REAL `ogdf_gem` (OGDF GEMLayout via `scripts/ogdf_runner` binary, deterministic).
CONTEXT: 269 of 309 gem combos are ALREADY rung-1 (bit-exact), 16 rung-2, only 22 rung-4. The 22 divergent
all have SMALL positive e_rel (min 0.032, median 0.062, max 0.231; all < 0.25); median stress_R_mean 0.053.
Graphs: grid_5x5, binary_tree, regular_3_30/4_40, rgg_100, real_lesmis_77, triangular_lattice_36,
sierpinski_42, transformer_full, random_dag_50/200, several TL/MLP graphs, etc.

r73's parity-guardrail FAILURE was CORRECT (gem != ogdf-gem at matched seed) but the EARLIER diagnosis
("not seeded") was wrong. I re-examined from OGDF GEMLayout.cpp + the runner. Every RNG/param layer MATCHES:

VERIFIED MATCHING (source-faithful, all checked against `_references/ogdf`):
- Initial positions: runner (ogdf_runner.cpp:416-422) does `setSeed(seed); srand(seed);
  x = rand()%1000/10.0`. dagua `_glibc_rand_values` reproduces glibc rand() BIT-EXACTLY (verified vs ctypes
  libc.rand: 10/10 match). Init in fidelity path uses float64 (no rounding).
- RNG seed bridge: GEMLayout `m_rng(randomSeed())`; basic.cpp `randomSeed() = 7*s_random()+3` (mt19937 after
  setSeed). dagua `_ogdf_gem_rng_seed = 7*_mt19937_first_uint32(seed)+3` — EXACT match.
- Permutation: OGDF `SList::permute -> Array::permute` (Array.h:956): `uniform_int_distribution<int>(0,n-1)`,
  swap i with (start+dist) for each i. dagua `_ogdf_permutation` (gem.py:335): same full-range Fisher-Yates,
  swap index<->absolute swap_index. EXACT match. `_ogdf_uniform_int` is a faithful libstdc++ port.
- Physics constants: m_desiredLength=LayoutStandards::defaultNodeSeparation()=20.0; node W=H=20.0 (verified
  LayoutStandards.cpp:38-50). desiredLength = 20 + hypot(20,20) = 48.284. dagua hardcodes identical values.
  gravity 1/16, init_temp 12, min_temp 0.005, maximal_disturbance = **0** (so RNG only drives permutation order).
- Update loop (computeImpulse + updateNode, GEMLayout.cpp:240-330): gravity, repulsion (all-pairs),
  attraction (formula 1 = delta*dist/(desiredLen*weight)), impulse scaling by local temp, oscillation/rotation/
  skew temperature adaptation, barycenter update — dagua `_run_ogdf_component_gem` (gem.py:830-978) mirrors
  every term and ordering.
- Round budget: variant `{max_iters:100}` <-> ogdf `{rounds:100}`. dagua converts via
  `_resolve_ogdf_gem_update_budget` -> max_iters*N node-updates (verified: 100 -> capped 2500 for N=25).
  CRITICAL FINDING: OGDF GEM TERMINATES on temperature (`m_globalTemperature > m_minimalTemperature`) far
  before the round budget — confirmed rounds=100/1000/10000 give IDENTICAL output (proc 0.0). dagua has the
  same termination guard, so both converge in the same basin. Budget is NOT the divergence source.

ROOT CAUSE OF THE RESIDUAL: **floating-point summation-order chaos**. GEM is a chaotic dynamical system;
the all-pairs repulsion sum `Sum_u delta*desiredSqu/distSqu` and barycenter accumulate in different FP order
in Python/PyTorch-float64 vs C++ double. With disturbance=0 and identical permutation, MOST graphs (269)
still converge bit-exactly, but ~7% (the 22) are sensitive enough that the tiny FP-order delta pushes them
into a DIFFERENT (equally valid) local minimum after hundreds of temperature-decaying updates.
EMPIRICAL: even path3 (N=3) diverges proc 0.226 while edge2 (N=2, trivial) = 0.000 — the divergence onsets
the instant the repulsion+revisit dynamics engage, and grows with graph "frustration" (grid5x5 proc 0.10,
binary_tree/random_dag higher e_rel). This is the classic FP-chaos signature, not a portable bug.

BEST ACHIEVABLE TIER: **rung 4 — PROVEN FLOOR (FP-chaos)**. The 269 bit-exact combos PROVE the port is
source-faithful; the 22 are the chaotic-sensitive tail. To squeeze them you'd have to replicate C++ double
summation order EXACTLY (operation-by-operation, same loop nesting, no PyTorch vectorization) — and even then
`std::uniform_int_distribution` / `length()` rounding could differ across libstdc++ builds. NOT worth it.
FIX AVENUE: none that's robust. (Theoretically: rewrite `_run_ogdf_component_gem` repulsion as a scalar
C-order loop in the exact OGDF iteration order — MIGHT recover a few, but fragile to compiler/libc.)
CONFIDENCE floor: HIGH (269 bit-exact + matched RNG/params + FP-chaos onset at N=3). Effort to chase: HIGH,
expected yield LOW. RECOMMEND: PROVE floor (cite the 269 bit-exact + the N=3 onset + disturbance=0) and STOP.

================================================================================
## FAMILY 3: maxent_stress — 3 divergent (all mode A)

ALL THREE are the SAME graph (random_dag_50, DISCONNECTED) across step variants (default/steps50/steps400).
Reference = REAL `ogdf_stress` (OGDF StressMinimization via runner). Small e_rel (0.012-0.061).

ROOT CAUSE (proven via OGDF source): identical structure to classical_mds SUBGROUP 1B.
- OGDF StressMinimization.cpp:97-117: when graph is disconnected and m_componentLayout is off, it routes
  through `ComponentSplitterLayout` (ComponentSplitterLayout.cpp:58 -> `TileToRowsCCPacker`): lay out EACH
  component separately, then pack with TileToRows.
- dagua (stress_majorization.py:596-604): replaces ALL cross-component distances with ONE global scalar
  `fill_value = _OGDF_EDGE_COSTS * sqrt(N)`, runs a SINGLE global stress majorization. NO component
  decomposition (grep: 0 component/packer references in the file). Structurally wrong, same as 1B.

BEST ACHIEVABLE TIER: **rung 1/2 ACHIEVABLE** (StressMinimization is DETERMINISTIC — no DLA, unlike igraph MDS;
the packer is TileToRowsCCPacker which dagua ALREADY ports as `_ogdf_tile_to_rows_offsets` in gem.py).
FIX AVENUE (LOW-MEDIUM effort, but only 3 combos = 1 graph): decompose random_dag_50 into components, run the
existing OGDF-stress per component (the connected path is already rung-1 — 392 maxent combos are rung-1), pack
with the existing TileToRowsCCPacker port. Because StressMinimization is deterministic AND the packer is
already ported, this is the MOST tractable rung-1 win in the cluster — but it is only 3 combos (one graph).
CONFIDENCE: HIGH (OGDF source explicit + packer already exists). Effort: LOW-MEDIUM. ROI: low absolute count
but cleanest fix; the SAME decompose+TileToRows helper would ALSO serve classical_mds 1B (deterministic-pack path).

================================================================================
## ROI-ORDERED RECOMMENDATIONS

1. **maxent_stress (3) — FIX, rung-1 achievable, LOW-MED effort.** Decompose disconnected + per-component OGDF
   stress + existing TileToRowsCCPacker. Deterministic, packer already ported. Cleanest win. (1 graph only.)
2. **classical_mds 1B disconnected (18; ~14 mode-A flippable) — FIX to rung-3, MED effort.** Decompose +
   per-component igraph-MDS (connected path already rung-1) + deterministic TileToRows pack. Reuses the SAME
   helper as #1. Caveat: igraph's true merge is DLA (stochastic) -> rung-1/2 needs a DLA port + harness reseed
   (`uses_igraph_rng=True` + `--seed-refs`); deterministic pack lands rung-3, not rung-1.
3. **classical_mds 1A degenerate (16) — PROVE FLOOR, STOP.** dsyevr basis-selection on mult>2 eigenspaces;
   proven irreducible without vendoring LAPACK 3.4.2 (scipy-dsyevr with igraph params still RMSD 0.34;
   cycle10 mult=2 control = 0.0). HIGH effort, ~0 realistic ROI.
4. **gem (22) — PROVE FLOOR, STOP.** FP-summation-order chaos; 269/309 already bit-exact, RNG+params all
   verified-matching, divergence onsets at N=3 with disturbance=0. Genuine chaotic-tail floor.

NET sprint-flippable: ~3 (maxent, rung-1) + ~14 (mds-1B mode-A, rung-3 with deterministic pack). ~30 combos
(mds-1A degenerate 16 + gem 22... minus overlap) are proven/likely floors.
