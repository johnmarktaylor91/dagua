<task>
Build a NEW layout-equivalence-metrics module for dagua's fidelity eval system. Project root:
/home/jtaylor/projects/dagua (python 3.11 env at ~/anaconda3/envs/py311; activate via that python).

## Why (background -- read fully)
dagua reimplements ~24 graph-layout algorithms and benchmarks them against reference engines
(igraph/graphviz/OGDF/networkx/...). Fidelity is currently measured by per-seed Procrustes RMSD of
node coordinates vs the reference. For most engines that works. But a class of DETERMINISTIC engines
and SYMMETRIC-graph cases show HIGH Procrustes RMSD while being GEOMETRICALLY / STRUCTURALLY EQUIVALENT
to the reference -- the RMSD is a metric artifact, not a real difference:
  - sugiyama on symmetric graphs (complete5/petersen_10/wheel7): the reference picks one of several
    EQUALLY-VALID within-layer orderings; dagua picks another. The two differ by a GRAPH AUTOMORPHISM
    (a relabeling that maps the graph to itself).
  - classical_mds / pivot_mds / spectral on graphs with REPEATED (degenerate) eigenvalues: any
    orthonormal basis of the degenerate eigenspace is equally correct; igraph's vendored LAPACK picks
    one, dagua another. The layouts differ by a rotation within the degenerate subspace.
These are not fidelity failures -- both are correct layouts. We need metrics that SHOW practical
equivalence where coordinate-RMSD cannot. JMT approved building all three below.

## Deliverable
A new module `dagua/eval/equivalence_metrics.py` + a CLI report script
`scripts/equivalence_report.py` + unit tests `tests/test_eval/test_equivalence_metrics.py`.
Compute, per (graph, engine, seed) pair of reimpl-vs-reference positions, THREE equivalence signals:

### 1. Automorphism-aligned Procrustes  (the rigorous proof for discrete tie-breaks)
- Input: reimpl positions P_d (N x 2 tensor/array), reference positions P_r (N x 2), graph edge_index.
- Compute the graph's AUTOMORPHISM GROUP via igraph: build an igraph.Graph from edge_index (undirected
  unless the layout is inherently directed -- default undirected for automorphism purposes), call
  `g.get_automorphisms_vf2()` (returns a list of permutations, each a length-N list mapping node->image).
- For EACH automorphism perm (including identity): permute the REFERENCE rows by perm, then compute
  standard Procrustes RMSD(P_d, perm(P_r)) WITH reflection allowed (full orthogonal Procrustes:
  translation + uniform scale + rotation + reflection removed). Reuse the Procrustes routine in
  scripts/fast_fidelity_report.py (import or copy its aligned-RMSD function -- match its normalization
  exactly so numbers are comparable to the existing report).
- Output: `aut_procrustes_rmsd` = MIN over all automorphisms; also report `plain_procrustes_rmsd`
  (identity only, = the existing metric) and `aut_group_size`.
- GUARD (do not blow up on large symmetric graphs): if |Aut| would be huge, cap enumeration. Use
  MAX_AUT (default 20000). If get_automorphisms_vf2 returns more (or to avoid enumerating), fall back to
  the generators from `g.automorphism_group()` (BLISS) and either (a) BFS-generate the group up to
  MAX_AUT elements then stop, or (b) if still capped, evaluate only the generator set + identity and
  flag `aut_capped=True`. Document the choice in a docstring. The cases that MATTER are small symmetric
  graphs (|Aut| <= ~120 for petersen/complete5/wheel), so the cap rarely triggers.

### 2. Stress + quality equivalence  (the broad practical verdict)
- Normalized stress: sigma = sum_{i<j} w_ij (||x_i - x_j|| - d_ij)^2 with w_ij = 1/d_ij^2 and d_ij =
  graph-theoretic shortest-path distance (APSP). Use a STANDARD normalization (e.g. divide by
  sum_{i<j} w_ij d_ij^2, or the normalized-stress form already used in dagua's eval -- SEARCH
  dagua/eval for an existing stress metric and REUSE it; only implement fresh if none exists). Compute
  stress(P_d) and stress(P_r); report both + `stress_rel_delta` = |s_d - s_r| / max(s_r, eps).
- Edge crossings: count crossing edge pairs for each layout; report counts + delta. REUSE an existing
  crossing-count metric in dagua/eval if present.
- Neighborhood preservation (trustworthiness): for each node, compare its k nearest neighbors in LAYOUT
  space vs its k nearest by GRAPH distance; report mean overlap for P_d and P_r + delta. k default 10
  (or min(10, N-1)). REUSE if an existing metric exists.
- APSP distances: reuse dagua/layout/ops/graph_utils.py (it has BFS/Dijkstra/APSP) or igraph
  `shortest_paths`. Do NOT hand-roll if a util exists.

### 3. Spectrum / distance diagnostic  (proves it IS basis-choice, for eigendecomposition engines)
- Pairwise-distance-matrix agreement (rotation/reflection-invariant): D_d = pairwise euclidean distance
  matrix of P_d, D_r = same for P_r. Report `dist_matrix_corr` = Pearson corr of the upper triangles,
  and `dist_matrix_rel_frob` = ||D_d - D_r||_F / ||D_r||_F.
- Gram-eigenvalue match (shape, basis-invariant): center each layout (subtract mean), G = X_c @ X_c.T;
  the SORTED eigenvalues of G are rotation/reflection invariant and capture the layout's shape. Report
  `gram_eig_max_absdiff` = max abs difference of sorted eigenvalues of G_d vs G_r (after matching scale,
  e.g. normalize each layout to unit Frobenius norm first). For a pure degenerate-subspace rotation
  these match to ~1e-10.

### Combined verdict (per pair) + aggregation
- Per pair, emit a verdict: PRACTICALLY_EQUIVALENT if ANY of:
    aut_procrustes_rmsd < 1e-3, OR (dist_matrix_corr > 0.999 AND gram_eig_max_absdiff < 1e-3),
    OR (stress_rel_delta < 0.02 AND neighborhood_preservation_delta < 0.02).
  Else NOT_EQUIVALENT. Always emit ALL raw signals (don't hide them) so a human can judge.
- scripts/equivalence_report.py: CLI taking --results <results.json> --positions <positions.h5 or
  positions/ dir> [--combos <json list of {graph,engine} or auto-read the non-bit-exact set from a
  fidelity report per_variant.json>] --graphs-from <benchmark graph registry/loader> --output <dir>.
  It loads each (graph,engine,seed) reimpl+reference position pair, computes the trio, writes
  equivalence_report.md (table: graph, engine, plain_rmsd, aut_rmsd, aut_group, stress_delta,
  dist_corr, gram_eig_diff, verdict) + equivalence.json. Focus on the deterministic/symmetric holdouts
  (classic_sugiyama_*, classic_classical_mds_*, classic_pivot_mds_*, classic_spectral_random_walk) but
  work for any combo.
</task>

<constraints>
- A 5-SEED BENCHMARK IS CURRENTLY RUNNING (pid 3453115, writing eval_output/benchmark_5seed_final).
  DO NOT modify dagua/eval/run_benchmark.py, dagua/eval/competitors/*, dagua/eval/variants.py,
  dagua/eval/benchmark.py, or anything the running benchmark imports. NEW files only
  (dagua/eval/equivalence_metrics.py, scripts/equivalence_report.py, the test file). You MAY import
  read-only from existing modules. Reading benchmark OUTPUTS (results.json, positions/*.pt) is fine.
- NO DELEGATION: compute purely on stored positions + graph structure. Do NOT call/import reference
  layout engines to GENERATE layouts (igraph for AUTOMORPHISM-GROUP computation is fine -- that's graph
  analysis, not layout delegation; igraph shortest_paths is fine too). See the project rule: reimpl
  pipelines never delegate, but this is an ANALYSIS module so igraph-for-graph-analysis is allowed.
- Reuse existing utilities (Procrustes from scripts/fast_fidelity_report.py; stress/crossings from
  dagua/eval if present; APSP from dagua/layout/ops/graph_utils.py). Grep before writing fresh.
- Positions may be stored as positions.h5 (h5py) OR as per-run .pt files in positions/. Support BOTH
  (mirror how scripts/fast_fidelity_report.py and scripts/consolidate_positions_hdf5.py read them).
- float64 throughout for the metrics. Do NOT commit (CC reviews + commits).
</constraints>

<verification>
- pytest tests/test_eval/test_equivalence_metrics.py -x -q  (all pass).
  Tests MUST include: (a) petersen_10 or complete5 with two layouts differing by a known automorphism
  -> aut_procrustes_rmsd ~0 while plain_procrustes_rmsd is large; (b) a layout vs its own rotation ->
  dist_matrix_corr ~1, gram_eig_max_absdiff ~0; (c) stress computed correctly on a trivial known layout;
  (d) identity case (P_d == P_r) -> all signals show perfect equivalence.
- Run scripts/equivalence_report.py on the live benchmark data for the holdout combos and REPORT the
  before/after: e.g. classic_sugiyama_default on petersen_10/complete5 -- show plain_rmsd (~0.37-0.93)
  vs aut_rmsd (expect a large drop), and classic_classical_mds_default -- show dist_matrix_corr +
  gram_eig match despite high plain_rmsd. Paste the actual numbers.
  (Use: export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:$LD_LIBRARY_PATH)
- ruff check on the new files passes; mypy --follow-imports=silent on the new module passes.
</verification>

<default_follow_through_policy>
Proceed autonomously with the most reasonable low-risk interpretation. Stop only for: genuine
correctness ambiguity you cannot resolve from the code, a need to modify a running-benchmark file
(forbidden -- find another way), or an irreversible action. If an existing stress/crossing metric is
found, prefer reusing it (note which). If the automorphism group is intractable for some graph, apply
the documented cap and flag it rather than failing the whole run. Report what you reused vs wrote fresh,
the verification numbers (the sugiyama/classical_mds before/after is the key result), and any choices.
</default_follow_through_policy>

<!-- =========================================================================================
FOLLOW-UP ADDITIONS (JMT 2026-06-03) -- DISPATCH ONLY AFTER THE TRIO MODULE ABOVE IS COMMITTED.
Two more invariances complete the criteria set (the principled ceiling: rigid + automorphism +
degenerate-eigenspace + per-component + per-axis-opt-in; anything further launders real differences).
This is a SEPARATE codex task editing the SAME module -- never run it concurrently with the trio build.

### Addition 4 -- Per-connected-component rigid placement (a legitimate exact invariance)
When a graph has >1 connected component, the engines pack/place components arbitrarily, so a SINGLE
global rigid transform cannot align them even when each component is drawn identically. Add a
`component_aligned_rmsd`:
- Decompose the graph into connected components (igraph `.connected_components()` / `.clusters()`).
- Align EACH component independently: per-component rotation + reflection + translation, but a SINGLE
  GLOBAL uniform scale shared across components (packing arbitrariness is placement+orientation, not
  size -- the engine computes each component at its natural scale then packs). Reuse the Procrustes
  routine restricted to each component's node subset.
- Aggregate: report `component_aligned_rmsd` = RMSD pooled across all components after per-component
  alignment, plus `n_components`. For a connected graph (1 component) it equals the global Procrustes
  RMSD (no-op) -- so it only changes anything for disconnected graphs / forests (and is part of the
  fmmm component-packing wall).
- Report it as its OWN signal alongside aut_procrustes_rmsd (do not try to compose automorphism x
  component in one search -- keep them separate signals; the verdict accepts EITHER).

### Addition 5 -- Per-axis (anisotropic) scaling -- OPT-IN per engine ONLY
Some engines have a genuinely FREE aspect ratio (independent x/y spacing). For THOSE engines only,
allow independent scale_x, scale_y in the alignment. Applied globally this is over-permissive (an
x-vs-y stretch can change crossings), so it is GATED by a per-engine allowlist.
- Implement `anisotropic_rmsd`: align rotation+reflection+translation, then fit per-axis scales
  (scale_x, scale_y) by least squares, then RMSD. (Anisotropic Procrustes.)
- ALLOWLIST constant `FREE_ASPECT_ENGINES` (documented, conservative, default = {"classic_sugiyama"}
  -- sugiyama's layer-spacing vs within-layer-ordering axes are semantically independent and
  independently scaled). Extensible WITH JUSTIFICATION; default OFF for every other engine.
- For engines NOT in the allowlist, do NOT compute/use anisotropic_rmsd (leave it null) -- granting an
  invariance an engine does not actually have would HIDE real bugs.

### Verdict update
Extend the PRACTICALLY_EQUIVALENT disjunction to also pass if:
  component_aligned_rmsd < 1e-3, OR (engine in FREE_ASPECT_ENGINES AND anisotropic_rmsd < 1e-3).
Keep emitting all raw signals.

### Tests (add)
- A 2-component graph whose components are placed/oriented differently in P_r vs P_d but each drawn
  identically -> component_aligned_rmsd ~0 while plain/global RMSD is large.
- A sugiyama-like layout vs the same stretched independently in x and y -> anisotropic_rmsd ~0 while
  plain RMSD is large; AND confirm a NON-allowlisted engine does NOT get the anisotropic pass.

### Constraints (same as the trio task)
- New code in the SAME module dagua/eval/equivalence_metrics.py + report script + tests. Do NOT touch
  run_benchmark/competitors/variants (the 5-seed benchmark may STILL be running). No layout delegation
  (igraph for component-decomposition/automorphism analysis is fine). float64. Do NOT commit (CC commits).
- VERIFY: rerun scripts/equivalence_report.py on the holdouts; report which combos now flip to
  PRACTICALLY_EQUIVALENT via component or anisotropic alignment (expect sugiyama to benefit from
  anisotropic; disconnected/forest graphs + fmmm from per-component).
========================================================================================= -->
