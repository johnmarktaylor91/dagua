# r76-F1 Superior-Distinct Fairness Probe

Date: 2026-07-03
Repo: `/home/jtaylor/projects/dagua`, branch `develop`
Scope: research/probe only. No repo code changed except this findings file.

## Executive verdict

The task text says r75 had 79 `quality_superior_distinct=true` rows, but the on-disk
files currently contain 89 in both `r75_final.jsonl` and `per_combo_r75.jsonl`. I
audited all 89.

Superior-distinct is not uniformly fair. Buckets:

| Bucket | Rows | Engines | Verdict |
|---|---:|---|---|
| Reference-bug / stale-reference regen needed | 8 | UMAP 4, GEM 2, FMMM 2 | Not acceptable as superior-distinct. r76 evidence already flips GEM/FMMM; UMAP tiny multiedge refs must be regenerated/rescored with the N=3 adapter fix. |
| Reference-param-noop / not same-params fair | 13 | SFDP | Graphviz SFDP refs are bit-identical across theta/steps/repulsive variants on several connected graphs while still seed-sensitive. Do not accept variant-specific superiority for those rows. |
| Fair but portable worse-reference behavior | 54 | Sugiyama 45, SFDP 9 | Honest refs, but the worse behavior is a concrete reference algorithm/tie/packing behavior that dagua can plausibly port under fidelity mode. |
| Fair, non-portable or low-ROI superior-distinct | 11 | MDS 6, Maxent 2, FMMM/fdp 1, UMAP random-dag 1, MDS disconnected 1 | Accept only with caveats: mostly eigenspace/floating-library or low-ROI emergent behavior. |
| Reclassify / margin-direction artifact | 3 | MDS disconnected | Mixed direction or inside-margin stress with crossing-only advantage; should not remain a clean "superior" label without rescore review. |

ROI order:

1. Regenerate/rescore known reference-bug rows: UMAP `parallel_multiedge_bundle::*`
   with the N=3 fix, GEM `iters100`, and FMMM `deep_chain_20` rows. GEM/FMMM r76
   rescores already show closure.
2. Add integrity gate: per graph/reference-family, same-seed positions must not be
   bit-identical across parameter variants unless the variant is explicitly declared
   non-expressible/no-op. This catches the SFDP connected variant issue and the old UMAP
   tiny fallback.
3. Finish Sugiyama Graphviz/igraph parity work before granting "superior-distinct":
   these 45 rows are the highest portable cluster.
4. Continue SFDP connected/disconnected parity work: graphviz neighbor-order,
   component orchestration, and weighted handling are named portable causes.
5. Leave MDS/UMAP eigenspace-like rows as low-ROI fairness-audited superior-distinct
   only after ledger text states the non-portable basis-selection cause.

## Commands run

```bash
pwd && git status --short --branch
sed -n '1,240p' .project-context/research/sprint_rng_matching/r76_final_sprint_STATE.md
ls -lh eval_output/fidelity_definitive
wc -l eval_output/fidelity_definitive/r75_final.jsonl eval_output/fidelity_definitive/per_combo_r75.jsonl
python - <<'PY'  # counted quality_superior_distinct rows and summarized fields
python - <<'PY'  # compared r75 flagged rows against r76_gem/fmmm/umap rescores
python - <<'PY'  # overlay tensor probe: same-seed param identity and seed response
python - <<'PY'  # graph metadata: nodes, edges, components, multiedges, self-loops
sed -n '850,1188p' dagua/eval/variants.py
sed -n '1580,1716p' dagua/eval/variants.py
sed -n '1,320p' dagua/eval/competitors/{umap_competitor.py,ogdf_competitor.py,graphviz_competitor.py,igraph_competitor.py}
```

Overlay order used, newest wins per combo:
`escalation_final`, `seeded_refs`, `drlref_realfix`, `umap_realfix`,
`gem_realfix`, `r72_fixes`, `fmmm_r3`, `fdp_fix`, `r73_fixes`, `r75_fixes`,
`r75_mds_topup`, `r75_topup2`, `r76_refs`, `r76_gem_fix`, `r76_umap_refs`,
`r76_umap_refs2`.

## Adapter sanity findings

| Reference | Adapter path | Param parity | Seed response | Degenerate/fallback risk |
|---|---|---|---|---|
| `umap_graph` | `umap_competitor.py` | Variant forwards `n_neighbors`, `min_dist`, `spread`, etc.; `n_neighbors` clamps to `N-1`. | Fresh refs respond to seed. | r75 bug: N<=3 returned seeded `torch.randn`; current develop fixed to N<=2. This invalidates r75 `parallel_multiedge_bundle` superiority. |
| `ogdf_gem` | `ogdf_competitor.py` -> `scripts/ogdf_runner` | `rounds`/`max_iters` -> `gemRounds`. | Fresh refs respond to seeds and rounds. | r75 stale runner ignored iteration params; r76 GEM rescore flips the flagged rows to equivalent. |
| `ogdf_fmmm` | same | `fixed_iterations`/`steps` -> `fmmmFixedIterations`. | Fresh refs respond to seeds and steps. | r75 stale-runner suspicion confirmed by deep-chain closure in r76. |
| `ogdf_stress` | same | `iterations`/`steps` -> `iterations`. | Fresh refs respond to seeds and iterations. | No param-noop found in tensor probe. |
| `graphviz_sfdp` | `graphviz_competitor.py` | Adapter passes `seed`, `start`, and `-G` variant attrs. | Responds to seed. | On connected flagged graphs, many same-seed refs are bit-identical across all six variants: param-noop alarm. |
| `graphviz_dot` | same | Only one flagged r75 Graphviz-fidelity variant. | Deterministic; no seed response expected. | No oracle fallback found; superiority is algorithm/tie/coordinate mismatch. |
| `igraph_sugiyama` | `igraph_competitor.py` | Forwards `maxiter`, `vgap`, `hgap`; seeds igraph RNG. | Deterministic or tie-sensitive; no seeded tensors in overlay for these rows. | No fallback found. Scale params can be normalized away by scoring, so identical metrics are not alone a bug. |
| `igraph_mds` | same | No material params; default and fidelity share the same reference. | Connected MDS deterministic; disconnected layouts seed-sensitive in overlay. | No fallback found. Connected rows are likely eigenspace/library-basis class rather than oracle bug. |

## Tensor oracle probes

Same-seed param identity means positions were byte-identical across sibling reference variants
for the same graph and seed. This is an alarm when the variant claims a changed parameter.

| Reference family / graph | Variants compared | Files | Same-seed param identity | Seed identity | Interpretation |
|---|---:|---:|---:|---:|---|
| `graphviz_sfdp` / `hexagonal_lattice_42` | 6 | 252 | 630/630 | 0/9 per variant | Params no-op; seeds honored. |
| `graphviz_sfdp` / `real_karate_34` | 6 | 252 | 630/630 | 0/9 per variant | Params no-op; seeds honored. |
| `graphviz_sfdp` / `weighted_karate_34` | 6 | 252 | 630/630 | 0/9 per variant | Params no-op; seeds honored. |
| `graphviz_sfdp` / `sparse_pair_50` | 6 | 252 | 630/630 | 0/9 per variant | Params no-op; seeds honored. |
| `graphviz_sfdp` / `random_dag_200` | 6 | 252 | 0/630 | 0/9 per variant | Params and seeds honored; disconnected issue is real algorithm/packing behavior. |
| `umap_graph` / `parallel_multiedge_bundle` | 6 | 600 | 300/1500 | 0/9 per variant | Fresh N=3 refs no longer globally degenerate; exact pairs are expected clamp pairs such as default vs nn30 where `n_neighbors` clamps to 2. r75 rows still invalid because old refs used random fallback. |
| `umap_graph` / `random_dag_50` | 6 | 600 | 0/1500 | 0/9 per variant | Params and seeds honored. |
| `ogdf_fmmm` / `deep_chain_20` | 3 | 300 | 0/300 | 0/9 per variant | Fresh refs are sane; r75 rows were stale-reference artifacts. |
| `ogdf_gem` / regular/triangular | 3 each | 300 each | 0/300 | 0/9 per variant | Fresh refs are sane; r75 rows were stale-reference artifacts. |
| `ogdf_stress` / `random_dag_50` | 8 | 336 | 0/1176 | 0/9 per variant | Fresh refs are sane. |
| `igraph_mds` connected graphs | 2 | 84 each | 42/42 | 9/9 | Expected: same reference, deterministic. |
| `igraph_mds` disconnected graphs | 2 | 84 each | 0/42 for `random_dag_200`; 42/42 for `multi_component_80` | mixed | Disconnected packing/initialization affects seed sensitivity; not an adapter fallback. |

## Row clusters

### UMAP: 5 rows

| Cluster | Rows | Magnitude | Oracle sanity | Disposition |
|---|---:|---|---|---|
| `parallel_multiedge_bundle::{default,mindist001,nn30,spread2}` | 4 | r75 stress D 0.032-0.041 vs R 0.119; delta 13-15x margin | Reference was the known N=3 random fallback in r75. Fresh develop fixed cutoff to N<=2; fresh tensors now respond to params except clamp-equivalent pairs. | `reference-bug`: regenerate/rescore. Do not accept r75 superiority. |
| `random_dag_50::classic_umap_nn5` | 1 | r75 stress D 0.582 vs R 0.746, delta 4.4x margin; crossing within margin | Params/seeds honored; disconnected graph with 52 components. r76 notes name spectral/eigenspace basis as the UMAP residual class. | Fair superior-distinct if ledger documents non-portable spectral basis/chaos class; low ROI to port. |

### OGDF GEM: 2 rows

| Rows | r75 magnitude | r76 rescore | Disposition |
|---|---|---|---|
| `regular_4_40::classic_gem_iters100`, `triangular_lattice_36::classic_gem_iters100` | Stress deltas 2.9x and 14.8x margin; crossings equivalent by margin | Both now `quality_equivalent_raw=true`, stress/cross effectively equal in `r76_gem_rescore.jsonl`. | `reference-bug`: stale OGDF runner. Already fixed/regenerated; remove superior label. |

### OGDF FMMM / Graphviz FDP: 4 rows

| Cluster | Rows | Magnitude | Oracle sanity | Disposition |
|---|---:|---|---|---|
| `deep_chain_20::{steps100,steps200}` | 2 | r75 stress D ~0.001 vs R 0.0275, about 19x margin | Fresh OGDF FMMM tensors respond to steps/seeds. `r76_fmmm_rescore.jsonl` makes both rows `quality_identical_raw=true`. | `reference-bug`: stale OGDF runner; regen already proves closure. |
| `random_dag_200::classic_fmmm_steps10` | 1 | r75 crossing D 8211 vs R 9082, 1.8x margin; stress within margin | Fresh params/seeds honored; disconnected 202-component graph. r76 rescore no longer superior and D stress is worse. | `reclassify` after r76 rescore; not a clean superior row. |
| `extreme_mixed_width_transformer::classic_fmmm_graphviz_fdp_fidelity` | 1 | Stress D 0.089 vs R 0.124, 5.5x margin; crossing D worse but inside margin | Graphviz FDP seed responds; no param variant. | Fair but low-ROI. Keep superior-distinct only if accepting emergent FDP behavior as non-portable; otherwise port Graphviz FDP packing/force details. |

### Graphviz SFDP: 22 rows

| Cluster | Rows | Magnitude | Oracle sanity | Disposition |
|---|---:|---|---|---|
| Connected base-param rows: `hexagonal_lattice_42::{default,graphviz_fidelity}`, `real_karate_34::graphviz_fidelity`, `weighted_karate_34::{default,graphviz_fidelity}` | 5 | Stress deltas 2.2-5.6x margin; crossing deltas mostly near 1.0-4.7x margin | Same params are fair for default/graphviz-fidelity. Seeds honored. | `port-the-worse-behavior`: r76 bisection named Graphviz symmetrized-CSR neighbor order and weighted residual handling. |
| Connected variant rows: `hexagonal_lattice_42::{p_neg2,theta04,theta08,steps200}`, `real_karate_34::{p_neg2,theta04,theta08,steps200}`, `weighted_karate_34::{p_neg2,theta04,theta08,steps200}`, `sparse_pair_50::p_neg2` | 13 | Stress deltas 2.1-5.6x margin; crossing often just over margin | Reference tensors are bit-identical across all six SFDP variants on these graphs. This is not same-params fairness for the variant rows. | `reference-param-noop`: do not accept as fair superior-distinct. Ledger should collapse to base behavior or mark reference non-expressible/no-op. |
| Disconnected `random_dag_200::{default,graphviz_fidelity,p_neg2,theta08}` | 4 | Crossing deltas 347-652 over margins 321-362; stress favors reference or is inside margin | Params/seeds honored; disconnected 202-component graph. | `port-the-worse-behavior`: r76 notes name component orchestration, shared RNG stream, and `packSubgraphs` as portable Graphviz behavior. |

### Sugiyama: 45 rows

| Reference | Rows | Representative magnitude | Oracle sanity | Disposition |
|---|---:|---|---|---|
| Graphviz dot fidelity | 18 | Stress deltas range hairline 1.1x margin to large 12x margin; crossings often equal or D better. Several rows are battery-only superiority. | Adapter has one flagged variant; deterministic reference. No fallback found. Graph classes include self-loop/multiedge cases (`kitchen_sink_*`, `nested_cluster_label_stack`) but Graphviz dot accepts them. | `port-the-worse-behavior`: rank/mincross/position-stage parity work, not acceptable as final superior-distinct while A1/A4 remain open. |
| igraph Sugiyama | 27 | Crossing wins are often large: e.g. `planar_60` +83/+126 crossings, `weighted_clusters_3x10` +56, karate +25. Stress deltas usually 1.2-4.3x margin. | Params are forwarded (`maxiter`, `vgap`, `hgap`). No degenerate adapter path found. Identical metrics across wide/tight/default can be scoring-scale normalization or stable order, not an oracle bug by itself. | `port-the-worse-behavior`: igraph tie/order/rank quirks are portable enough to pursue; r76 A3 notes already name GLPK/HiGHS/tie/qsort classes. |

Row list: Graphviz dot rows are
`asymmetric_hourglass_hub`, `bipartite_4_3_4`, `edge_label_braid`,
`hierarchical_residual_stage`, `hub_fanout_label_skew`,
`interleaved_cluster_crosstalk`, `kitchen_sink_hybrid_net`,
`kitchen_sink_platform_graph`, `long_skip_only_24`, `moe_router_sparse`,
`multiscale_skip_cascade`, `nested_cluster_label_stack`, `org_chart_1_5_4_8`,
`real_karate_34`, `residual_block`, `transformer_full_4h_2l`,
`transformer_layer`, `width_skew_late_merge`.

igraph rows are `heavy_tail_weights_50::passes4`, `hexagonal_lattice_42` x5,
`interleaved_cluster_crosstalk` x5, `planar_60::{passes4,passes48}`,
`real_karate_34::{default,tight,wide}`, `regular_3_30::passes4`,
`regular_4_40::passes4`, `weighted_clusters_3x10::passes4`,
`weighted_karate_34::{default,tight,wide}`, and `width_skew_late_merge` x5.

### MDS: 9 rows

| Cluster | Rows | Magnitude | Oracle sanity | Disposition |
|---|---:|---|---|---|
| Connected/self-loop: `center_port_backedge_hub`, `densenet_block` x2, `petersen_10`, `wide_single_layer_1_50_1` x2 | 6 | Stress deltas 2.6-8.3x margin; crossing wins on self-loop/wide rows | `igraph_mds` has no material params; connected refs are deterministic, as expected. | Fair superior-distinct if ledger documents non-portable LAPACK/eigenspace/library basis class. JMT already ruled no vendoring. |
| Disconnected/packing: `multi_component_80` x2, `random_dag_200::default` | 3 | `multi_component_80` stress delta is inside margin; `random_dag_200` stress favors reference while crossings favor dagua | Ref seed behavior is mixed on disconnected graphs; no fallback. | `reclassify` or split by leg. These are not clean superior-distinct rows. |

### OGDF Stress / Maxent: 2 rows

| Rows | Magnitude | Oracle sanity | Disposition |
|---|---|---|---|
| `random_dag_50::{classic_maxent_stress_default,classic_maxent_stress_steps50}` | Stress D 0.262 vs R 0.334-0.340, 4.3-4.6x margin; crossing equivalent | Fresh `ogdf_stress` refs respond to iterations and seeds. Graph is disconnected with 52 components. | Fair but low ROI. Keep superior-distinct only with note that worse behavior is emergent OGDF stress/disconnected packing; porting it is possible but not worth prioritizing over Sugiyama/SFDP. |

## Recommendations

1. Remove or override r75 superior labels for known-bug rows:
   `parallel_multiedge_bundle::classic_umap_{default,mindist001,nn30,spread2}`,
   `regular_4_40::classic_gem_iters100`,
   `triangular_lattice_36::classic_gem_iters100`,
   `deep_chain_20::classic_fmmm_{steps100,steps200}`.
2. Add an oracle invariant to `validate_benchmark_integrity.py`: for any reference
   family with multiple parameter variants on the same graph and seed, require at least
   one non-identical output unless the variant is explicitly listed as non-expressible or
   clamp-equivalent. This should hard-fail the SFDP connected variant pattern.
3. Do not final-ledger the 45 Sugiyama rows as superior-distinct until the r76 Sugiyama
   parity attempts are either merged or parked with named unportable causes.
4. Do not final-ledger SFDP variant rows as fair when their references are bit-identical
   across supposedly different params. Base/default SFDP rows can remain fair only if
   paired with the r76 bisection notes and ROI decision.
5. For MDS and UMAP random-dag rows, use a narrow "fair, non-portable basis/packing
   difference" label rather than a broad quality win. The direction is metric-dependent
   on disconnected graphs.

## Knowledge to carry forward

- `quality_superior_distinct` count in current r75 files is 89, not 79.
- Fresh UMAP N=3 references in `benchmark_100seed_r76_umap_refs2` are no longer globally
  param-degenerate; default and `nn30` remain legitimately clamp-equivalent because
  `n_neighbors` becomes `N-1=2`.
- Graphviz SFDP can be seed-sensitive while still parameter-insensitive on connected
  graphs. Seed response alone is not an adequate oracle sanity check.
- r76 GEM and FMMM rescores already prove several r75 superior rows were stale-reference
  artifacts.
