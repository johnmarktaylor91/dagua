# R41 Variant Parameter Semantic Equivalence Audit

Scope: audited every entry in `dagua/eval/variants.py` for whether the Dagua
variant parameter values have the same meaning as the configured reference-side
parameter values. Pipeline and adapter code were read only except for the
registry fix noted below.

## Verdict

- Clear mismatch fixed: `classic_tsnet_steps200`.
- Clear mismatches not fixed because adapter changes are out of scope:
  Graphviz adapters ignore `original_params`; OGDF GEM/FMMM reference adapters
  do not expose the Dagua variant iteration parameters.
- No pipeline algorithm code was changed.
- `results.json` was not touched.

## Adjustment Applied

| variant_id | before | after | reason |
| --- | --- | --- | --- |
| `classic_tsnet_steps200` | Dagua `steps=200`; reference `max_iter=200`, then adapter clamps to `250` | Dagua `steps=250`; reference `max_iter=250`; display label updated | `TSNEGraph` enforces sklearn's current minimum `max_iter >= 250`, so the reference side was already running 250 iterations. The Dagua side now matches the actual reference semantics. |

## Smoke Tests

Inline smoke probes were run without creating a harness file.

```text
igraph_sugiyama_forwarded {'algo': 'sugiyama', 'maxiter': 4, 'vgap': 1.0, 'hgap': 1.0} error None
dagua_sugiyama_trace_count 1 requested_barycenter_passes 4 y_span 2.0
tsne_forwarded_after_adapter {'n_components': 2, 'metric': 'precomputed', 'init': 'random', 'random_state': 7, 'perplexity': 3.0, 'method': 'exact', 'max_iter': 250} error None
```

Interpretation:

- Sugiyama: Dagua forwards `barycenter_passes=4` as a four-pass upper bound and
  igraph receives `maxiter=4`. In igraph fidelity mode Dagua also stops early
  when stable, so the actual pass count can be lower than the budget; that is a
  compatible max-iteration semantic rather than a 4 down + 4 up vs 4 total
  mismatch.
- t-SNE: the reference adapter converts `max_iter=200` to `250`; this was the
  only clear bit-exactness-affecting registry mismatch found and was fixed.

## Audit Table

| variant_id | dagua param dict | reference param dict | semantic equivalence |
| --- | --- | --- | --- |
| `classic_fr_steps50` | `{'steps': 50}` | `{'iterations': 50}` | `steps` -> `iterations`: identical outer FR iteration count. |
| `classic_fr_steps100` | `{'steps': 100}` | `{'iterations': 100}` | `steps` -> `iterations`: identical outer FR iteration count. |
| `classic_fr_steps200` | `{'steps': 200}` | `{'iterations': 200}` | `steps` -> `iterations`: identical outer FR iteration count. |
| `classic_fr_steps500` | `{'steps': 500}` | `{'iterations': 500}` | `steps` -> `iterations`: identical outer FR iteration count. |
| `classic_kk_steps100` | `{'steps': None, 'orient_to_direction': False}` | `{}` | No active count parameter: Dagua `steps=None` and NetworkX default are both unconstrained SciPy solve; `steps100/300/1000` labels are legacy labels only. |
| `classic_kk_steps300` | `{'steps': None, 'orient_to_direction': False}` | `{}` | No active count parameter: Dagua `steps=None` and NetworkX default are both unconstrained SciPy solve; `steps100/300/1000` labels are legacy labels only. |
| `classic_kk_steps1000` | `{'steps': None, 'orient_to_direction': False}` | `{}` | No active count parameter: Dagua `steps=None` and NetworkX default are both unconstrained SciPy solve; `steps100/300/1000` labels are legacy labels only. |
| `classic_fr_kk_default` | `{'first_steps': 50, 'second_steps': 300}` | `{}` | No original-side reference params; no semantic pair to validate. |
| `classic_fr_kk_long` | `{'first_steps': 100, 'second_steps': 300}` | `{}` | No original-side reference params; no semantic pair to validate. |
| `classic_kk_fr_default` | `{'first_steps': 300, 'second_steps': 50}` | `{}` | No original-side reference params; no semantic pair to validate. |
| `classic_kk_fr_long` | `{'first_steps': 300, 'second_steps': 100}` | `{}` | No original-side reference params; no semantic pair to validate. |
| `classic_fa2_default` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True}` | `steps`, `gravity`, `scaling_ratio`, `strong_gravity`, `outbound_attraction_distribution` map directly to FA2 reference names. |
| `classic_fa2_gravity0` | `{'steps': 200, 'gravity': 0.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 0.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True}` | Direct FA2 API equivalents. |
| `classic_fa2_gravity2` | `{'steps': 200, 'gravity': 2.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 2.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True}` | Direct FA2 API equivalents. |
| `classic_fa2_scaling1` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 1.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 1.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True}` | Direct FA2 API equivalents. |
| `classic_fa2_scaling4` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 4.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 4.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True}` | Direct FA2 API equivalents. |
| `classic_fa2_strong_gravity` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 2.0, 'strong_gravity': True, 'outbound_attraction_distribution': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 2.0, 'strongGravityMode': True, 'outboundAttractionDistribution': True}` | Direct FA2 API equivalents. |
| `classic_fa2_no_outbound` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': False, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': False}` | Direct FA2 API equivalents. |
| `classic_fa2_dissuade_hubs` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'dissuade_hubs': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True}` | Shared params are equivalent; `dissuade_hubs` has no `fa2_ref` param, so this remains a proxy variant. |
| `classic_fa2_linlog` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'linlog': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True, 'linLogMode': True}` | Shared params plus `linlog` -> `linLogMode` are equivalent; registry marks this proxy. |
| `classic_fa2_barnes_hut` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'barnes_hut': True, 'barnes_hut_theta': 1.2}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True, 'barnesHutOptimize': True, 'barnesHutTheta': 1.2}` | Direct FA2 API equivalents including Barnes-Hut settings. |
| `classic_fa2_exact` | `{'steps': 200, 'gravity': 1.0, 'scaling_ratio': 2.0, 'strong_gravity': False, 'outbound_attraction_distribution': True, 'barnes_hut': False}` | `{'iterations': 200, 'gravity': 1.0, 'scalingRatio': 2.0, 'strongGravityMode': False, 'outboundAttractionDistribution': True, 'barnesHutOptimize': False}` | Direct FA2 API equivalents. |
| `classic_stress_sgd_steps30` | `{'steps': 30, 'eps': 0.01}` | `{'t_max': 30, 'eps': 0.01}` | `steps` -> `t_max` and `eps` -> `eps`: identical s_gd2 schedule horizon/tolerance names. |
| `classic_stress_sgd_steps300` | `{'steps': 300, 'eps': 0.01}` | `{'t_max': 300, 'eps': 0.01}` | Equivalent. |
| `classic_stress_sgd_eps001` | `{'steps': 300, 'eps': 0.001}` | `{'t_max': 300, 'eps': 0.001}` | Equivalent. |
| `classic_stress_sgd_eps01` | `{'steps': 300, 'eps': 0.1}` | `{'t_max': 300, 'eps': 0.1}` | Equivalent. |
| `classic_spectral_default` | `{}` | `{}` | No scalar params; implementation path targets NetworkX spectral layout. |
| `classic_spectral_nx_fidelity` | `{'networkx_fidelity': True}` | `{}` | `networkx_fidelity` selects the NetworkX-compatible implementation path; no reference scalar param. |
| `classic_spectral_random_walk` | `{'normalization': 'random_walk'}` | `{}` | No original-side reference params; random-walk normalization has no configured reference pair. |
| `classic_spectral_unnormalized` | `{'normalization': 'unnormalized'}` | `{}` | No original-side reference params in this worktree. |
| `classic_classical_mds_default` | `{}` | `{}` | No scalar params. |
| `classic_classical_mds_igraph_fidelity` | `{'igraph_fidelity': True}` | `{}` | `igraph_fidelity` selects the igraph-compatible MDS path; no reference scalar param. |
| `classic_stress_maj_default` | `{'iterations': 200}` | `{'iterations': 200}` | `iterations` -> `iterations`: identical majorization sweep count for the OGDF stress proxy. |
| `classic_neato` | `{'maxiter': 200, 'epsilon': 0.0001, 'pack': True}` | `{}` | Dagua knobs are Graphviz-compat settings; `graphviz_neato` adapter accepts no variant params, so reference uses Graphviz defaults. |
| `classic_neato_graphviz_fidelity` | `{'maxiter': 200, 'epsilon': 0.0001, 'pack': True, 'fidelity_mode': 'graphviz'}` | `{}` | Same as `classic_neato`; `fidelity_mode` is Dagua-side only. |
| `classic_stress_maj_iter50` | `{'iterations': 50}` | `{'iterations': 50}` | Equivalent. |
| `classic_stress_maj_iter500` | `{'iterations': 500}` | `{'iterations': 500}` | Equivalent. |
| `classic_sugiyama_default` | `{'barycenter_passes': 24, 'rank_sep': 1.0, 'node_sep': 1.0, 'fidelity_mode': 'igraph'}` | `{'maxiter': 24, 'vgap': 1.0, 'hgap': 1.0}` | `barycenter_passes` -> `maxiter`: matching max full down+up crossing-minimization pass budget; `rank_sep/node_sep` -> `vgap/hgap`: matching library gap parameters. |
| `classic_sugiyama_graphviz_fidelity` | `{'barycenter_passes': 24, 'rank_sep': 1.0, 'node_sep': 1.0, 'fidelity_mode': 'graphviz'}` | `{'maxiter': 24, 'vgap': 1.0, 'hgap': 1.0}` | Dagua params target Graphviz mode, but `graphviz_dot` ignores `original_params`; reference uses Graphviz dot defaults. |
| `classic_sugiyama_passes4` | `{'barycenter_passes': 4, 'rank_sep': 1.0, 'node_sep': 1.0, 'fidelity_mode': 'igraph'}` | `{'maxiter': 4, 'vgap': 1.0, 'hgap': 1.0}` | Equivalent max-pass and gap semantics. |
| `classic_sugiyama_passes48` | `{'barycenter_passes': 48, 'rank_sep': 1.0, 'node_sep': 1.0, 'fidelity_mode': 'igraph'}` | `{'maxiter': 48, 'vgap': 1.0, 'hgap': 1.0}` | Equivalent max-pass and gap semantics. |
| `classic_sugiyama_wide` | `{'barycenter_passes': 24, 'rank_sep': 2.0, 'node_sep': 2.0, 'fidelity_mode': 'igraph'}` | `{'maxiter': 24, 'vgap': 2.0, 'hgap': 2.0}` | Equivalent max-pass and gap semantics. |
| `classic_sugiyama_tight` | `{'barycenter_passes': 24, 'rank_sep': 0.5, 'node_sep': 0.5, 'fidelity_mode': 'igraph'}` | `{'maxiter': 24, 'vgap': 0.5, 'hgap': 0.5}` | Equivalent max-pass and gap semantics. |
| `classic_tsnet_default` | `{'perplexity': 30.0, 'steps': 500}` | `{'perplexity': 30.0, 'max_iter': 500}` | `perplexity` -> `perplexity`; `steps` -> `max_iter`: equivalent. |
| `classic_tsnet_perp5` | `{'perplexity': 5.0, 'steps': 500}` | `{'perplexity': 5.0, 'max_iter': 500}` | Equivalent. |
| `classic_tsnet_perp50` | `{'perplexity': 50.0, 'steps': 500}` | `{'perplexity': 50.0, 'max_iter': 500}` | Equivalent. |
| `classic_tsnet_steps200` | `{'perplexity': 30.0, 'steps': 250}` | `{'perplexity': 30.0, 'max_iter': 250}` | Equivalent after R41 fix; variant id remains stable although the actual semantic is now 250 iterations. |
| `classic_tsnet_steps2000` | `{'perplexity': 30.0, 'steps': 2000}` | `{'perplexity': 30.0, 'max_iter': 2000}` | Equivalent. |
| `classic_gem_iters100` | `{'max_iters': 100}` | `{}` | `max_iters` has no OGDF runner param in current adapter; reference uses OGDF default, so this is a proxy comparison. |
| `classic_gem_iters500` | `{'max_iters': 500}` | `{}` | Same proxy limitation. |
| `classic_gem_iters2000` | `{'max_iters': 2000}` | `{}` | Same proxy limitation. |
| `classic_fmmm_steps10` | `{'steps': 10}` | `{}` | `steps` has no OGDF runner param in current adapter; reference uses OGDF default, so this is a proxy comparison. |
| `classic_fmmm_steps100` | `{'steps': 100}` | `{}` | Same proxy limitation. |
| `classic_fmmm_steps200` | `{'steps': 200}` | `{}` | Same proxy limitation. |
| `classic_maxent_stress_default` | `{'steps': 200, 'alpha': 1.0, 'use_entropy': False}` | `{}` | `alpha/use_entropy` have no OGDF stress proxy equivalent; default row has no reference iteration override. |
| `classic_maxent_stress_entropy` | `{'steps': 200, 'alpha': 1.0, 'use_entropy': True}` | `{}` | Proxy comparison; entropy has no reference param. |
| `classic_maxent_stress_alpha2` | `{'steps': 200, 'alpha': 2.0, 'use_entropy': True}` | `{}` | Proxy comparison; alpha/entropy have no reference params. |
| `classic_maxent_stress_steps50` | `{'steps': 50, 'alpha': 1.0, 'use_entropy': False}` | `{'iterations': 50}` | `steps` -> `iterations` for the OGDF stress proxy; alpha/entropy not represented. |
| `classic_maxent_stress_steps400` | `{'steps': 400, 'alpha': 1.0, 'use_entropy': False}` | `{'iterations': 400}` | `steps` -> `iterations` for the OGDF stress proxy; alpha/entropy not represented. |
| `classic_davidson_harel_rounds50` | `{'rounds': 50}` | `{'maxiter': 50}` | `rounds` -> `maxiter`: identical annealing iteration budget; `fineiter` omitted on both sides. |
| `classic_davidson_harel_rounds100` | `{'rounds': 100}` | `{'maxiter': 100}` | Equivalent. |
| `classic_davidson_harel_rounds200` | `{'rounds': 200}` | `{'maxiter': 200}` | Equivalent. |
| `classic_linlog_default` | `{'a': 1.0, 'r': 0.0, 'steps': 300}` | `{'attrExponent': 1.0, 'repuExponent': 0.0, 'steps': 300}` | `a` -> `attrExponent`, `r` -> `repuExponent`, `steps` -> `steps`: direct LinLog energy/iteration equivalents. |
| `classic_linlog_quadratic` | `{'a': 2.0, 'r': 0.0, 'steps': 300}` | `{'attrExponent': 2.0, 'repuExponent': 0.0, 'steps': 300}` | Equivalent. |
| `classic_linlog_power` | `{'a': 1.0, 'r': 0.5, 'steps': 300}` | `{'attrExponent': 1.0, 'repuExponent': 0.5, 'steps': 300}` | Equivalent. |
| `classic_linlog_steps100` | `{'a': 1.0, 'r': 0.0, 'steps': 100}` | `{'attrExponent': 1.0, 'repuExponent': 0.0, 'steps': 100}` | Equivalent. |
| `classic_linlog_steps500` | `{'a': 1.0, 'r': 0.0, 'steps': 500}` | `{'attrExponent': 1.0, 'repuExponent': 0.0, 'steps': 500}` | Equivalent. |
| `classic_pivot_mds_10` | `{'n_pivots': 10, 'first_pivot': 'first_node', 'compute_dtype': 'float64', 'distance_scale': 100.0, 'ogdf_path_special_case': True}` | `{'n_pivots': 10}` | `n_pivots` -> OGDF `numberOfPivots`: identical landmark count; other Dagua fidelity knobs are fixed OGDF-compatible choices. |
| `classic_pivot_mds_50` | `{'n_pivots': 50, 'first_pivot': 'first_node', 'compute_dtype': 'float64', 'distance_scale': 100.0, 'ogdf_path_special_case': True}` | `{'n_pivots': 50}` | Equivalent pivot count. |
| `classic_pivot_mds_100` | `{'n_pivots': 100, 'first_pivot': 'first_node', 'compute_dtype': 'float64', 'distance_scale': 100.0, 'ogdf_path_special_case': True}` | `{'n_pivots': 100}` | Equivalent pivot count. |
| `classic_pivot_mds_200` | `{'n_pivots': 200, 'first_pivot': 'first_node', 'compute_dtype': 'float64', 'distance_scale': 100.0, 'ogdf_path_special_case': True}` | `{'n_pivots': 200}` | Equivalent pivot count. |
| `classic_rt_default` | `{}` | `{}` | No scalar params. |
| `classic_rt_horizontal` | `{'horizontal': True}` | `{}` | No original-side reference params; horizontal is presentation-only. |
| `classic_graphopt_default` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 30.0, 'spring_constant': 1.0}` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 30.0, 'spring_constant': 1.0}` | Direct igraph GraphOpt API equivalents; `spring_length` omitted and defaults on both sides. |
| `classic_graphopt_charge_low` | `{'niter': 500, 'node_charge': 0.0005, 'node_mass': 30.0, 'spring_constant': 1.0}` | `{'niter': 500, 'node_charge': 0.0005, 'node_mass': 30.0, 'spring_constant': 1.0}` | Equivalent. |
| `classic_graphopt_charge_high` | `{'niter': 500, 'node_charge': 0.002, 'node_mass': 30.0, 'spring_constant': 1.0}` | `{'niter': 500, 'node_charge': 0.002, 'node_mass': 30.0, 'spring_constant': 1.0}` | Equivalent. |
| `classic_graphopt_mass_low` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 10.0, 'spring_constant': 1.0}` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 10.0, 'spring_constant': 1.0}` | Equivalent. |
| `classic_graphopt_mass_high` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 50.0, 'spring_constant': 1.0}` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 50.0, 'spring_constant': 1.0}` | Equivalent. |
| `classic_graphopt_spring2` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 30.0, 'spring_constant': 2.0}` | `{'niter': 500, 'node_charge': 0.001, 'node_mass': 30.0, 'spring_constant': 2.0}` | Equivalent. |
| `classic_drl_default` | `{'options': 'default'}` | `{'options': 'default'}` | `options` -> `options`: direct igraph DrL option preset equivalent. |
| `classic_drl_coarsen` | `{'options': 'coarsen'}` | `{'options': 'coarsen'}` | Equivalent. |
| `classic_drl_coarsest` | `{'options': 'coarsest'}` | `{'options': 'coarsest'}` | Equivalent. |
| `classic_drl_refine` | `{'options': 'refine'}` | `{'options': 'refine'}` | Equivalent. |
| `classic_drl_final` | `{'options': 'final'}` | `{'options': 'final'}` | Equivalent. |
| `classic_lgl_default` | `{'maxiter': 150, 'coolexp': 1.5}` | `{'maxiter': 150, 'coolexp': 1.5}` | `maxiter/coolexp` -> `maxiter/coolexp`: direct igraph LGL equivalents; `repulserad` is not varied in registry. |
| `classic_lgl_iter50` | `{'maxiter': 50, 'coolexp': 1.5}` | `{'maxiter': 50, 'coolexp': 1.5}` | Equivalent. |
| `classic_lgl_iter300` | `{'maxiter': 300, 'coolexp': 1.5}` | `{'maxiter': 300, 'coolexp': 1.5}` | Equivalent. |
| `classic_lgl_cool1` | `{'maxiter': 150, 'coolexp': 1.0}` | `{'maxiter': 150, 'coolexp': 1.0}` | Equivalent. |
| `classic_lgl_cool2` | `{'maxiter': 150, 'coolexp': 2.0}` | `{'maxiter': 150, 'coolexp': 2.0}` | Equivalent. |
| `classic_sfdp_default` | `{'steps': 500, 'theta': 0.6, 'repulsive_exponent': -1.0}` | `{}` | Dagua `steps/theta/repulsive_exponent` have no Graphviz adapter params; `graphviz_sfdp` uses defaults, so variants are proxy comparisons. |
| `classic_sfdp_graphviz_fidelity` | `{'steps': 500, 'theta': 0.6, 'repulsive_exponent': -1.0, 'fidelity_mode': 'graphviz'}` | `{}` | Same proxy limitation; `fidelity_mode` is Dagua-side only. |
| `classic_sfdp_theta04` | `{'steps': 500, 'theta': 0.4, 'repulsive_exponent': -1.0}` | `{}` | Same proxy limitation. |
| `classic_sfdp_theta08` | `{'steps': 500, 'theta': 0.8, 'repulsive_exponent': -1.0}` | `{}` | Same proxy limitation. |
| `classic_sfdp_p_neg2` | `{'steps': 500, 'theta': 0.6, 'repulsive_exponent': -2.0}` | `{}` | Same proxy limitation. |
| `classic_sfdp_steps200` | `{'steps': 200, 'theta': 0.6, 'repulsive_exponent': -1.0}` | `{}` | Same proxy limitation. |
| `classic_umap_default` | `{'n_neighbors': 15, 'min_dist': 0.1, 'spread': 1.0}` | `{'n_neighbors': 15, 'min_dist': 0.1, 'spread': 1.0}` | Direct UMAP API equivalents; small-graph clamping is graph-dependent and applies on the reference side. |
| `classic_umap_nn5` | `{'n_neighbors': 5, 'min_dist': 0.1, 'spread': 1.0}` | `{'n_neighbors': 5, 'min_dist': 0.1, 'spread': 1.0}` | Equivalent. |
| `classic_umap_nn30` | `{'n_neighbors': 30, 'min_dist': 0.1, 'spread': 1.0}` | `{'n_neighbors': 30, 'min_dist': 0.1, 'spread': 1.0}` | Equivalent. |
| `classic_umap_mindist001` | `{'n_neighbors': 15, 'min_dist': 0.01, 'spread': 1.0}` | `{'n_neighbors': 15, 'min_dist': 0.01, 'spread': 1.0}` | Equivalent. |
| `classic_umap_mindist05` | `{'n_neighbors': 15, 'min_dist': 0.5, 'spread': 1.0}` | `{'n_neighbors': 15, 'min_dist': 0.5, 'spread': 1.0}` | Equivalent. |
| `classic_umap_spread2` | `{'n_neighbors': 15, 'min_dist': 0.1, 'spread': 2.0}` | `{'n_neighbors': 15, 'min_dist': 0.1, 'spread': 2.0}` | Equivalent. |
| `classic_neulay_default` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.1, 'radius': 0.4}` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.1, 'radius': 0.4}` | Direct recovered NeuLay wrapper equivalents. |
| `classic_neulay_lr001` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.01, 'radius': 0.4}` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.01, 'radius': 0.4}` | Equivalent. |
| `classic_neulay_lr05` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.5, 'radius': 0.4}` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.5, 'radius': 0.4}` | Equivalent. |
| `classic_neulay_radius02` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.1, 'radius': 0.2}` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.1, 'radius': 0.2}` | Equivalent. |
| `classic_neulay_radius08` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.1, 'radius': 0.8}` | `{'steps': 20000, 'gcn_steps': 2000, 'use_gcn': True, 'lr': 0.1, 'radius': 0.8}` | Equivalent. |
| `classic_neulay_no_gcn` | `{'steps': 20000, 'use_gcn': False, 'lr': 0.1, 'radius': 0.4}` | `{'steps': 20000, 'use_gcn': False, 'lr': 0.1, 'radius': 0.4}` | Equivalent; `gcn_steps` is irrelevant when `use_gcn=False`. |
| `classic_sgd2_multi_default` | `{'criteria': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'lr': 0.01, 'steps': 2000, 'grad_clamp': 5.0}` | `{'criteria_weights': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.01}}` | `criteria` -> `criteria_weights`, `steps` -> `max_iter`, `lr` -> `optimizer_kwargs.lr`, `grad_clamp` -> `grad_clamp`: direct upstream GD2 equivalents. |
| `classic_sgd2_multi_stress_only` | `{'criteria': {'stress': 1.0}, 'lr': 0.01, 'steps': 2000, 'grad_clamp': 5.0}` | `{'criteria_weights': {'stress': 1.0}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.01}}` | Equivalent. |
| `classic_sgd2_multi_with_crossing` | `{'criteria': {'stress': 1.0, 'crossings': 0.5}, 'lr': 0.01, 'steps': 2000, 'grad_clamp': 5.0}` | `{'criteria_weights': {'stress': 1.0, 'crossings': 0.5}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.01}}` | Equivalent. |
| `classic_sgd2_multi_with_aspect` | `{'criteria': {'stress': 0.8, 'aspect_ratio': 0.2}, 'lr': 0.01, 'steps': 2000, 'grad_clamp': 5.0}` | `{'criteria_weights': {'stress': 0.8, 'aspect_ratio': 0.2}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.01}}` | Equivalent. |
| `classic_sgd2_multi_lr001` | `{'criteria': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'lr': 0.001, 'steps': 2000, 'grad_clamp': 5.0}` | `{'criteria_weights': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.001}}` | Equivalent. |
| `classic_sgd2_multi_lr01` | `{'criteria': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'lr': 0.1, 'steps': 2000, 'grad_clamp': 5.0}` | `{'criteria_weights': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.1}}` | Equivalent. |
| `classic_sgd2_multi_batch8` | `{'criteria': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'lr': 0.01, 'steps': 2000, 'grad_clamp': 5.0, 'batch_size': 8}` | `{'criteria_weights': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.01}, 'sample_sizes': {'stress': 8, 'ideal_edge_length': 8}}` | `batch_size` -> per-criterion `sample_sizes`: equivalent. |
| `classic_sgd2_multi_batch128` | `{'criteria': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'lr': 0.01, 'steps': 2000, 'grad_clamp': 5.0, 'batch_size': 128}` | `{'criteria_weights': {'stress': 1.0, 'ideal_edge_length': 1.0}, 'max_iter': 2000, 'grad_clamp': 5.0, 'optimizer_kwargs': {'lr': 0.01}, 'sample_sizes': {'stress': 128, 'ideal_edge_length': 128}}` | Equivalent. |
| `classic_fcose_default` | `{'quality': 'default', 'randomize': True, 'steps': 2500}` | `{}` | `quality=randomize/steps` are Dagua-side fixed to Cytoscape defaults; current Cytoscape adapter receives no reference params. |
| `classic_fcose_proof` | `{'quality': 'proof', 'randomize': True, 'steps': 2500}` | `{'quality': 'proof'}` | `quality` maps directly; `randomize/steps` remain Dagua-side fixed defaults. |
| `cytoscape_fcose_default` | `{}` | `{}` | External reference variant, not a Dagua reimplementation; no original-side parameter pair. |
| `cytoscape_fcose_quality` | `{'quality': 'proof'}` | `{}` | External reference variant; no original-side parameter pair. |
| `gephi_yifanhu_default` | `{}` | `{}` | External reference variant; no original-side parameter pair. |

## Concerns

- `classic_sugiyama_graphviz_fidelity` still lists Graphviz-side
  `original_params`, but the Graphviz adapter ignores variant params. Fixing
  that would require adapter support, which is out of scope for this task.
- `classic_sfdp_*`, `classic_neato*`, `classic_gem_*`, and `classic_fmmm_*`
  contain Dagua-side semantic variants that the current reference adapters do
  not parameterize. Treat their RMSD comparisons as proxy/reference-default
  comparisons, not strict parameter-matched comparisons.
- `classic_kk_steps100/300/1000` have legacy IDs/display labels; the active
  Dagua parameter is `steps=None` for all three, matching NetworkX default
  behavior rather than the numeric labels.

## Knowledge

- `barycenter_passes` is a maximum number of full crossing-minimization passes
  in Dagua Sugiyama. In igraph fidelity mode, early stopping can reduce the
  actual pass count after stable ordering.
- `TSNEGraph` clamps `max_iter` to at least `250` before constructing sklearn's
  `TSNE`, so any registry value below 250 is not the actual reference
  iteration count.
- Graphviz benchmark adapters currently do not implement `layout_with_variant`,
  so `original_params` for Graphviz references are metadata only.
