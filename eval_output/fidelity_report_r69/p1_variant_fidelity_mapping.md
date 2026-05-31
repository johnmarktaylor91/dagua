# R69 P1 Variant Fidelity Mapping

Main table: variants whose `reimpl_params` now explicitly route through a pipeline fidelity selector. Existing selectors were preserved unchanged.

| variant_id | original_engine (reference) | pipeline file | chosen fidelity_mode | verified-routes-to-bit-exact-path (Y/N/no-op/no-port) |
|---|---|---|---|---|
| `classic_fa2_default` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_gravity0` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_gravity2` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_scaling1` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_scaling4` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_strong_gravity` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_no_outbound` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_dissuade_hubs` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_linlog` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_barnes_hut` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_fa2_exact` | `fa2_ref` | `dagua/layout/ops/pipelines/fa2.py` | `True` | Y |
| `classic_stress_sgd_steps30` | `sgd2` | `dagua/layout/ops/pipelines/stress_sgd.py` | `True` | Y |
| `classic_stress_sgd_steps300` | `sgd2` | `dagua/layout/ops/pipelines/stress_sgd.py` | `True` | Y |
| `classic_stress_sgd_eps001` | `sgd2` | `dagua/layout/ops/pipelines/stress_sgd.py` | `True` | Y |
| `classic_stress_sgd_eps01` | `sgd2` | `dagua/layout/ops/pipelines/stress_sgd.py` | `True` | Y |
| `classic_spectral_default` | `nx_spectral` | `dagua/layout/ops/pipelines/spectral.py` | `'networkx'` | Y |
| `classic_spectral_nx_fidelity` | `nx_spectral` | `dagua/layout/ops/pipelines/spectral.py` | `'networkx'` | Y |
| `classic_stress_maj_default` | `ogdf_stress` | `dagua/layout/ops/pipelines/stress_majorization.py` | `'ogdf'` | Y |
| `classic_neato` | `graphviz_neato` | `dagua/layout/ops/pipelines/neato.py` | `'graphviz'` | Y |
| `classic_neato_graphviz_fidelity` | `graphviz_neato` | `dagua/layout/ops/pipelines/neato.py` | `'graphviz'` | Y |
| `classic_stress_maj_iter50` | `ogdf_stress` | `dagua/layout/ops/pipelines/stress_majorization.py` | `'ogdf'` | Y |
| `classic_stress_maj_iter500` | `ogdf_stress` | `dagua/layout/ops/pipelines/stress_majorization.py` | `'ogdf'` | Y |
| `classic_sugiyama_default` | `igraph_sugiyama` | `dagua/layout/ops/pipelines/sugiyama.py` | `'igraph'` | Y |
| `classic_sugiyama_graphviz_fidelity` | `graphviz_dot` | `dagua/layout/ops/pipelines/sugiyama.py` | `'graphviz'` | Y |
| `classic_sugiyama_passes4` | `igraph_sugiyama` | `dagua/layout/ops/pipelines/sugiyama.py` | `'igraph'` | Y |
| `classic_sugiyama_passes48` | `igraph_sugiyama` | `dagua/layout/ops/pipelines/sugiyama.py` | `'igraph'` | Y |
| `classic_sugiyama_wide` | `igraph_sugiyama` | `dagua/layout/ops/pipelines/sugiyama.py` | `'igraph'` | Y |
| `classic_sugiyama_tight` | `igraph_sugiyama` | `dagua/layout/ops/pipelines/sugiyama.py` | `'igraph'` | Y |
| `classic_tsnet_default` | `tsne_graph` | `dagua/layout/ops/pipelines/tsnet.py` | `True` | Y |
| `classic_tsnet_perp5` | `tsne_graph` | `dagua/layout/ops/pipelines/tsnet.py` | `True` | Y |
| `classic_tsnet_perp50` | `tsne_graph` | `dagua/layout/ops/pipelines/tsnet.py` | `True` | Y |
| `classic_tsnet_steps200` | `tsne_graph` | `dagua/layout/ops/pipelines/tsnet.py` | `True` | Y |
| `classic_tsnet_steps2000` | `tsne_graph` | `dagua/layout/ops/pipelines/tsnet.py` | `True` | Y |
| `classic_gem_iters100` | `ogdf_gem` | `dagua/layout/ops/pipelines/gem.py` | `'ogdf'` | Y |
| `classic_gem_iters500` | `ogdf_gem` | `dagua/layout/ops/pipelines/gem.py` | `'ogdf'` | Y |
| `classic_gem_iters2000` | `ogdf_gem` | `dagua/layout/ops/pipelines/gem.py` | `'ogdf'` | Y |
| `classic_fmmm_steps10` | `ogdf_fmmm` | `dagua/layout/ops/pipelines/fmmm.py` | `True` | Y |
| `classic_fmmm_steps100` | `ogdf_fmmm` | `dagua/layout/ops/pipelines/fmmm.py` | `True` | Y |
| `classic_fmmm_steps200` | `ogdf_fmmm` | `dagua/layout/ops/pipelines/fmmm.py` | `True` | Y |
| `classic_fmmm_graphviz_fdp_fidelity` | `graphviz_fdp` | `dagua/layout/ops/pipelines/fmmm.py` | `True` | Y |
| `classic_davidson_harel_rounds50` | `igraph_davidson_harel` | `dagua/layout/ops/pipelines/davidson_harel.py` | `True` | Y |
| `classic_davidson_harel_rounds100` | `igraph_davidson_harel` | `dagua/layout/ops/pipelines/davidson_harel.py` | `True` | Y |
| `classic_davidson_harel_rounds200` | `igraph_davidson_harel` | `dagua/layout/ops/pipelines/davidson_harel.py` | `True` | Y |
| `classic_linlog_default` | `linlog` | `dagua/layout/ops/pipelines/linlog.py` | `True` | Y |
| `classic_linlog_quadratic` | `linlog` | `dagua/layout/ops/pipelines/linlog.py` | `True` | Y |
| `classic_linlog_power` | `linlog` | `dagua/layout/ops/pipelines/linlog.py` | `True` | Y |
| `classic_linlog_steps100` | `linlog` | `dagua/layout/ops/pipelines/linlog.py` | `True` | Y |
| `classic_linlog_steps500` | `linlog` | `dagua/layout/ops/pipelines/linlog.py` | `True` | Y |
| `classic_rt_default` | `igraph_rt` | `dagua/layout/ops/pipelines/reingold_tilford.py` | `'igraph'` | Y |
| `classic_graphopt_default` | `igraph_graphopt` | `dagua/layout/ops/pipelines/graphopt.py` | `True` | Y |
| `classic_graphopt_charge_low` | `igraph_graphopt` | `dagua/layout/ops/pipelines/graphopt.py` | `True` | Y |
| `classic_graphopt_charge_high` | `igraph_graphopt` | `dagua/layout/ops/pipelines/graphopt.py` | `True` | Y |
| `classic_graphopt_mass_low` | `igraph_graphopt` | `dagua/layout/ops/pipelines/graphopt.py` | `True` | Y |
| `classic_graphopt_mass_high` | `igraph_graphopt` | `dagua/layout/ops/pipelines/graphopt.py` | `True` | Y |
| `classic_graphopt_spring2` | `igraph_graphopt` | `dagua/layout/ops/pipelines/graphopt.py` | `True` | Y |
| `classic_drl_default` | `igraph_drl` | `dagua/layout/ops/pipelines/drl.py` | `True` | Y |
| `classic_drl_coarsen` | `igraph_drl` | `dagua/layout/ops/pipelines/drl.py` | `True` | Y |
| `classic_drl_coarsest` | `igraph_drl` | `dagua/layout/ops/pipelines/drl.py` | `True` | Y |
| `classic_drl_refine` | `igraph_drl` | `dagua/layout/ops/pipelines/drl.py` | `True` | Y |
| `classic_drl_final` | `igraph_drl` | `dagua/layout/ops/pipelines/drl.py` | `True` | Y |
| `classic_lgl_default` | `igraph_lgl` | `dagua/layout/ops/pipelines/lgl.py` | `True` | Y |
| `classic_lgl_iter50` | `igraph_lgl` | `dagua/layout/ops/pipelines/lgl.py` | `True` | Y |
| `classic_lgl_iter300` | `igraph_lgl` | `dagua/layout/ops/pipelines/lgl.py` | `True` | Y |
| `classic_lgl_cool1` | `igraph_lgl` | `dagua/layout/ops/pipelines/lgl.py` | `True` | Y |
| `classic_lgl_cool2` | `igraph_lgl` | `dagua/layout/ops/pipelines/lgl.py` | `True` | Y |
| `classic_sfdp_default` | `graphviz_sfdp` | `dagua/layout/ops/pipelines/sfdp.py` | `'graphviz'` | Y |
| `classic_sfdp_graphviz_fidelity` | `graphviz_sfdp` | `dagua/layout/ops/pipelines/sfdp.py` | `'graphviz'` | Y |
| `classic_sfdp_theta04` | `graphviz_sfdp` | `dagua/layout/ops/pipelines/sfdp.py` | `'graphviz'` | Y |
| `classic_sfdp_theta08` | `graphviz_sfdp` | `dagua/layout/ops/pipelines/sfdp.py` | `'graphviz'` | Y |
| `classic_sfdp_p_neg2` | `graphviz_sfdp` | `dagua/layout/ops/pipelines/sfdp.py` | `'graphviz'` | Y |
| `classic_sfdp_steps200` | `graphviz_sfdp` | `dagua/layout/ops/pipelines/sfdp.py` | `'graphviz'` | Y |
| `classic_umap_default` | `umap_graph` | `dagua/layout/ops/pipelines/umap_layout.py` | `True` | Y |
| `classic_umap_nn5` | `umap_graph` | `dagua/layout/ops/pipelines/umap_layout.py` | `True` | Y |
| `classic_umap_nn30` | `umap_graph` | `dagua/layout/ops/pipelines/umap_layout.py` | `True` | Y |
| `classic_umap_mindist001` | `umap_graph` | `dagua/layout/ops/pipelines/umap_layout.py` | `True` | Y |
| `classic_umap_mindist05` | `umap_graph` | `dagua/layout/ops/pipelines/umap_layout.py` | `True` | Y |
| `classic_umap_spread2` | `umap_graph` | `dagua/layout/ops/pipelines/umap_layout.py` | `True` | Y |
| `classic_neulay_default` | `neulay` | `dagua/layout/ops/pipelines/neulay.py` | `'old_code'` | Y |
| `classic_neulay_lr001` | `neulay` | `dagua/layout/ops/pipelines/neulay.py` | `'old_code'` | Y |
| `classic_neulay_lr05` | `neulay` | `dagua/layout/ops/pipelines/neulay.py` | `'old_code'` | Y |
| `classic_neulay_radius02` | `neulay` | `dagua/layout/ops/pipelines/neulay.py` | `'old_code'` | Y |
| `classic_neulay_radius08` | `neulay` | `dagua/layout/ops/pipelines/neulay.py` | `'old_code'` | Y |
| `classic_neulay_no_gcn` | `neulay` | `dagua/layout/ops/pipelines/neulay.py` | `'old_code'` | Y |
| `classic_sgd2_multi_default` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |
| `classic_sgd2_multi_stress_only` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |
| `classic_sgd2_multi_with_crossing` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |
| `classic_sgd2_multi_with_aspect` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |
| `classic_sgd2_multi_lr001` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |
| `classic_sgd2_multi_lr01` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |
| `classic_sgd2_multi_batch8` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |
| `classic_sgd2_multi_batch128` | `sgd2_multi_ref` | `dagua/layout/ops/pipelines/sgd2_multi.py` | `True` | Y |

## No-Op Variants

None found in the requested smoke coverage. `classic_neato` and `classic_graphopt_default` both produced different coordinates with fidelity mode enabled versus disabled.

## No-Port / No `fidelity_mode` Selector Variants

| variant_id | original_engine (reference) | pipeline file | chosen fidelity_mode | verified-routes-to-bit-exact-path (Y/N/no-op/no-port) |
|---|---|---|---|---|
| `classic_fr_steps50` | `nx_spring` | `dagua/layout/ops/pipelines/fr.py` | n/a - NetworkX spring path uses networkx_compat, not fidelity_mode; FR fidelity_mode targets igraph. | no-port |
| `classic_fr_steps100` | `nx_spring` | `dagua/layout/ops/pipelines/fr.py` | n/a - NetworkX spring path uses networkx_compat, not fidelity_mode; FR fidelity_mode targets igraph. | no-port |
| `classic_fr_steps200` | `nx_spring` | `dagua/layout/ops/pipelines/fr.py` | n/a - NetworkX spring path uses networkx_compat, not fidelity_mode; FR fidelity_mode targets igraph. | no-port |
| `classic_fr_steps500` | `nx_spring` | `dagua/layout/ops/pipelines/fr.py` | n/a - NetworkX spring path uses networkx_compat, not fidelity_mode; FR fidelity_mode targets igraph. | no-port |
| `classic_kk_steps100` | `nx_kamada_kawai` | `dagua/layout/ops/pipelines/kk.py` | n/a - NetworkX KK reference; KK fidelity_mode targets igraph only. | no-port |
| `classic_kk_steps300` | `nx_kamada_kawai` | `dagua/layout/ops/pipelines/kk.py` | n/a - NetworkX KK reference; KK fidelity_mode targets igraph only. | no-port |
| `classic_kk_steps1000` | `nx_kamada_kawai` | `dagua/layout/ops/pipelines/kk.py` | n/a - NetworkX KK reference; KK fidelity_mode targets igraph only. | no-port |
| `classic_fr_kk_default` | `None` | `dagua/layout/ops/pipelines/fr.py + dagua/layout/ops/pipelines/kk.py` | n/a - No paired reference engine. | no-port |
| `classic_fr_kk_long` | `None` | `dagua/layout/ops/pipelines/fr.py + dagua/layout/ops/pipelines/kk.py` | n/a - No paired reference engine. | no-port |
| `classic_kk_fr_default` | `None` | `dagua/layout/ops/pipelines/kk.py + dagua/layout/ops/pipelines/fr.py` | n/a - No paired reference engine. | no-port |
| `classic_kk_fr_long` | `None` | `dagua/layout/ops/pipelines/kk.py + dagua/layout/ops/pipelines/fr.py` | n/a - No paired reference engine. | no-port |
| `classic_spectral_random_walk` | `None` | `dagua/layout/ops/pipelines/spectral.py` | n/a - No paired reference engine. | no-port |
| `classic_spectral_unnormalized` | `None` | `dagua/layout/ops/pipelines/spectral.py` | n/a - No paired reference engine. | no-port |
| `classic_classical_mds_default` | `igraph_mds` | `dagua/layout/ops/pipelines/classical_mds.py` | n/a - Pipeline exposes igraph_fidelity/ogdf_fidelity, not fidelity_mode. | no-port |
| `classic_classical_mds_igraph_fidelity` | `igraph_mds` | `dagua/layout/ops/pipelines/classical_mds.py` | n/a - Existing bit-exact selector is igraph_fidelity=True, not fidelity_mode. | no-port |
| `classic_maxent_stress_default` | `ogdf_stress` | `dagua/layout/ops/pipelines/maxent_stress.py` | n/a - No fidelity_mode selector in maxent_stress.py. | no-port |
| `classic_maxent_stress_entropy` | `ogdf_stress` | `dagua/layout/ops/pipelines/maxent_stress.py` | n/a - No fidelity_mode selector in maxent_stress.py. | no-port |
| `classic_maxent_stress_alpha2` | `ogdf_stress` | `dagua/layout/ops/pipelines/maxent_stress.py` | n/a - No fidelity_mode selector in maxent_stress.py. | no-port |
| `classic_maxent_stress_steps50` | `ogdf_stress` | `dagua/layout/ops/pipelines/maxent_stress.py` | n/a - No fidelity_mode selector in maxent_stress.py. | no-port |
| `classic_maxent_stress_steps400` | `ogdf_stress` | `dagua/layout/ops/pipelines/maxent_stress.py` | n/a - No fidelity_mode selector in maxent_stress.py. | no-port |
| `classic_pivot_mds_10` | `ogdf_pivot_mds` | `dagua/layout/ops/pipelines/pivot_mds.py` | n/a - OGDF parity is encoded by first_pivot/compute_dtype/distance_scale, not fidelity_mode. | no-port |
| `classic_pivot_mds_50` | `ogdf_pivot_mds` | `dagua/layout/ops/pipelines/pivot_mds.py` | n/a - OGDF parity is encoded by first_pivot/compute_dtype/distance_scale, not fidelity_mode. | no-port |
| `classic_pivot_mds_100` | `ogdf_pivot_mds` | `dagua/layout/ops/pipelines/pivot_mds.py` | n/a - OGDF parity is encoded by first_pivot/compute_dtype/distance_scale, not fidelity_mode. | no-port |
| `classic_pivot_mds_200` | `ogdf_pivot_mds` | `dagua/layout/ops/pipelines/pivot_mds.py` | n/a - OGDF parity is encoded by first_pivot/compute_dtype/distance_scale, not fidelity_mode. | no-port |
| `classic_rt_horizontal` | `None` | `dagua/layout/ops/pipelines/reingold_tilford.py` | n/a - No paired reference engine. | no-port |
| `classic_fcose_default` | `cytoscape_fcose` | `dagua/layout/ops/pipelines/fcose.py` | n/a - Cytoscape fCoSE has no bit-exact Python fidelity port. | no-port |
| `classic_fcose_proof` | `cytoscape_fcose` | `dagua/layout/ops/pipelines/fcose.py` | n/a - Cytoscape fCoSE has no bit-exact Python fidelity port. | no-port |
