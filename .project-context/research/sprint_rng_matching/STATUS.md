# RNG Matching Status

Single source of truth for matched-seed small-graph bit-exact checks.

| engine | reference | best(max) RMSD over fixtures&seeds | worst fixture | verdict | exact_match_count/total | timestamp |
|---|---|---:|---|---|---:|---|
| classic_fcose_proof | cytoscape_fcose__for__classic_fcose_proof | 1.410639785e+00 | complete5 | DIVERGENT | 0/42 | 2026-06-03T00:17:12+00:00 |
| classic_fcose_default | cytoscape_fcose__for__classic_fcose_default | 1.410609947e+00 | complete5 | DIVERGENT | 0/42 | 2026-06-03T00:16:55+00:00 |
| classic_fmmm_graphviz_fdp_fidelity | graphviz_fdp__for__classic_fmmm_graphviz_fdp_fidelity | 1.385764406e+00 | star8 | DIVERGENT | 0/42 | 2026-06-02T22:52:18+00:00 |
| classic_drl_final | igraph_drl__for__classic_drl_final | 1.129169125e+00 | wheel7 | DIVERGENT | 25/42 | 2026-06-02T23:01:27+00:00 |
| classic_drl_refine | igraph_drl__for__classic_drl_refine | 1.094608443e+00 | small_random_12 | DIVERGENT | 28/42 | 2026-06-02T23:00:52+00:00 |
| classic_sgd2_multi_with_crossing | sgd2_multi_ref__for__classic_sgd2_multi_with_crossing | 1.087668309e+00 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T23:56:02+00:00 |
| classic_drl_coarsen | igraph_drl__for__classic_drl_coarsen | 1.040209403e+00 | grid4x4 | DIVERGENT | 36/42 | 2026-06-02T22:59:06+00:00 |
| classic_drl_default | igraph_drl__for__classic_drl_default | 1.008306431e+00 | petersen_10 | DIVERGENT | 35/42 | 2026-06-02T22:58:00+00:00 |
| classic_drl_coarsest | igraph_drl__for__classic_drl_coarsest | 9.897451043e-01 | grid4x4 | DIVERGENT | 28/42 | 2026-06-02T23:00:18+00:00 |
| classic_sfdp_p_neg2 | graphviz_sfdp__for__classic_sfdp_p_neg2 | 9.644042100e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T23:03:21+00:00 |
| classic_sugiyama_wide | igraph_sugiyama__for__classic_sugiyama_wide | 9.295416300e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T22:50:41+00:00 |
| classic_sugiyama_tight | igraph_sugiyama__for__classic_sugiyama_tight | 9.295416300e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T22:50:44+00:00 |
| classic_sugiyama_passes48 | igraph_sugiyama__for__classic_sugiyama_passes48 | 9.295416300e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T22:50:38+00:00 |
| classic_sugiyama_passes4 | igraph_sugiyama__for__classic_sugiyama_passes4 | 9.295416300e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T22:50:35+00:00 |
| classic_sugiyama_default | igraph_sugiyama__for__classic_sugiyama_default | 9.295416300e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T22:50:26+00:00 |
| classic_sugiyama_graphviz_fidelity | graphviz_dot__for__classic_sugiyama_graphviz_fidelity | 8.714770525e-01 | complete_bipartite_3x3 | DIVERGENT | 0/42 | 2026-06-02T22:50:31+00:00 |
| classic_classical_mds_igraph_fidelity | igraph_mds__for__classic_classical_mds_igraph_fidelity | 7.688183270e-01 | petersen_10 | DIVERGENT | 3/42 | 2026-06-02T22:49:31+00:00 |
| classic_classical_mds_default | igraph_mds__for__classic_classical_mds_default | 7.688183270e-01 | petersen_10 | DIVERGENT | 3/42 | 2026-06-02T22:49:27+00:00 |
| classic_sgd2_multi_lr01 | sgd2_multi_ref__for__classic_sgd2_multi_lr01 | 6.594965755e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-03T00:08:37+00:00 |
| classic_sfdp_steps200 | graphviz_sfdp__for__classic_sfdp_steps200 | 6.051651556e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-02T23:03:28+00:00 |
| classic_sgd2_multi_batch8 | sgd2_multi_ref__for__classic_sgd2_multi_batch8 | 5.745635225e-01 | complete5 | DIVERGENT | 0/42 | 2026-06-03T00:12:29+00:00 |
| classic_sfdp_theta08 | graphviz_sfdp__for__classic_sfdp_theta08 | 4.249041952e-01 | balanced_tree_2x3 | DIVERGENT | 0/42 | 2026-06-02T23:03:14+00:00 |
| classic_sfdp_theta04 | graphviz_sfdp__for__classic_sfdp_theta04 | 4.249041952e-01 | balanced_tree_2x3 | DIVERGENT | 0/42 | 2026-06-02T23:03:07+00:00 |
| classic_sfdp_graphviz_fidelity | graphviz_sfdp__for__classic_sfdp_graphviz_fidelity | 4.249041952e-01 | balanced_tree_2x3 | DIVERGENT | 0/42 | 2026-06-02T23:03:00+00:00 |
| classic_sfdp_default | graphviz_sfdp__for__classic_sfdp_default | 4.249041952e-01 | balanced_tree_2x3 | DIVERGENT | 0/42 | 2026-06-02T23:02:54+00:00 |
| classic_sgd2_multi_stress_only | sgd2_multi_ref__for__classic_sgd2_multi_stress_only | 3.945538797e-01 | grid3x3 | DIVERGENT | 2/42 | 2026-06-02T23:39:21+00:00 |
| classic_davidson_harel_rounds50 | igraph_davidson_harel__for__classic_davidson_harel_rounds50 | 3.673371369e-01 | grid3x3 | DIVERGENT | 0/42 | 2026-06-02T22:53:25+00:00 |
| classic_davidson_harel_rounds200 | igraph_davidson_harel__for__classic_davidson_harel_rounds200 | 3.669361328e-01 | grid3x3 | DIVERGENT | 0/42 | 2026-06-02T22:54:52+00:00 |
| classic_davidson_harel_rounds100 | igraph_davidson_harel__for__classic_davidson_harel_rounds100 | 3.563406326e-01 | grid3x3 | DIVERGENT | 0/42 | 2026-06-02T22:53:56+00:00 |
| classic_gem_iters2000 | ogdf_gem__for__classic_gem_iters2000 | 1.651442357e-01 | star8 | DIVERGENT | 0/42 | 2026-06-02T22:51:48+00:00 |
| classic_sgd2_multi_default | sgd2_multi_ref__for__classic_sgd2_multi_default | 7.854999669e-02 | small_random_12 | DIVERGENT | 0/42 | 2026-06-02T23:36:09+00:00 |
| classic_sgd2_multi_with_aspect | sgd2_multi_ref__for__classic_sgd2_multi_with_aspect | 5.325837463e-02 | grid3x3 | DIVERGENT | 0/42 | 2026-06-03T00:00:18+00:00 |
| classic_sgd2_multi_lr001 | sgd2_multi_ref__for__classic_sgd2_multi_lr001 | 3.267863902e-02 | small_random_12 | DIVERGENT | 0/42 | 2026-06-03T00:04:28+00:00 |
| classic_fmmm_steps100 | ogdf_fmmm__for__classic_fmmm_steps100 | 2.085059331e-02 | wheel7 | DIVERGENT | 35/42 | 2026-06-02T22:51:58+00:00 |
| classic_fmmm_steps200 | ogdf_fmmm__for__classic_fmmm_steps200 | 1.828780234e-02 | wheel7 | DIVERGENT | 36/42 | 2026-06-02T22:52:07+00:00 |
| classic_fmmm_steps10 | ogdf_fmmm__for__classic_fmmm_steps10 | 1.121821378e-02 | grid3x3 | DIVERGENT | 34/42 | 2026-06-02T22:51:52+00:00 |
| classic_sgd2_multi_batch128 | sgd2_multi_ref__for__classic_sgd2_multi_batch128 | 9.611627686e-06 | complete5 | CLOSE | 0/42 | 2026-06-03T00:16:41+00:00 |
| classic_gem_iters500 | ogdf_gem__for__classic_gem_iters500 | 8.246905089e-08 | complete_bipartite_3x3 | BIT_EXACT | 0/42 | 2026-06-02T22:51:43+00:00 |
| classic_gem_iters100 | ogdf_gem__for__classic_gem_iters100 | 7.965414923e-08 | star8 | BIT_EXACT | 0/42 | 2026-06-02T22:51:38+00:00 |
| classic_fa2_gravity0 | fa2_ref__for__classic_fa2_gravity0 | 6.793456318e-08 | complete5 | BIT_EXACT | 0/42 | 2026-06-02T22:48:24+00:00 |
| classic_stress_sgd_eps01 | sgd2__for__classic_stress_sgd_eps01 | 6.282334990e-08 | petersen_10 | BIT_EXACT | 0/42 | 2026-06-02T22:49:22+00:00 |
| classic_stress_sgd_steps300 | sgd2__for__classic_stress_sgd_steps300 | 6.217201132e-08 | petersen_10 | BIT_EXACT | 0/42 | 2026-06-02T22:49:12+00:00 |
| classic_graphopt_charge_low | igraph_graphopt__for__classic_graphopt_charge_low | 4.786601167e-08 | petersen_10 | BIT_EXACT | 0/42 | 2026-06-02T22:56:32+00:00 |
| classic_graphopt_charge_high | igraph_graphopt__for__classic_graphopt_charge_high | 4.327179497e-08 | grid4x4 | BIT_EXACT | 0/42 | 2026-06-02T22:56:38+00:00 |
| classic_graphopt_mass_high | igraph_graphopt__for__classic_graphopt_mass_high | 4.302123090e-08 | path8 | BIT_EXACT | 0/42 | 2026-06-02T22:56:49+00:00 |
| classic_graphopt_spring2 | igraph_graphopt__for__classic_graphopt_spring2 | 4.257931858e-08 | cycle6 | BIT_EXACT | 0/42 | 2026-06-02T22:56:55+00:00 |
| classic_graphopt_default | igraph_graphopt__for__classic_graphopt_default | 4.132285080e-08 | petersen_10 | BIT_EXACT | 0/42 | 2026-06-02T22:56:26+00:00 |
| classic_graphopt_mass_low | igraph_graphopt__for__classic_graphopt_mass_low | 4.072235581e-08 | complete_bipartite_3x3 | BIT_EXACT | 0/42 | 2026-06-02T22:56:43+00:00 |
| classic_stress_sgd_eps001 | sgd2__for__classic_stress_sgd_eps001 | 3.953044039e-08 | wheel7 | BIT_EXACT | 0/42 | 2026-06-02T22:49:17+00:00 |
| classic_fa2_strong_gravity | fa2_ref__for__classic_fa2_strong_gravity | 3.909358029e-08 | complete_bipartite_3x3 | BIT_EXACT | 0/42 | 2026-06-02T22:48:41+00:00 |
| classic_fa2_dissuade_hubs | fa2_ref__for__classic_fa2_dissuade_hubs | 3.530052760e-08 | grid4x4 | BIT_EXACT | 0/42 | 2026-06-02T22:48:50+00:00 |
| classic_fa2_default | fa2_ref__for__classic_fa2_default | 3.530052760e-08 | grid4x4 | BIT_EXACT | 0/42 | 2026-06-02T22:48:20+00:00 |
| classic_fa2_barnes_hut | fa2_ref__for__classic_fa2_barnes_hut | 3.530052760e-08 | grid4x4 | BIT_EXACT | 0/42 | 2026-06-02T22:48:58+00:00 |
| classic_pivot_mds_10 | ogdf_pivot_mds__for__classic_pivot_mds_10 | 3.484338453e-08 | small_random_12 | BIT_EXACT | 0/42 | 2026-06-02T22:56:03+00:00 |
| classic_lgl_default | igraph_lgl__for__classic_lgl_default | 3.345951645e-08 | small_random_12 | BIT_EXACT | 0/42 | 2026-06-02T23:01:42+00:00 |
| classic_fa2_no_outbound | fa2_ref__for__classic_fa2_no_outbound | 3.326113960e-08 | cycle6 | BIT_EXACT | 0/42 | 2026-06-02T22:48:45+00:00 |
| classic_fa2_scaling4 | fa2_ref__for__classic_fa2_scaling4 | 3.229567968e-08 | small_random_12 | BIT_EXACT | 0/42 | 2026-06-02T22:48:37+00:00 |
| classic_fa2_linlog | fa2_ref__for__classic_fa2_linlog | 3.159291798e-08 | cycle6 | BIT_EXACT | 0/42 | 2026-06-02T22:48:54+00:00 |
| classic_fa2_gravity2 | fa2_ref__for__classic_fa2_gravity2 | 3.134642364e-08 | small_random_12 | BIT_EXACT | 0/42 | 2026-06-02T22:48:29+00:00 |
| classic_lgl_iter300 | igraph_lgl__for__classic_lgl_iter300 | 3.048904538e-08 | ladder5 | BIT_EXACT | 0/42 | 2026-06-02T23:02:17+00:00 |
| classic_pivot_mds_50 | ogdf_pivot_mds__for__classic_pivot_mds_50 | 3.046587601e-08 | balanced_tree_2x3 | BIT_EXACT | 0/42 | 2026-06-02T22:56:08+00:00 |
| classic_pivot_mds_200 | ogdf_pivot_mds__for__classic_pivot_mds_200 | 3.046587601e-08 | balanced_tree_2x3 | BIT_EXACT | 0/42 | 2026-06-02T22:56:19+00:00 |
| classic_pivot_mds_100 | ogdf_pivot_mds__for__classic_pivot_mds_100 | 3.046587601e-08 | balanced_tree_2x3 | BIT_EXACT | 0/42 | 2026-06-02T22:56:13+00:00 |
| classic_lgl_iter50 | igraph_lgl__for__classic_lgl_iter50 | 3.025559426e-08 | small_random_12 | BIT_EXACT | 0/42 | 2026-06-02T23:01:50+00:00 |
| classic_fa2_exact | fa2_ref__for__classic_fa2_exact | 2.980684564e-08 | small_dag_10 | BIT_EXACT | 0/42 | 2026-06-02T22:49:03+00:00 |
| classic_stress_sgd_steps30 | sgd2__for__classic_stress_sgd_steps30 | 2.951697137e-08 | grid4x4 | BIT_EXACT | 0/42 | 2026-06-02T22:49:07+00:00 |
| classic_lgl_cool2 | igraph_lgl__for__classic_lgl_cool2 | 2.806924510e-08 | cycle6 | BIT_EXACT | 0/42 | 2026-06-02T23:02:47+00:00 |
| classic_lgl_cool1 | igraph_lgl__for__classic_lgl_cool1 | 2.780744896e-08 | star8 | BIT_EXACT | 0/42 | 2026-06-02T23:02:31+00:00 |
| classic_fa2_scaling1 | fa2_ref__for__classic_fa2_scaling1 | 2.543431409e-08 | path8 | BIT_EXACT | 0/42 | 2026-06-02T22:48:33+00:00 |
| classic_linlog_power | linlog__for__classic_linlog_power | 1.290871704e-08 | path8 | BIT_EXACT | 42/42 | 2026-06-02T22:55:30+00:00 |
| classic_neulay_radius08 | neulay__for__classic_neulay_radius08 | 7.561034184e-16 | ladder5 | BIT_EXACT | 42/42 | 2026-06-02T23:30:24+00:00 |
| classic_neulay_lr001 | neulay__for__classic_neulay_lr001 | 6.636105244e-16 | star8 | BIT_EXACT | 42/42 | 2026-06-02T23:25:18+00:00 |
| classic_linlog_quadratic | linlog__for__classic_linlog_quadratic | 6.631092523e-16 | path8 | BIT_EXACT | 42/42 | 2026-06-02T22:55:17+00:00 |
| classic_neulay_default | neulay__for__classic_neulay_default | 6.628369095e-16 | small_dag_10 | BIT_EXACT | 42/42 | 2026-06-02T23:23:37+00:00 |
| classic_neulay_radius02 | neulay__for__classic_neulay_radius02 | 6.592679578e-16 | small_dag_10 | BIT_EXACT | 42/42 | 2026-06-02T23:28:41+00:00 |
| classic_neulay_lr05 | neulay__for__classic_neulay_lr05 | 6.554459226e-16 | ladder5 | BIT_EXACT | 42/42 | 2026-06-02T23:27:00+00:00 |
| classic_fr_steps200 | nx_spring__for__classic_fr_steps200 | 6.489628114e-16 | ladder5 | BIT_EXACT | 42/42 | 2026-06-02T22:48:07+00:00 |
| classic_tsnet_steps2000 | tsne_graph__for__classic_tsnet_steps2000 | 6.457739684e-16 | petersen_10 | BIT_EXACT | 42/42 | 2026-06-02T22:51:34+00:00 |
| classic_fr_steps500 | nx_spring__for__classic_fr_steps500 | 6.275600440e-16 | two_triangles_bridge | BIT_EXACT | 42/42 | 2026-06-02T22:48:11+00:00 |
| classic_tsnet_steps200 | tsne_graph__for__classic_tsnet_steps200 | 6.204783605e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T22:51:16+00:00 |
| classic_neato_graphviz_fidelity | graphviz_neato__for__classic_neato_graphviz_fidelity | 6.073117992e-16 | path8 | BIT_EXACT | 42/42 | 2026-06-02T22:50:01+00:00 |
| classic_neato | graphviz_neato__for__classic_neato | 6.073117992e-16 | path8 | BIT_EXACT | 42/42 | 2026-06-02T22:49:51+00:00 |
| classic_tsnet_perp50 | tsne_graph__for__classic_tsnet_perp50 | 6.049187461e-16 | star8 | BIT_EXACT | 42/42 | 2026-06-02T22:51:10+00:00 |
| classic_tsnet_default | tsne_graph__for__classic_tsnet_default | 6.049187461e-16 | star8 | BIT_EXACT | 42/42 | 2026-06-02T22:50:54+00:00 |
| classic_umap_spread2 | umap_graph__for__classic_umap_spread2 | 5.804328758e-16 | balanced_tree_2x3 | BIT_EXACT | 42/42 | 2026-06-02T23:21:56+00:00 |
| classic_neulay_no_gcn | neulay__for__classic_neulay_no_gcn | 5.776915684e-16 | path8 | BIT_EXACT | 42/42 | 2026-06-02T23:32:00+00:00 |
| classic_stress_maj_iter50 | ogdf_stress__for__classic_stress_maj_iter50 | 5.722379215e-16 | small_dag_10 | BIT_EXACT | 42/42 | 2026-06-02T22:50:06+00:00 |
| classic_maxent_stress_steps50 | ogdf_stress__for__classic_maxent_stress_steps50 | 5.722379215e-16 | small_dag_10 | BIT_EXACT | 42/42 | 2026-06-02T22:52:52+00:00 |
| classic_fr_steps100 | nx_spring__for__classic_fr_steps100 | 5.613641232e-16 | grid4x4 | BIT_EXACT | 42/42 | 2026-06-02T22:48:03+00:00 |
| classic_umap_mindist001 | umap_graph__for__classic_umap_mindist001 | 5.603344797e-16 | two_triangles_bridge | BIT_EXACT | 42/42 | 2026-06-02T23:15:48+00:00 |
| classic_linlog_default | linlog__for__classic_linlog_default | 5.523289809e-16 | two_triangles_bridge | BIT_EXACT | 42/42 | 2026-06-02T22:55:05+00:00 |
| classic_linlog_steps100 | linlog__for__classic_linlog_steps100 | 5.031056976e-16 | two_triangles_bridge | BIT_EXACT | 42/42 | 2026-06-02T22:55:38+00:00 |
| classic_rt_default | igraph_rt__for__classic_rt_default | 4.976691580e-16 | ladder5 | BIT_EXACT | 14/14 | 2026-06-02T22:56:20+00:00 |
| classic_maxent_stress_steps400 | ogdf_stress__for__classic_maxent_stress_steps400 | 4.892781337e-16 | path8 | BIT_EXACT | 42/42 | 2026-06-02T22:53:06+00:00 |
| classic_umap_mindist05 | umap_graph__for__classic_umap_mindist05 | 4.865056653e-16 | cycle6 | BIT_EXACT | 42/42 | 2026-06-02T23:18:51+00:00 |
| classic_umap_nn5 | umap_graph__for__classic_umap_nn5 | 4.811942818e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T23:09:42+00:00 |
| classic_umap_nn30 | umap_graph__for__classic_umap_nn30 | 4.811942818e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T23:12:45+00:00 |
| classic_umap_default | umap_graph__for__classic_umap_default | 4.811942818e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T23:06:37+00:00 |
| classic_stress_maj_iter500 | ogdf_stress__for__classic_stress_maj_iter500 | 4.775249788e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T22:50:23+00:00 |
| classic_stress_maj_default | ogdf_stress__for__classic_stress_maj_default | 4.775249788e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T22:49:40+00:00 |
| classic_maxent_stress_entropy | ogdf_stress__for__classic_maxent_stress_entropy | 4.775249788e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T22:52:37+00:00 |
| classic_maxent_stress_default | ogdf_stress__for__classic_maxent_stress_default | 4.775249788e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T22:52:27+00:00 |
| classic_maxent_stress_alpha2 | ogdf_stress__for__classic_maxent_stress_alpha2 | 4.775249788e-16 | wheel7 | BIT_EXACT | 42/42 | 2026-06-02T22:52:46+00:00 |
| classic_tsnet_perp5 | tsne_graph__for__classic_tsnet_perp5 | 4.514149147e-16 | grid3x3 | BIT_EXACT | 42/42 | 2026-06-02T22:51:02+00:00 |
| classic_kk_steps300 | nx_kamada_kawai__for__classic_kk_steps300 | 4.494120179e-16 | grid4x4 | BIT_EXACT | 14/14 | 2026-06-02T22:48:14+00:00 |
| classic_kk_steps1000 | nx_kamada_kawai__for__classic_kk_steps1000 | 4.494120179e-16 | grid4x4 | BIT_EXACT | 14/14 | 2026-06-02T22:48:16+00:00 |
| classic_kk_steps100 | nx_kamada_kawai__for__classic_kk_steps100 | 4.494120179e-16 | grid4x4 | BIT_EXACT | 14/14 | 2026-06-02T22:48:13+00:00 |
| classic_linlog_steps500 | linlog__for__classic_linlog_steps500 | 4.472223437e-16 | path8 | BIT_EXACT | 42/42 | 2026-06-02T22:55:56+00:00 |
| classic_fr_steps50 | nx_spring__for__classic_fr_steps50 | 4.424055047e-16 | cycle6 | BIT_EXACT | 42/42 | 2026-06-02T22:47:59+00:00 |
| classic_spectral_nx_fidelity | nx_spectral__for__classic_spectral_nx_fidelity | 3.186921264e-16 | small_dag_10 | BIT_EXACT | 14/14 | 2026-06-02T22:49:24+00:00 |
| classic_spectral_default | nx_spectral__for__classic_spectral_default | 3.186921264e-16 | small_dag_10 | BIT_EXACT | 14/14 | 2026-06-02T22:49:23+00:00 |
| gephi_yifanhu_default | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-03T00:17:12+00:00 |
| cytoscape_fcose_quality | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-03T00:17:12+00:00 |
| cytoscape_fcose_default | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-03T00:17:12+00:00 |
| classic_spectral_unnormalized | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-02T22:49:24+00:00 |
| classic_spectral_random_walk | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-02T22:49:24+00:00 |
| classic_rt_horizontal | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-02T22:56:20+00:00 |
| classic_kk_fr_long | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-02T22:48:16+00:00 |
| classic_kk_fr_default | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-02T22:48:16+00:00 |
| classic_fr_kk_long | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-02T22:48:16+00:00 |
| classic_fr_kk_default | no_reference | -- |  | NO_REFERENCE | 0/0 | 2026-06-02T22:48:16+00:00 |
