# R41 Reference-Pairing Audit

Scope: audited every entry in `dagua/eval/variants.py` against the pipeline docstring target and registered competitor adapters. Pipeline/layout files were read only.

## Verdict Counts

- OPTIMAL: 106
- ACCEPTABLE: 14
- SUBOPTIMAL: 0

## Re-pairs Applied

| variant_id | before | after | justification | expected RMSD shift |
| --- | --- | --- | --- | --- |
| classic_spectral_unnormalized | None | nx_spectral | The spectral pipeline targets NetworkX spectral_layout; this variant selects the unnormalized Laplacian that NetworkX uses. | From unpaired/no RMSD to near NetworkX-fidelity residual; expected near-zero median after Procrustes alignment. |
| classic_fmmm_steps10 | ogdf_fmmm | graphviz_fdp | The FMMM pipeline docstring names Graphviz fdp/Hachul-Junger as the fidelity target, and graphviz_fdp is registered. | Expected lower algorithm-target RMSD than the OGDF proxy; likely modest-to-material improvement toward the documented 0.067-0.179 median band for step variants. |
| classic_fmmm_steps100 | ogdf_fmmm | graphviz_fdp | Same FMMM target/reference mismatch as steps10. | Expected lower algorithm-target RMSD than the OGDF proxy; likely modest-to-material improvement toward the documented 0.067-0.179 median band for step variants. |
| classic_fmmm_steps200 | ogdf_fmmm | graphviz_fdp | Same FMMM target/reference mismatch as steps10. | Expected lower algorithm-target RMSD than the OGDF proxy; likely modest-to-material improvement toward the documented 0.067-0.179 median band for step variants. |
| classic_rt_horizontal | None | igraph_rt | Reingold-Tilford horizontal mode is a presentation rotation; the algorithmic target remains igraph Reingold-Tilford. | From unpaired/no RMSD to the existing RT fidelity residual; expected near-zero median because alignment handles rotation/reflection. |

## Proposed But Not Applied

| variant_id(s) | candidate | reason deferred |
| --- | --- | --- |
| classic_fr_kk_default, classic_fr_kk_long, classic_kk_fr_default, classic_kk_fr_long | Chain of nx_spring and nx_kamada_kawai | Would require a new original-side chain adapter, not a mechanical original_engine edit. |
| classic_spectral_random_walk | New random-walk spectral reference mode | No registered external competitor exposes this mode through variant params. |
| classic_maxent_stress_* | True OGDF maxent-stress adapter | Current registry only has ogdf_stress; adding maxent-stress support is adapter work, not a pairing edit. |
| classic_fa2_dissuade_hubs | fa2_ref with dissuade-hubs support | Current fa2_ref parameter mapping does not expose that knob; changing this would require adapter validation. |

## Full Audit Table

| variant_id | base_engine | current_ref | algorithm implemented | verdict | note |
| --- | --- | --- | --- | --- | --- |
| classic_fr_steps50 | classic_fr | nx_spring | Fruchterman-Reingold; targets NetworkX spring_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fr_steps100 | classic_fr | nx_spring | Fruchterman-Reingold; targets NetworkX spring_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fr_steps200 | classic_fr | nx_spring | Fruchterman-Reingold; targets NetworkX spring_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fr_steps500 | classic_fr | nx_spring | Fruchterman-Reingold; targets NetworkX spring_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_kk_steps100 | classic_kk | nx_kamada_kawai | Kamada-Kawai; targets NetworkX kamada_kawai_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_kk_steps300 | classic_kk | nx_kamada_kawai | Kamada-Kawai; targets NetworkX kamada_kawai_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_kk_steps1000 | classic_kk | nx_kamada_kawai | Kamada-Kawai; targets NetworkX kamada_kawai_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fr_kk_default | classic_fr_kk | None | FR warm start followed by KK refinement; no single original adapter. | ACCEPTABLE | No registered two-stage FR->KK original adapter. |
| classic_fr_kk_long | classic_fr_kk | None | FR warm start followed by KK refinement; no single original adapter. | ACCEPTABLE | No registered two-stage FR->KK original adapter. |
| classic_kk_fr_default | classic_kk_fr | None | KK warm start followed by FR refinement; no single original adapter. | ACCEPTABLE | No registered two-stage KK->FR original adapter. |
| classic_kk_fr_long | classic_kk_fr | None | KK warm start followed by FR refinement; no single original adapter. | ACCEPTABLE | No registered two-stage KK->FR original adapter. |
| classic_fa2_default | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_gravity0 | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_gravity2 | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_scaling1 | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_scaling4 | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_strong_gravity | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_no_outbound | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_dissuade_hubs | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | ACCEPTABLE | Current fa2_ref mapping does not expose dissuade-hubs; fa2_ref remains closest available FA2 reference. |
| classic_fa2_linlog | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_barnes_hut | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fa2_exact | classic_fa2 | fa2_ref | ForceAtlas2; targets fa2 1.1.2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_stress_sgd_steps30 | classic_stress_sgd | sgd2 | Stress-SGD; targets s_gd2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_stress_sgd_steps300 | classic_stress_sgd | sgd2 | Stress-SGD; targets s_gd2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_stress_sgd_eps001 | classic_stress_sgd | sgd2 | Stress-SGD; targets s_gd2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_stress_sgd_eps01 | classic_stress_sgd | sgd2 | Stress-SGD; targets s_gd2. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_spectral_default | classic_spectral | nx_spectral | Spectral layout; targets NetworkX spectral_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_spectral_nx_fidelity | classic_spectral | nx_spectral | Spectral layout; targets NetworkX spectral_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_spectral_random_walk | classic_spectral | None | Spectral layout; targets NetworkX spectral_layout. | ACCEPTABLE | NetworkX spectral reference does not expose a random-walk Laplacian mode. |
| classic_spectral_unnormalized | classic_spectral | nx_spectral | Spectral layout; targets NetworkX spectral_layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_classical_mds_default | classic_classical_mds | igraph_mds | Classical metric MDS; targets igraph MDS. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_classical_mds_igraph_fidelity | classic_classical_mds | igraph_mds | Classical metric MDS; targets igraph MDS. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_stress_maj_default | classic_stress_maj | ogdf_stress | SMACOF stress majorization; targets OGDF stress and Graphviz neato modes. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neato | classic_neato | graphviz_neato | Graphviz neato-compatible stress majorization. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neato_graphviz_fidelity | classic_neato | graphviz_neato | Graphviz neato-compatible stress majorization. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_stress_maj_iter50 | classic_stress_maj | ogdf_stress | SMACOF stress majorization; targets OGDF stress and Graphviz neato modes. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_stress_maj_iter500 | classic_stress_maj | ogdf_stress | SMACOF stress majorization; targets OGDF stress and Graphviz neato modes. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sugiyama_default | classic_sugiyama | igraph_sugiyama | Sugiyama layered drawing; targets igraph Sugiyama or Graphviz dot by fidelity mode. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sugiyama_graphviz_fidelity | classic_sugiyama | graphviz_dot | Sugiyama layered drawing; targets igraph Sugiyama or Graphviz dot by fidelity mode. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sugiyama_passes4 | classic_sugiyama | igraph_sugiyama | Sugiyama layered drawing; targets igraph Sugiyama or Graphviz dot by fidelity mode. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sugiyama_passes48 | classic_sugiyama | igraph_sugiyama | Sugiyama layered drawing; targets igraph Sugiyama or Graphviz dot by fidelity mode. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sugiyama_wide | classic_sugiyama | igraph_sugiyama | Sugiyama layered drawing; targets igraph Sugiyama or Graphviz dot by fidelity mode. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sugiyama_tight | classic_sugiyama | igraph_sugiyama | Sugiyama layered drawing; targets igraph Sugiyama or Graphviz dot by fidelity mode. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_tsnet_default | classic_tsnet | tsne_graph | tsNET graph layout; targets sklearn t-SNE graph adapter. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_tsnet_perp5 | classic_tsnet | tsne_graph | tsNET graph layout; targets sklearn t-SNE graph adapter. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_tsnet_perp50 | classic_tsnet | tsne_graph | tsNET graph layout; targets sklearn t-SNE graph adapter. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_tsnet_steps200 | classic_tsnet | tsne_graph | tsNET graph layout; targets sklearn t-SNE graph adapter. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_tsnet_steps2000 | classic_tsnet | tsne_graph | tsNET graph layout; targets sklearn t-SNE graph adapter. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_gem_iters100 | classic_gem | ogdf_gem | GEM graph embedder; targets OGDF GEMLayout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_gem_iters500 | classic_gem | ogdf_gem | GEM graph embedder; targets OGDF GEMLayout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_gem_iters2000 | classic_gem | ogdf_gem | GEM graph embedder; targets OGDF GEMLayout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fmmm_steps10 | classic_fmmm | graphviz_fdp | FM3 multilevel force-directed; pipeline docstring targets Graphviz fdp/Hachul-Junger. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fmmm_steps100 | classic_fmmm | graphviz_fdp | FM3 multilevel force-directed; pipeline docstring targets Graphviz fdp/Hachul-Junger. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fmmm_steps200 | classic_fmmm | graphviz_fdp | FM3 multilevel force-directed; pipeline docstring targets Graphviz fdp/Hachul-Junger. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_maxent_stress_default | classic_maxent_stress | ogdf_stress | MaxEnt-Stress; targets OGDF maxent-stress; current registry has OGDF stress proxy only. | ACCEPTABLE | No registered OGDF maxent-stress adapter; ogdf_stress is the closest available OGDF stress-family proxy. |
| classic_maxent_stress_entropy | classic_maxent_stress | ogdf_stress | MaxEnt-Stress; targets OGDF maxent-stress; current registry has OGDF stress proxy only. | ACCEPTABLE | No registered OGDF maxent-stress adapter; ogdf_stress is the closest available OGDF stress-family proxy. |
| classic_maxent_stress_alpha2 | classic_maxent_stress | ogdf_stress | MaxEnt-Stress; targets OGDF maxent-stress; current registry has OGDF stress proxy only. | ACCEPTABLE | No registered OGDF maxent-stress adapter; ogdf_stress is the closest available OGDF stress-family proxy. |
| classic_maxent_stress_steps50 | classic_maxent_stress | ogdf_stress | MaxEnt-Stress; targets OGDF maxent-stress; current registry has OGDF stress proxy only. | ACCEPTABLE | No registered OGDF maxent-stress adapter; ogdf_stress is the closest available OGDF stress-family proxy. |
| classic_maxent_stress_steps400 | classic_maxent_stress | ogdf_stress | MaxEnt-Stress; targets OGDF maxent-stress; current registry has OGDF stress proxy only. | ACCEPTABLE | No registered OGDF maxent-stress adapter; ogdf_stress is the closest available OGDF stress-family proxy. |
| classic_davidson_harel_rounds50 | classic_davidson_harel | igraph_davidson_harel | Davidson-Harel simulated annealing; targets igraph Davidson-Harel. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_davidson_harel_rounds100 | classic_davidson_harel | igraph_davidson_harel | Davidson-Harel simulated annealing; targets igraph Davidson-Harel. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_davidson_harel_rounds200 | classic_davidson_harel | igraph_davidson_harel | Davidson-Harel simulated annealing; targets igraph Davidson-Harel. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_linlog_default | classic_linlog | linlog | LinLog force layout; targets LinLog reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_linlog_quadratic | classic_linlog | linlog | LinLog force layout; targets LinLog reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_linlog_power | classic_linlog | linlog | LinLog force layout; targets LinLog reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_linlog_steps100 | classic_linlog | linlog | LinLog force layout; targets LinLog reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_linlog_steps500 | classic_linlog | linlog | LinLog force layout; targets LinLog reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_pivot_mds_10 | classic_pivot_mds | ogdf_pivot_mds | Pivot-MDS; targets OGDF PivotMDS. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_pivot_mds_50 | classic_pivot_mds | ogdf_pivot_mds | Pivot-MDS; targets OGDF PivotMDS. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_pivot_mds_100 | classic_pivot_mds | ogdf_pivot_mds | Pivot-MDS; targets OGDF PivotMDS. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_pivot_mds_200 | classic_pivot_mds | ogdf_pivot_mds | Pivot-MDS; targets OGDF PivotMDS. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_rt_default | classic_rt | igraph_rt | Reingold-Tilford tidy tree; targets igraph Reingold-Tilford. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_rt_horizontal | classic_rt | igraph_rt | Reingold-Tilford tidy tree; targets igraph Reingold-Tilford. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_graphopt_default | classic_graphopt | igraph_graphopt | GraphOpt; targets igraph GraphOpt. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_graphopt_charge_low | classic_graphopt | igraph_graphopt | GraphOpt; targets igraph GraphOpt. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_graphopt_charge_high | classic_graphopt | igraph_graphopt | GraphOpt; targets igraph GraphOpt. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_graphopt_mass_low | classic_graphopt | igraph_graphopt | GraphOpt; targets igraph GraphOpt. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_graphopt_mass_high | classic_graphopt | igraph_graphopt | GraphOpt; targets igraph GraphOpt. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_graphopt_spring2 | classic_graphopt | igraph_graphopt | GraphOpt; targets igraph GraphOpt. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_drl_default | classic_drl | igraph_drl | Distributed Recursive Layout; targets igraph DrL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_drl_coarsen | classic_drl | igraph_drl | Distributed Recursive Layout; targets igraph DrL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_drl_coarsest | classic_drl | igraph_drl | Distributed Recursive Layout; targets igraph DrL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_drl_refine | classic_drl | igraph_drl | Distributed Recursive Layout; targets igraph DrL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_drl_final | classic_drl | igraph_drl | Distributed Recursive Layout; targets igraph DrL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_lgl_default | classic_lgl | igraph_lgl | Large Graph Layout; targets igraph LGL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_lgl_iter50 | classic_lgl | igraph_lgl | Large Graph Layout; targets igraph LGL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_lgl_iter300 | classic_lgl | igraph_lgl | Large Graph Layout; targets igraph LGL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_lgl_cool1 | classic_lgl | igraph_lgl | Large Graph Layout; targets igraph LGL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_lgl_cool2 | classic_lgl | igraph_lgl | Large Graph Layout; targets igraph LGL. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sfdp_default | classic_sfdp | graphviz_sfdp | SFDP multilevel force-directed; targets Graphviz sfdp. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sfdp_graphviz_fidelity | classic_sfdp | graphviz_sfdp | SFDP multilevel force-directed; targets Graphviz sfdp. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sfdp_theta04 | classic_sfdp | graphviz_sfdp | SFDP multilevel force-directed; targets Graphviz sfdp. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sfdp_theta08 | classic_sfdp | graphviz_sfdp | SFDP multilevel force-directed; targets Graphviz sfdp. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sfdp_p_neg2 | classic_sfdp | graphviz_sfdp | SFDP multilevel force-directed; targets Graphviz sfdp. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sfdp_steps200 | classic_sfdp | graphviz_sfdp | SFDP multilevel force-directed; targets Graphviz sfdp. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_umap_default | classic_umap | umap_graph | UMAP graph layout; targets umap-learn graph layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_umap_nn5 | classic_umap | umap_graph | UMAP graph layout; targets umap-learn graph layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_umap_nn30 | classic_umap | umap_graph | UMAP graph layout; targets umap-learn graph layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_umap_mindist001 | classic_umap | umap_graph | UMAP graph layout; targets umap-learn graph layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_umap_mindist05 | classic_umap | umap_graph | UMAP graph layout; targets umap-learn graph layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_umap_spread2 | classic_umap | umap_graph | UMAP graph layout; targets umap-learn graph layout. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neulay_default | classic_neulay | neulay | NeuLay two-phase graph layout; targets NeuLay old_code/NeuLay-2.py. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neulay_lr001 | classic_neulay | neulay | NeuLay two-phase graph layout; targets NeuLay old_code/NeuLay-2.py. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neulay_lr05 | classic_neulay | neulay | NeuLay two-phase graph layout; targets NeuLay old_code/NeuLay-2.py. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neulay_radius02 | classic_neulay | neulay | NeuLay two-phase graph layout; targets NeuLay old_code/NeuLay-2.py. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neulay_radius08 | classic_neulay | neulay | NeuLay two-phase graph layout; targets NeuLay old_code/NeuLay-2.py. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_neulay_no_gcn | classic_neulay | neulay | NeuLay two-phase graph layout; targets NeuLay old_code/NeuLay-2.py. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_default | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_stress_only | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_with_crossing | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_with_aspect | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_lr001 | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_lr01 | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_batch8 | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_sgd2_multi_batch128 | classic_sgd2_multi | sgd2_multi_ref | SGD2 multicriteria; targets historical sgd2_multi reference. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fcose_default | classic_fcose | cytoscape_fcose | fCoSE; targets Cytoscape fCoSE. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| classic_fcose_proof | classic_fcose | cytoscape_fcose | fCoSE; targets Cytoscape fCoSE. | OPTIMAL | Best available algorithmic reference after R41 audit. |
| cytoscape_fcose_default | cytoscape_fcose | None | External Cytoscape fCoSE reference variant; not a Dagua reimplementation. | ACCEPTABLE | Reference-side Cytoscape variant; no original side applies. |
| cytoscape_fcose_quality | cytoscape_fcose | None | External Cytoscape fCoSE reference variant; not a Dagua reimplementation. | ACCEPTABLE | Reference-side Cytoscape variant; no original side applies. |
| gephi_yifanhu_default | gephi_yifanhu | None | External Gephi Yifan Hu reference variant; not a Dagua reimplementation. | ACCEPTABLE | Reference-side Gephi variant; no original side applies. |
