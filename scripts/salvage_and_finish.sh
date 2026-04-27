#!/bin/bash
# Salvage round: retry every feasibly-recoverable benchmark row we haven't
# already gotten, then run the full post-benchmark pipeline. No time limit --
# user wants this to be the LAST round.
#
# Expected retry scope (via --resume picking up cleared rows):
#   - 9,449 overnight_time_limit_5am (mostly neulay on small/medium graphs)
#   - 10,450 "skipped after 3 consecutive errors" pairs that succeed elsewhere
#   - 1,439 watchdog-residual errors from the overnight run
#   Total ~21.3k items.
#
# Uses workers=4 and MAX_INFLIGHT_GROUPS=32 (set in run_benchmark.py).
set -uo pipefail

export PATH="/home/jtaylor/anaconda3/envs/py311/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jtaylor/projects/dagua

echo "=== [salvage] started at $(date -Iseconds) ==="

# --- Step 1: benchmark retry ------------------------------------------------
echo "=== [salvage] step 1: benchmark --resume (~21.3k items) ==="
python scripts/run_benchmark.py \
    --resume \
    --workers 4 \
    --timeout 120 \
    --watchdog-timeout 600 \
    --variants \
    --additive-variants \
    --output-dir eval_output/variant_bench_full \
    --seeds 60 \
    --engines classic_davidson_harel,classic_davidson_harel_rounds100,classic_davidson_harel_rounds200,classic_davidson_harel_rounds50,classic_drl,classic_drl_coarsen,classic_drl_coarsest,classic_drl_default,classic_drl_final,classic_drl_refine,classic_fa2,classic_fa2_barnes_hut,classic_fa2_default,classic_fa2_exact,classic_fa2_gravity0,classic_fa2_gravity2,classic_fa2_linlog,classic_fa2_scaling1,classic_fa2_scaling4,classic_fa2_strong_gravity,classic_fmmm,classic_fmmm_steps10,classic_fmmm_steps100,classic_fr,classic_fr_kk,classic_fr_kk_default,classic_fr_kk_long,classic_fr_steps500,classic_gem_iters2000,classic_graphopt,classic_graphopt_charge_high,classic_graphopt_charge_low,classic_graphopt_default,classic_graphopt_mass_high,classic_graphopt_mass_low,classic_graphopt_spring2,classic_kk_fr,classic_lgl,classic_lgl_cool1,classic_lgl_cool2,classic_lgl_default,classic_lgl_iter300,classic_lgl_iter50,classic_linlog,classic_linlog_default,classic_linlog_power,classic_linlog_quadratic,classic_linlog_steps500,classic_maxent_stress,classic_maxent_stress_alpha2,classic_maxent_stress_default,classic_maxent_stress_entropy,classic_maxent_stress_steps400,classic_neulay,classic_neulay_default,classic_neulay_lr001,classic_neulay_lr05,classic_neulay_no_gcn,classic_neulay_radius02,classic_neulay_radius08,classic_sfdp,classic_sfdp_default,classic_sfdp_p_neg2,classic_sfdp_steps200,classic_sfdp_theta04,classic_sfdp_theta08,classic_sgd2_multi,classic_sgd2_multi_with_crossing,classic_stress_sgd,classic_sugiyama,classic_sugiyama_passes48,classic_sugiyama_tight,classic_sugiyama_wide,classic_tsnet,classic_tsnet_default,classic_tsnet_perp5,classic_tsnet_perp50,classic_tsnet_steps200,classic_tsnet_steps2000,classic_umap,classic_umap_default,classic_umap_mindist001,classic_umap_mindist05,classic_umap_nn30,classic_umap_nn5,classic_umap_spread2,cytoscape_fcose,graphviz_fdp,igraph_davidson_harel,igraph_davidson_harel__for__classic_davidson_harel_rounds100,igraph_davidson_harel__for__classic_davidson_harel_rounds200,igraph_davidson_harel__for__classic_davidson_harel_rounds50,neulay,neulay__for__classic_neulay_default,neulay__for__classic_neulay_lr001,neulay__for__classic_neulay_lr05,neulay__for__classic_neulay_no_gcn,neulay__for__classic_neulay_radius02,neulay__for__classic_neulay_radius08,nx_spring__for__classic_fr_steps500,sgd2_multi_ref,sgd2_multi_ref__for__classic_sgd2_multi_batch128,sgd2_multi_ref__for__classic_sgd2_multi_batch8,sgd2_multi_ref__for__classic_sgd2_multi_default,sgd2_multi_ref__for__classic_sgd2_multi_lr001,sgd2_multi_ref__for__classic_sgd2_multi_lr01,sgd2_multi_ref__for__classic_sgd2_multi_stress_only,sgd2_multi_ref__for__classic_sgd2_multi_with_aspect,sgd2_multi_ref__for__classic_sgd2_multi_with_crossing,tsne_graph,tsne_graph__for__classic_tsnet_default,tsne_graph__for__classic_tsnet_perp5,tsne_graph__for__classic_tsnet_perp50,tsne_graph__for__classic_tsnet_steps200,tsne_graph__for__classic_tsnet_steps2000 \
    --graphs asymmetric_hourglass_hub,ba_2000,ba_500,ba_5000,binary_tree,bipartite_4_3_4,braided_feedback_tails,broken_symmetry_residual_pair,center_port_backedge_hub,chung_lu_150,citation_dag_300,cluster_member_style_stress,clustered_longlabel_handoffs,clustered_medium_5x20,complete_bipartite_8x12,compound_10x20,compound_dag_5x30,deep_chain_20,dense_pair_50,densenet_block,dependency_500,dependency_graph_100,disconnected_encoder_residual,disconnected_label_cycle_collage,edge_label_braid,er_100,er_2000,er_500,extreme_mixed_width_transformer,grid_20x20,grid_50x50,grid_5x5,grid_rect_6x8,heavy_tail_weights_50,hexagonal_lattice_42,hierarchical_residual_stage,hub_and_spoke_3x20,hub_fanout_label_skew,hub_skip_superfan,hub_spoke_10x20,hub_spoke_5x50,inception_block,interleaved_cluster_crosstalk,kitchen_sink_hybrid_net,kitchen_sink_platform_graph,linear_3layer_mlp,long_range_residual_ladder,long_skip_only_24,mixed_width_labels,moe_router_sparse,multi_component_80,multiscale_skip_cascade,nested_cluster_label_stack,nested_shallow_enc_dec,org_chart_1_5_4_8,org_chart_deep,outerplanar_dag_20,parallel_cycles_4x5,parallel_multiedge_bundle,petersen_10,planar_60,powerlaw_2000,powerlaw_500,protein_ppi_200,ragged_feature_pyramid,random_bipartite_60,random_dag_200,random_dag_50,real_football_115,real_karate_34,real_lesmis_77,recurrent_feedback_cell,regular_3_30,regular_4_40,residual_block,resnet_stack_4x16,rgg_100,rgg_2000,rgg_500,sbm_4x30,sbm_5x50,sbm_8x100,scale_free_ba_120,shape_and_routing_matrix,sierpinski_42,small_label_storm,small_world_100,small_world_2000,small_world_500,sparse_pair_50,tl_cnn_small,tl_mlp_3layer,tl_resnet_2block,tl_transformer_1layer,transformer_full_4h_2l,transformer_layer,triangular_lattice_36,unet_small,weighted_chain_20,weighted_clusters_3x10,weighted_karate_34,wide_1_100_1,wide_3_50_3,wide_single_layer_1_50_1,width_skew_late_merge
BENCH_EXIT=$?
echo "=== [salvage] benchmark exited $BENCH_EXIT at $(date -Iseconds) ==="

# Sanity: everything complete?
REMAIN=$(python3 -c "
import json
r = json.load(open('eval_output/variant_bench_full/results.json'))
print(sum(1 for v in r.values() if v.get('status') == 'running'))
")
echo "=== [salvage] running rows after benchmark: $REMAIN ==="

# If benchmark crashed midway, flip any leftover running rows so pipeline can proceed
if [ "$REMAIN" -ne 0 ]; then
    echo "[salvage] Flipping $REMAIN residual running rows to skipped (abort-recovery)"
    python3 scripts/flip_running_to_skipped.py --reason salvage_abort_residual
fi

if [ $BENCH_EXIT -ne 0 ]; then
    ~/.claude/scripts/send-to-jmt.sh "Dagua salvage: benchmark exit=$BENCH_EXIT. Will still run post-pipeline on what was saved." || true
fi

# --- Step 2: post-benchmark pipeline ----------------------------------------
echo "=== [salvage] step 2: post-benchmark pipeline ==="
./scripts/post_benchmark_pipeline.sh
POST_EXIT=$?
echo "=== [salvage] post-pipeline exit $POST_EXIT at $(date -Iseconds) ==="

# --- Step 3: notify ---------------------------------------------------------
SUMMARY=$(python3 -c "
import json
r = json.load(open('eval_output/variant_bench_full/results.json'))
from collections import Counter
c = Counter(v.get('status','') for v in r.values())
print(f\"ok={c.get('ok',0):,} err={c.get('error',0):,} skip={c.get('skipped',0):,} to={c.get('timeout',0):,}\")
" 2>/dev/null || echo "summary unavailable")

if [ $POST_EXIT -eq 0 ]; then
    ~/.claude/scripts/send-to-jmt.sh "Dagua salvage DONE: $SUMMARY. Reports in eval_output/report/. Final scope complete -- benchmark bench_exit=$BENCH_EXIT." || true
else
    ~/.claude/scripts/send-to-jmt.sh "Dagua salvage PARTIAL: post-pipeline exit=$POST_EXIT. $SUMMARY." || true
fi

echo "=== [salvage] finished at $(date -Iseconds) post_exit=$POST_EXIT ==="
exit $POST_EXIT
