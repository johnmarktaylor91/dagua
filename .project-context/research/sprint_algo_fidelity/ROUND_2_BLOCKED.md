# Round 2 Blocked: Live Baseline Does Not Match Round 1 Cache

Round 2 stopped before editing Sugiyama source because the required live-vs-cache
sanity gate failed.

The new live comparator initially reproduced the Round 1 methodology bug: it
loaded graphs from `get_test_graphs()` but did not call `compute_node_sizes()`.
The benchmark harness does call `graph.compute_node_sizes()` before invoking a
competitor (`dagua/eval/benchmark.py:974`), so the comparator was corrected to
do the same.

After that correction, live `classic_sugiyama` still differs from Round 1 cached
RMSDs by more than 0.005 on 8 of 22 dot graphs:

| graph | Round 1 cached RMSD | Round 2 live RMSD | delta |
|---|---:|---:|---:|
| cluster_member_style_stress | 0.359973 | 0.420493 | +0.060520 |
| clustered_longlabel_handoffs | 0.340040 | 0.371887 | +0.031847 |
| extreme_mixed_width_transformer | 0.324255 | 0.311750 | -0.012504 |
| hierarchical_residual_stage | 0.318046 | 0.377013 | +0.058967 |
| mixed_width_labels | 0.347613 | 0.404615 | +0.057002 |
| nested_cluster_label_stack | 0.324699 | 0.400884 | +0.076185 |
| shape_and_routing_matrix | 0.437163 | 0.456349 | +0.019186 |
| small_label_storm | 0.474379 | 0.485187 | +0.010808 |

The smallest reproducer is no longer a stable algorithm-only reproducer:
`mixed_width_labels` cached Sugiyama positions place the wide-label nodes at
`x=-11.5`, while the current live run places them at `x=-51.337` after current
node-size measurement. That difference comes before any candidate Sugiyama
algorithm change and is outside the allowed Round 2 fix surface.

Per the prompt's missing-context gate, this round is blocked from source edits
until the benchmark cache is regenerated or the live comparator is given the
exact node-size context used by `eval_output/benchmark_full`.
