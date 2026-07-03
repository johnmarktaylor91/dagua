# RNG Matching Status

Single source of truth for matched-seed small-graph bit-exact checks.

| engine | reference | best(max) RMSD over fixtures&seeds | worst fixture | verdict | exact_match_count/total | timestamp |
|---|---|---:|---|---|---:|---|
| classic_spectral_random_walk | nx_spectral_random_walk__for__classic_spectral_random_walk | 1.270781607e+00 | petersen_10 | DIVERGENT | 3/14 | 2026-06-03T03:56:54+00:00 |
| classic_gem_iters100 | ogdf_gem__for__classic_gem_iters100 | 1.154231678e+00 | grid4x4 | DIVERGENT | 0/42 | 2026-07-02T01:03:52+00:00 |
| classic_gem_iters500 | ogdf_gem__for__classic_gem_iters500 | 6.502790277e-01 | small_random_12 | DIVERGENT | 0/42 | 2026-07-02T01:04:03+00:00 |
| classic_sugiyama_wide | igraph_sugiyama__for__classic_sugiyama_wide | 3.659339770e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-03T03:35:20+00:00 |
| classic_sugiyama_passes48 | igraph_sugiyama__for__classic_sugiyama_passes48 | 3.659339770e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-03T03:35:35+00:00 |
| classic_sugiyama_passes4 | igraph_sugiyama__for__classic_sugiyama_passes4 | 3.659339770e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-03T03:35:27+00:00 |
| classic_sugiyama_default | igraph_sugiyama__for__classic_sugiyama_default | 3.659339770e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-03T12:44:22+00:00 |
| classic_sgd2_multi_default | sgd2_multi_ref__for__classic_sgd2_multi_default | 1.100488958e-01 | petersen_10 | DIVERGENT | 0/42 | 2026-06-03T04:00:38+00:00 |
| classic_fmmm_steps100 | ogdf_fmmm__for__classic_fmmm_steps100 | 2.549335867e-02 | wheel7 | DIVERGENT | 35/42 | 2026-07-02T01:04:51+00:00 |
| classic_sgd2_multi_lr001 | sgd2_multi_ref__for__classic_sgd2_multi_lr001 | 2.283828359e-02 | small_random_12 | DIVERGENT | 0/42 | 2026-06-03T04:10:13+00:00 |
| classic_sgd2_multi_batch128 | sgd2_multi_ref__for__classic_sgd2_multi_batch128 | 9.756672289e-06 | complete5 | CLOSE | 0/42 | 2026-06-03T04:05:27+00:00 |
| classic_rt_horizontal | igraph_rt_horizontal__for__classic_rt_horizontal | 3.597533770e-16 | complete_bipartite_3x3 | BIT_EXACT | 14/14 | 2026-06-03T03:56:50+00:00 |
| classic_spectral_unnormalized | nx_spectral__for__classic_spectral_unnormalized | 3.186921264e-16 | small_dag_10 | BIT_EXACT | 14/14 | 2026-06-03T03:56:45+00:00 |
