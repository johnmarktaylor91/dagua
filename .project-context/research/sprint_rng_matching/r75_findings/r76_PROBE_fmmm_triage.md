# r76 PROBE: FMMM Divergence Triage Against r75 Honest References

Date: 2026-07-03
Scope: research/probe only. Repo was read-only except this results file; scratch artifacts are under `/tmp`.

## Inputs and Commands

- Read first: `.project-context/research/sprint_rng_matching/r75_RESULTS.md`, especially "What r75 discovered".
- Source rows: `eval_output/fidelity_definitive/r75_final.jsonl` filtered to `engine` containing `classic_fmmm`, `quality_identical_raw=false`, and `no_canonical_reference!=true`. Count: 33.
- Dagua saved positions checked in `eval_output/benchmark_100seed_r75_fixes/positions`. I found Dagua FMMM positions there, but no `ogdf_fmmm__for__*` position files in that r75 directory.
- Saved `ogdf_fmmm__for__*` position files exist in `eval_output/benchmark_100seed_seeded_refs`, but their mtimes are 2026-06-12, predating r75. I therefore regenerated probe references directly through the current `scripts/ogdf_runner` adapter instead of treating those tensors as honest.

Commands run:

```bash
rg -n "What r75 discovered|fmmm|FMMM|fixedIterations|0817427|honest" .project-context/research/sprint_rng_matching/r75_RESULTS.md
python /tmp/r76_analyze_fmmm.py > /tmp/r76_leg_breakdown.txt
MPLCONFIGDIR=/tmp/matplotlib-r76 python /tmp/r76_direct_probe.py > /tmp/r76_direct_probe.jsonl
MPLCONFIGDIR=/tmp/matplotlib-r76 python /tmp/r76_direct_all_ogdf.py > /tmp/r76_direct_all_ogdf.jsonl
g++ -std=c++17 /tmp/std_rng_probe.cpp -o /tmp/std_rng_probe && /tmp/std_rng_probe
```

Note: an attempted `scripts/run_benchmark.py --workers 4` probe failed because the sandbox blocks forkserver process creation (`PermissionError: [Errno 1] Operation not permitted`). A `--workers 1` retry later completed (90/90 ok), but the report uses direct adapter calls as the primary evidence because they expose the exact per-variant options and RMSD script outputs.

## Executive Verdict

- The 33 rows split into 30 OGDF FMMM rows and 3 `classic_fmmm_graphviz_fdp_fidelity` rows. The Graphviz-FDP rows are not OGDF FMMM residuals and should be triaged separately.
- OGDF RMSD probe: 24/30 rows are Procrustes near-matches (`median RMSD < 0.01` across 5 seeds). 6/30 are structurally apart.
- Every structurally apart OGDF row is disconnected: `kitchen_sink_platform_graph`, `multi_component_80`, `random_dag_200`, and `random_dag_50` at steps10/100/200.
- Higher iterations are not the general problem. On connected probes, steps100/200 are as close or closer than steps10. `random_dag_50` is apart at all step counts, so its issue is systematic disconnected-component handling, not chaotic growth with iteration count.
- `deep_chain_20::classic_fmmm_steps200` is not a Dagua-vs-current-OGDF layout defect. Direct current-runner probe gives median RMSD `3.94e-17` and max `1.31e-4`; the r75 scoring row has stale/incorrect reference metrics identical to steps10 (`stress_R_mean=0.0275418` for steps10/100/200). Root cause is reference/overlay/scoring artifact for that row, not FMMM geometry.
- RNG stream basics are already aligned: Dagua `_OgdfMt19937.randint` matched a C++ `std::mt19937` + `std::uniform_int_distribution` probe for the first 20 draws. Full bit-exactness is therefore blocked by downstream stream consumption/order/geometry differences in structural disconnected cases, not by the base RNG generator.

## 1. Leg Breakdown

| Failing legs | Rows | Stress rel gap median/min/max | Crossing abs gap median/min/max | NP rel gap median/min/max | Verdict |
|---|---:|---:|---:|---:|---|
| `battery_stress+stress+cross` | 17 | 0.3285/-0.4008/49.58 | 0.5952/-871.4/103.6 | 0/-0.07413/0.6214 | mixed hairline/structural; crossings often margin-discrete |
| `battery_stress+stress+cross+np` | 12 | 1.986/0.155/5.191 | 1.571/0.3571/75.29 | -0.07261/-0.1823/-0.02588 | mixed hairline/structural; crossings often margin-discrete |
| `battery_stress+stress` | 4 | 1.092/-0.9671/44.89 | 0/0/0 | 0.003323/0/0.006646 | stress-only metric/ref artifact candidates |

Per-row leg details:

| Combo | Ref | Disc | Legs | Stress D/R/rel | Cross D/R/abs | Cross margin/self-spread | NP D/R/rel | RMSD verdict |
|---|---|---:|---|---:|---:|---:|---:|---|
| `asymmetric_hourglass_hub::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.04226/0.02319/0.8228 | 0.4524/0/0.4524 | 0.5/0 | 0.9179/0.9327/-0.01586 | near (0.000956) |
| `clustered_longlabel_handoffs::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.01218/0.006768/0.7997 | 0.119/0.07143/0.04762 | 0.5/0.2607 | 1/1/0 | near (8.32e-17) |
| `deep_chain_20::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.1404/0.02754/4.099 | 0.3571/0/0.3571 | 0.5/0 | 0.8123/0.9934/-0.1823 | near (0.000601) |
| `deep_chain_20::classic_fmmm_steps100` | `ogdf_fmmm__for__classic_fmmm_steps100` | N | `battery_stress+stress` | 0.001116/0.02754/-0.9595 | 0/0/0 | 0.5/0 | 1/0.9934/0.006646 | near (6.55e-17) |
| `deep_chain_20::classic_fmmm_steps200` | `ogdf_fmmm__for__classic_fmmm_steps200` | N | `battery_stress+stress` | 0.0009073/0.02754/-0.9671 | 0/0/0 | 0.5/0 | 1/0.9934/0.006646 | near (3.94e-17) |
| `grid_5x5::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.0346/0.01507/1.296 | 1.81/0/1.81 | 0.5/0 | 0.8762/0.9127/-0.03997 | near (3.9e-17) |
| `grid_rect_6x8::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.03754/0.0197/0.9059 | 5.167/1.095/4.071 | 4.962/4.962 | 0.8204/0.8543/-0.03966 | near (0.000465) |
| `heavy_tail_weights_50::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.1079/0.09342/0.155 | 64.14/54.45/9.69 | 6.44/6.44 | 0.4949/0.529/-0.06454 | near (0.000814) |
| `extreme_mixed_width_transformer::classic_fmmm_graphviz_fdp_fidelity` | `graphviz_fdp__for__classic_fmmm_graphviz_fdp_fidelity` | N | `battery_stress+stress+cross` | 0.08938/0.1237/-0.2773 | 1.883/1.5/0.3833 | 1.123/1.123 | 1/1/0 | graphviz-fdp separate |
| `hexagonal_lattice_42::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.03784/0.01792/1.112 | 4.81/2.786/2.024 | 3.065/3.065 | 0.7892/0.8091/-0.02459 | near (0.000628) |
| `kitchen_sink_hybrid_net::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.05085/0.03828/0.3285 | 4.405/4.524/-0.119 | 0.7404/0.7404 | 0.8229/0.84/-0.02029 | near (0.000484) |
| `kitchen_sink_platform_graph::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | Y | `battery_stress+stress+cross+np` | 0.04826/0.01968/1.452 | 0.7857/0.1905/0.5952 | 0.8622/0.8622 | 0.8966/0.925/-0.03075 | structural (0.0572) |
| `long_range_residual_ladder::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.08129/0.01787/3.55 | 3.286/1.214/2.071 | 1.22/1.22 | 0.7763/0.8401/-0.07592 | near (0.000564) |
| `multi_component_80::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | Y | `battery_stress+stress+cross` | 0.07141/0.04483/0.5926 | 2.762/0.3095/2.452 | 0.7805/0.7805 | 0.7558/0.7517/0.005455 | structural (0.0902) |
| `multiscale_skip_cascade::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.07732/0.06489/0.1916 | 10.17/9.571/0.5952 | 1.81/1.81 | 0.857/0.8703/-0.01532 | near (5.32e-17) |
| `nested_cluster_label_stack::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress` | 0.005313/0.001282/3.144 | 0/0/0 | 0.5/0 | 1/1/0 | near (5.56e-17) |
| `parallel_cycles_4x5::classic_fmmm_graphviz_fdp_fidelity` | `graphviz_fdp__for__classic_fmmm_graphviz_fdp_fidelity` | Y | `battery_stress+stress+cross` | 0.07859/0.1311/-0.4008 | 2.4/0.2/2.2 | 0.8762/0.8762 | 0.8985/0.7592/0.1836 | graphviz-fdp separate |
| `random_dag_200::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | Y | `battery_stress+stress+cross` | 0.278/0.2814/-0.01206 | 8210.7/9082.1/-871.4 | 482.6/482.6 | 0.06266/0.05629/0.1131 | structural (0.0505) |
| `random_dag_50::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | Y | `battery_stress+stress+cross+np` | 0.331/0.2839/0.1658 | 454.5/379.2/75.29 | 43.3/43.3 | 0.235/0.263/-0.1065 | structural (0.0987) |
| `random_dag_50::classic_fmmm_steps200` | `ogdf_fmmm__for__classic_fmmm_steps200` | Y | `battery_stress+stress+cross` | 0.334/0.3069/0.08854 | 454.1/350.6/103.6 | 39.71/39.71 | 0.2323/0.2318/0.002185 | structural (0.0865) |
| `random_dag_50::classic_fmmm_steps100` | `ogdf_fmmm__for__classic_fmmm_steps100` | Y | `battery_stress+stress+cross` | 0.3343/0.3148/0.0617 | 457.9/428.0/29.95 | 48.72/48.72 | 0.2322/0.2508/-0.07413 | structural (0.0889) |
| `random_dag_50::classic_fmmm_graphviz_fdp_fidelity` | `graphviz_fdp__for__classic_fmmm_graphviz_fdp_fidelity` | Y | `battery_stress+stress+cross` | 0.2937/0.2652/0.1073 | 365.7/394.2/-28.5 | 42.33/42.33 | 0.2352/0.1451/0.6214 | graphviz-fdp separate |
| `residual_block::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.04097/0.01899/1.157 | 0.2143/0.04762/0.1667 | 0.5/0.3086 | 1/1/0 | near (9.45e-17) |
| `resnet_stack_4x16::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.06242/0.02137/1.921 | 1.357/0.09524/1.262 | 0.5/0.2971 | 0.8325/0.8945/-0.06929 | near (0.000619) |
| `sierpinski_42::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.03517/0.01294/1.718 | 4/0.2619/3.738 | 1.697/1.697 | 0.8549/0.8989/-0.049 | near (0.000603) |
| `sparse_pair_50::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.1506/0.04766/2.159 | 2.643/0.4286/2.214 | 0.9145/0.9145 | 0.8203/0.9245/-0.1127 | near (0.000331) |
| `tl_cnn_small::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.02949/0.0005829/49.58 | 0.04762/0/0.04762 | 0.5/0 | 1/1/0 | near (1.1e-16) |
| `tl_mlp_3layer::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress` | 0.006177/0.0001346/44.89 | 0/0/0 | 0.5/0 | 1/1/0 | near (9.61e-17) |
| `tl_resnet_2block::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.03781/0.01219/2.103 | 0.8095/0.119/0.6905 | 0.5/0.3952 | 0.8962/0.92/-0.02588 | near (0.00101) |
| `tl_transformer_1layer::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.08198/0.02687/2.051 | 2.357/1.024/1.333 | 0.9236/0.9236 | 0.7624/0.8265/-0.07755 | near (0.000603) |
| `transformer_full_4h_2l::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.03873/0.02969/0.3046 | 11.19/10.36/0.8333 | 1.859/1.859 | 0.856/0.8688/-0.01465 | near (0.000471) |
| `triangular_lattice_36::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross` | 0.01225/0.00363/2.373 | 4.429/0/4.429 | 0.5/0 | 0.8513/0.8652/-0.01613 | near (5.32e-17) |
| `weighted_chain_20::classic_fmmm_steps10` | `ogdf_fmmm__for__classic_fmmm_steps10` | N | `battery_stress+stress+cross+np` | 0.1266/0.02045/5.191 | 0.381/0.02381/0.3571 | 0.5/0.1543 | 0.8321/0.994/-0.1629 | near (0.000521) |

Crossing notes: `cross_margin` is often 0.5 on small connected graphs with ref self-spread 0, so a mean gap of 0.05-0.45 crossings can fail despite near-identical coordinates. Large disconnected random DAG rows have real crossing-count gaps: `random_dag_50` steps10 +75.29, steps100 +29.95, steps200 +103.57; `random_dag_200` steps10 -871.42.

## 2. Hairline vs Structural

| Bucket | Rows | Combos |
|---|---:|---|
| OGDF near-match, median RMSD < 0.01 | 24 | mostly connected steps10 rows plus `deep_chain_20` steps100/200 |
| OGDF structural, median RMSD >= 0.01 | 6 | `kitchen_sink_platform_graph::classic_fmmm_steps10` (0.0572), `multi_component_80::classic_fmmm_steps10` (0.0902), `random_dag_200::classic_fmmm_steps10` (0.0505), `random_dag_50::classic_fmmm_steps10` (0.0987), `random_dag_50::classic_fmmm_steps200` (0.0865), `random_dag_50::classic_fmmm_steps100` (0.0889) |
| Graphviz-FDP reference rows | 3 | not included in OGDF FMMM RMSD count |

Interpretation: the connected OGDF rows are not meaningfully apart; they are metric-threshold, discrete-crossing, or stale-reference accounting cases. The structural rows are a compact disconnected-component cluster.

## 3. steps100/steps200 Parity

Direct current-runner probe on 3 shared graphs, 5 seeds each:

| Graph | Steps10 RMSD med/min/max | Steps100 RMSD med/min/max | Steps200 RMSD med/min/max | Verdict |
|---|---:|---:|---:|---|
| `deep_chain_20` | 0.000601/0.000325/0.0015 | 6.55e-17/3.45e-17/0.000134 | 3.94e-17/1.88e-17/0.000131 | near-identical; no iteration divergence |
| `random_dag_50` | 0.105/0.0916/0.109 | 0.09/0.0849/0.0959 | 0.0971/0.0837/0.106 | structural at every step; disconnected handling, not iteration growth |
| `tl_mlp_3layer` | 9.61e-17/1.96e-17/0.00259 | 2.97e-17/4.92e-19/0.00109 | 1.03e-16/4.2e-17/0.00109 | near-identical; no iteration divergence |

Conclusion: divergence is not concentrated at higher step counts. The honest-runner parity is excellent for connected small graphs, including steps100/200. The disconnected random DAG stays apart at all step counts.

## 4. deep_chain_20::classic_fmmm_steps200 Root Cause

r75 row legs: `battery_stress+stress`; crossings and NP direct-equivalent. r75 numbers:

- `stress_D_mean=0.00090733968`, `stress_R_mean=0.027541838`, relative gap `-0.9671`. Dagua is much lower stress than the recorded reference.
- `cross_D_mean=0.0`, `cross_R_mean=0.0`, abs gap 0.
- `np_D_mean=1`, `np_R_mean=0.99339827`, within `np_margin=0.02`.
- Current-runner direct RMSD, seeds 100, 101, 102, 103, 104: median `3.94e-17`, max `0.000131`.

The recorded r75 reference stress for `deep_chain_20` is exactly the same for steps10, steps100, and steps200 (`0.0275418`), while direct current-runner positions at steps100/200 match Dagua nearly exactly and have expanded chains (`d_minmax`/`r_minmax` both around 724-788 in the probe). This is a stale/incorrect reference overlay or scoring-row artifact. Recommendation: floor-evidence this row immediately or rescore with regenerated per-seed references; do not change Dagua FMMM for this regression.

## 5. RNG Bit-Exact Feasibility

OGDF source used by the runner: `/home/jtaylor/tools/ogdf-src/src/ogdf/basic/basic.cpp`:

- `static std::mt19937 s_random;`
- `void setSeed(int val) { s_random.seed(val); }`
- `randomNumber(low, high)` constructs `std::uniform_int_distribution<>(low, high)` and returns `dist(s_random)`.

First 20 draws for seed 100 matched exactly between Dagua `_OgdfMt19937.randint` and a C++ standard-library probe:

```text
range 1..1000000000: 583476611 720647879 298896849 442431450 455822294 565198868 907071469 159563235 5066840 168267246 130533849 200217614 720211345 225601465 886752644 486125769 146787581 934308541 617501767 68376996
range 0..550: 299 369 153 227 233 290 465 81 2 86 66 102 369 115 455 249 75 479 316 35
```

Feasibility verdict: base per-seed bit-exactness is feasible and already achieved for the RNG primitive. Remaining exactness is not a simple seed-stream fix. For connected rows, Dagua is already effectively bit-/rounding-exact against the current runner. For disconnected structural rows, focus on component handling: component ordering, per-component RNG consumption, merge/packing offsets, and whether zero-origin component placement consumes random draws differently.

## Recommendations

| Cluster | Rows | Minimal path | Effort |
|---|---:|---|---|
| Connected OGDF near-match rows | 24 | Floor-evidence / mark aggregate-equivalent or metric-discrete. Do not patch geometry. For crossings-only small gaps, prefer crossing integer margin policy or population-equivalence tier. | S |
| `deep_chain_20` steps100/200 stale-reference rows | 2 | Rescore or override with direct current-runner evidence. Treat steps200 regression as accounting artifact. | S |
| Disconnected structural OGDF rows | 6 | Fix component merge/order/RNG-consumption parity. Start with `random_dag_50` because it covers steps10/100/200 and reproduces at RMSD ~0.09. Then verify `multi_component_80`, `kitchen_sink_platform_graph`, `random_dag_200`. | M-L |
| Graphviz-FDP FMMM fidelity rows | 3 | Route to Graphviz-FDP fidelity triage, not OGDF FMMM. | S/M, separate owner |
| RNG primitive | 0 open | No change needed. Keep `_OgdfMt19937`; it matches libstdc++ first draws. | Done |

## Concerns

- The task text says r75 has fresh regenerated OGDF reference positions in `benchmark_100seed_r75_fixes`; I could not find those files there. Direct current-runner probes were used for RMSD instead.
- `r75_final.jsonl` still contains at least one apparent stale-reference symptom (`deep_chain_20` reference stress identical across steps10/100/200). Verify the report loader/source-dir chain before using those rows as final truth.
- The 3 Graphviz-FDP rows satisfy the literal filter but are not part of OGDF FMMM parity. Keep them out of any OGDF-specific closure score.

## Knowledge

- FMMM residuals are mostly not residual geometry defects: 24/30 OGDF rows are coordinate near-matches.
- Structural OGDF FMMM residuals are disconnected-only in this filtered set.
- Base RNG parity is already proven against libstdc++ `std::mt19937`/`uniform_int_distribution`; future work should probe when component layout consumes the stream, not how integers are generated.
