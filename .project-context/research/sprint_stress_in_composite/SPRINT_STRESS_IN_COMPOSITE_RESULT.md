# Sprint STRESS IN COMPOSITE Result

Date: 2026-04-27
Branch: `codex/sprint-31a-gate-refinement`

## Decision

Ship the composite change. Keep `LayoutConfig.w_stress = 0.0`.

The new composite includes `sampled_stress` at 10% weight. The 93-graph
validation keeps dagua comfortably inside the shipping gate:

| gate | result | pass |
|---|---:|---|
| dagua new composite mean rank <= 2.5 | 1.274 | yes |
| dagua new best-or-tied >= 88% | 90.3% | yes |
| no engine gains 7+ mean ranks | max gain 0.172 | yes |

## New Formula

```python
composite = (
    22 * dag_consistency
  + 18 * (1 - normalized(edge_length_cv))
  + 13 * depth_spearman_rho
  +  8 * (1 - overlap_fraction)
  +  9 * straight_score
  +  9 * (1 - crossing_rate)
  + 10 * (1 - sampled_stress)
  +  5 * (angular_res_mean_deg / 180.0)
  +  6 * cluster_separation
)
```

Implementation detail: `dagua.metrics.composite()` uses the existing local
normalizations in code: edge straightness is `1 - deg/45`, crossing density is
`1 - crossing_rate * 10`, angular resolution is `deg/40`, and cluster
separation is `min(1, cluster_mean_sep_ratio / 5)` with a neutral no-cluster
contribution.

## Empirical Impact

Validation script: `/tmp/sprint_stress_in_composite_validate.py`

Outputs:

- `/tmp/sprint_stress_in_composite_validate.json`
- `/tmp/sprint_stress_in_composite_validate.csv`

Method: regenerate dagua layouts live on current HEAD for all 93 graphs with
`LayoutConfig(seed=42)`, load cached competitor positions with the same fallback
rules as `/tmp/rank_analysis.py`, compute metric rows once, then compare old
and new composite formulas on the same rows.

| engine | old mean rank | new mean rank | delta | old best-or-tied | new best-or-tied |
|---|---:|---:|---:|---:|---:|
| `dagua` | 1.263 | 1.274 | +0.011 | 90.3% | 90.3% |
| `graphviz_dot` | 2.629 | 2.704 | +0.075 | 7.5% | 6.5% |
| `dagre` | 3.785 | 3.796 | +0.011 | 4.3% | 3.2% |
| `igraph_sugiyama` | 4.366 | 4.301 | -0.065 | 4.3% | 5.4% |
| `elk_layered` | 4.462 | 4.473 | +0.011 | 5.4% | 7.5% |
| `sgd2` | 8.654 | 8.617 | -0.037 | 0.0% | 0.0% |
| `ogdf_fmmm` | 8.817 | 8.731 | -0.086 | 0.0% | 0.0% |
| `graphviz_neato` | 8.925 | 8.925 | +0.000 | 0.0% | 0.0% |
| `igraph_kamada_kawai` | 8.935 | 8.925 | -0.011 | 0.0% | 0.0% |
| `graphviz_sfdp` | 8.957 | 8.957 | +0.000 | 1.1% | 0.0% |
| `cytoscape_fcose` | 9.118 | 8.968 | -0.151 | 1.1% | 1.1% |
| `classic_sgd2_multi` | 9.925 | 9.989 | +0.065 | 0.0% | 0.0% |
| `nx_spring` | 10.602 | 10.774 | +0.172 | 0.0% | 0.0% |

The largest mean-rank gain is `nx_spring` worsening by +0.172 in the displayed
delta convention; the largest improvement is `cytoscape_fcose` at -0.151. This
is a measured tilt, not a metric overhaul.

## w_stress Default

Probe: `/tmp/sprint_w_stress_probe.py`

Output:

- `/tmp/w_stress_probe.csv`
- `/tmp/w_stress_probe_new_composite.log`

At `w_stress=0.05`, only 3 of 15 probe graphs improved new composite:
`scale_free_ba_120`, `er_500`, and `random_dag_200`. The acceptance threshold
was 8 of 15, so the default remains `0.0`.

Notable negative deltas at `w_stress=0.05`:

| graph | new composite delta | sampled_stress delta |
|---|---:|---:|
| `real_lesmis_77` | -0.31 | +0.0033 |
| `dependency_graph_100` | -0.42 | +0.0410 |
| `ba_500` | -0.08 | +0.0007 |

## Justification

Adding stress makes the composite more defensible, not more gameable. The
previous omission was historical, while stress is a canonical graph-drawing
aesthetic: Kruskal's multidimensional-scaling stress formalized distance
preservation; Kamada and Kawai used graph-theoretic distances as spring ideal
lengths; Brandes and Pich made stress approximation practical for larger
graphs; and later graph-drawing benchmark work such as Ahmed et al.'s
`(SGD)^2` and Hu and Shi's large-graph work treat stress as a primary or
co-equal aesthetic.

The 10% weight is intentionally mid-tier. Dagua targets directed graph
visualization, so DAG consistency, depth ordering, and edge straightness still
outweigh stress collectively. Stress now registers when layouts distort graph
distances, but it does not dominate the directed-readability objective.

## Notes

`sampled_stress()` normalizes positions to the unit square before measuring
Euclidean distances, so the stress score is scale-invariant and roughly
bounded in the expected 0-1 range for the benchmark layouts.

The old validation baseline in this run is close to but not bit-identical with
the published `HONEST_BENCHMARK.md` table because this script reran dagua live
and remeasured stochastic crossing samples in-process. The shipping conclusion
does not depend on the small baseline drift: the new formula leaves dagua at
rank 1.274 and 90.3% best-or-tied, above the sprint gates.
