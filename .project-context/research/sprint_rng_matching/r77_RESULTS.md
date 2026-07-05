# r77 -- Grand Finale Fidelity Results (2026-07-05)

## Headline
Full-universe authoritative pass: 3,955 combos, scored once with the complete r76 chain plus r77_mds2, r77_sfdp_pack2, r77_sugiyama_a5b, r77_sugiyama_final, r77_sugiyama_wired, r77_igraph_bk, r77_randomdag, and r77_era_refs into `eval_output/fidelity_definitive/per_combo_r77.jsonl`. Requested `r77_maar` was absent and skipped.

| Disposition | r76 | r77 | Delta |
| --- | ---: | ---: | ---: |
| `AGGREGATE_INSUFFICIENT` | 24 | 22 | -2 |
| `AGGREGATE_STALE_REFERENCE` | 27 | 3 | -24 |
| `DISTRIBUTIONAL_EQUIVALENT` | 45 | 36 | -9 |
| `DIVERGENT_NAMED_CAUSE` | 335 | 249 | -86 |
| `FIDELITY_IDENTICAL` | 2679 | 2678 | -1 |
| `INSUFFICIENT_DATA` | 183 | 221 | +38 |
| `MODE_B_BIT_EXACT` | 80 | 186 | +106 |
| `MODE_B_CLOSE` | 94 | 74 | -20 |
| `MODE_B_IDENTICAL_DISTANCE` | 39 | 40 | +1 |
| `NO_CANONICAL_REFERENCE` | 405 | 405 | +0 |
| `SUPERIOR_DISTINCT` | 44 | 41 | -3 |

Stop criteria:
- Rows with exactly one disposition: 3,955 / 3,955.
- Bare divergent rows without a named cause: 0.
- Generic fallback named rows: 0.
- Gate 5 laundering: 0 / 40.
- Scorer self-check: deterministic verdicts for 12 combos on the r77 chain.
- Random-dag shape mismatch hazard: no exclusions; all 69 random_dag rows resolve to r77_randomdag/new canonical seeds.

## What r77 discovered
1. Oracle bug #5: `_random_dag` used hash-order-dependent edge construction, so random_dag_50/200 comparisons across processes were permutation-corrupted. Same-realization r77_randomdag closes MAAR, random_dag SFDP, random_dag MDS, random_dag maxent/stress, neato, and UMAP rows that r76 had misattributed.
2. SFDP residual cause corrected: measured spline on/off showed splines were not the residual; Graphviz label-box occupancy in packSubgraphs is the remaining named cause.
3. UMAP random_dag eigenspace-floor rows are retracted to identical. The connected-MDS equivalence-class/eigensign floor remains and the M5 note explains why O(2) Procrustes scoring makes reflections equivalent.
4. Sugiyama far-tail causes are sharper: graphviz tails are recursive cluster rank-collapse/structural stragglers; igraph tails are GLPK-solved then BK/dummy residuals after wired r77 data.
5. Delegation/inference lessons: do not use runtime delegation for parity fixes; compare same-realization graph data before naming causes; measure the first divergent quantity before accepting a theory; read scoring registration before treating reflections or eigensigns as fidelity failures.

## What r77 shipped
- GLPK parity: `445bc61` merge, `b47d272` implementation, `a61bd2e` docs.
- BK/x-stage and Sugiyama wiring: `2f54853`, `118a50e`, `8960d61`, `8ed1003`.
- Sugiyama half-width, labels, and clusters: `3d1537f`, `4279b3e`, `092feb4`, `ecd85be`, `437e5d0`.
- SFDP label-box correction: `66be0f`, `59a462c`.
- MDS disconnected/eigensign chain: `bc72627`, `b029ab1`, `2ab8efd`, `a45087c`, plus docs `7c067bd` and `c56d8fd`.
- Graph determinism/oracle bug #5 fix: `4178b93`, `73bbb52`.
- Crash/cache/wiring fixes recorded in the r77 state, including recursion-limit/crash handling and `_quick_classic` cache-key wiring.

## Honest Residual Registry
| Cause | Rows |
| --- | ---: |
| igraph Sugiyama GLPK-solved then BK/dummy residual from wired r77 data | 145 |
| Graphviz neato exact CG/drand48/component packing residual | 54 |
| Graphviz Sugiyama recursive cluster rank-collapse far-tail residual (r77 A9 dossier) | 20 |
| MDS eigenspace/eigensign equivalence-class floor confirmed by r77 full-power rescore | 12 |
| Graphviz SFDP label-box disconnected packing residual | 8 |
| Graphviz FDP/neato packing residual | 5 |
| igraph DrL float32/density-grid stochastic residual | 3 |
| era-rescore low-power SGD2 crossing residual; EVIDENCE-THIN | 2 |

Full-power era-rescore honesty corrections:
- `complete_bipartite_8x12::classic_classical_mds_igraph_fidelity` -- MDS eigenspace/eigensign equivalence-class floor confirmed by r77 full-power rescore
- `disconnected_label_cycle_collage::classic_classical_mds_default` -- MDS eigenspace/eigensign equivalence-class floor confirmed by r77 full-power rescore
- `disconnected_label_cycle_collage::classic_classical_mds_igraph_fidelity` -- MDS eigenspace/eigensign equivalence-class floor confirmed by r77 full-power rescore

Low-power era stragglers:
- `real_football_115::classic_sgd2_multi_with_crossing` -- era-rescore low-power SGD2 crossing residual; EVIDENCE-THIN
- `wide_1_100_1::classic_sgd2_multi_with_crossing` -- era-rescore low-power SGD2 crossing residual; EVIDENCE-THIN

Retired/retracted r76 causes:
- MAAR packing tie residual: retired; same-realization random_dag rows are identical/equivalent.
- UMAP random_dag eigenspace floor: retracted; the observed divergence was oracle bug #5.
- SFDP spline-box cause: corrected to label-box occupancy.

## Gate Verdicts
- `gate_1_positive_mode_a`: passed=True; details={'pass_count': 39, 'pass_percent': 100.0, 'passed': True, 'scored': 39}
- `gate_2_positive_mode_b`: passed=True; details={'informative': 39, 'pass_count': 39, 'pass_percent': 100.0, 'passed': True}
- `gate_3_negative`: passed=False; details={'non_primary_percent': 90.0, 'passed': False, 'raw_tracking_count': 0, 'raw_tracking_limit': 3, 'scored': 20}
- `gate_4_chance`: passed=True; details={'ks_p': 0.8177374448453805, 'n': 20, 'passed': True, 'recovery_count': 21}
- `gate_5_quality_identical_laundering`: passed=True; details={'leaked_combo_ids': [], 'limit_percent': 0.0, 'missing_battery_count': 0, 'passed': True, 'scored': 40, 'three_q_count': 0, 'three_q_percent': 0.0}
- `gate_6_reference_self_split_positive`: passed=True; details={'passed': True, 'quality_identical_count': 1, 'scored': 1, 'three_q_count': 0}

Gate 3 remains the pre-existing negative-control calibration failure also present in r76. Gate 5, the anti-laundering gate requested for this finale, is clean at 0/40.

## D4 Integrity Validator
The validator was run over the full r77 chain and exited nonzero only for adjudicated historical classes. Counts: `ogdf_gem=103`, `ogdf_fmmm=94`, `igraph_mds=27`, `ogdf_stress=9`, `graphviz_neato=2`. The new graphviz_neato hits are random_dag default-vs-fidelity clamp-equivalent reference rows after the canonical realization fix.

## Aggregate, Low-Power, And Era Limits
- Aggregate/stale-reference rows: 3.
- Aggregate insufficient 2000/5000 rows: 22.
- Low-power `n < 100` rows: 260.
- Seed-42-era rows: 1951.
- Evidence-thin rows retained explicitly: 2.

## Key Artifacts
- `eval_output/fidelity_definitive/per_combo_r77.jsonl`
- `eval_output/fidelity_definitive_r77/DEFINITIVE_FIDELITY_REPORT.md`
- `eval_output/fidelity_definitive_r77/FOUR_TIER_CATEGORIZATION.md`
- `eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md`
- `eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.json`
- `.project-context/research/sprint_rng_matching/r77_RESULTS.md`
- `.project-context/research/sprint_rng_matching/r75_findings/r76_THIN_ROW_DOSSIERS.md`

## Honest Limitations
- The official report renderer was run with `--no-strict`; it reported the known mixed reference-mode assertion for `classic_neato`.
- Gate 3 remains false as a known negative-control calibration issue; gate 5 is clean.
- Requested `benchmark_100seed_r77_maar` was absent, so no r77_maar directory was included. The MAAR verdict is instead superseded by r77_randomdag same-realization scoring.
- The two SGD2 era stragglers are low-power n=42 rows and are explicitly evidence-thin.
- Three FMMM large/stale rows remain aggregate-only stale-reference rows, not per-seed verdicts.

## Knowledge
- The final overlay resolved 9,211 combos; 5,621 would have era-mixed under old union semantics.
- r77_randomdag resolves all 69 random_dag combos to the new canonical realization with no old-seed sides.
- Mode-B deterministic row quality must be read from `d_R`, not the Mode-A rung vocabulary.
- Fresh r77 Sugiyama data should use `r77_sugiyama_wired`; earlier r77 Sugiyama topups are superseded for ledger tiering.
