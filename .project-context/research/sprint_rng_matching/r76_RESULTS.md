# r76 -- Final Fidelity Sprint Results (2026-07-04)

## Headline
Full-universe authoritative pass: 3,955 combos, scored once with the complete r76 freshest-wins chain into `eval_output/fidelity_definitive/per_combo_r76.jsonl`.

Corrected disposition table, full universe:

| Disposition | r75 full | r76 | Delta |
|---|---:|---:|---:|
| `AGGREGATE_INSUFFICIENT` | 24 | 24 | +0 |
| `AGGREGATE_STALE_REFERENCE` | 27 | 27 | +0 |
| `DISTRIBUTIONAL_EQUIVALENT` | 1003 | 45 | -958 |
| `DIVERGENT_NAMED_CAUSE` | 363 | 335 | -28 |
| `FIDELITY_IDENTICAL` | 1739 | 2679 | +940 |
| `INSUFFICIENT_DATA` | 210 | 183 | -27 |
| `MODE_B_BIT_EXACT` | 50 | 80 | +30 |
| `MODE_B_CLOSE` | 99 | 94 | -5 |
| `MODE_B_IDENTICAL_DISTANCE` | 13 | 39 | +26 |
| `NO_CANONICAL_REFERENCE` | 405 | 405 | +0 |
| `SUPERIOR_DISTINCT` | 22 | 44 | +22 |

Stop criteria:
- Rows with exactly one disposition: 3,955 / 3,955.
- Bare divergent rows without a named cause: 0.
- Gate 5 laundering: 0 / 40.
- Scorer self-check: deterministic verdicts for 12 combos on the r76 chain.

## What r76 discovered
1. UMAP tiny-graph oracle bug: `parallel_multiedge_bundle` references used a seeded `randn` fallback at N=3; fixed so real UMAP runs at N=3.
2. SFDP reference param-noop attrs: Graphviz 7.0.5 ignores theta/maxiter/steps and p_neg2/repulsiveforce in the affected SFDP variants; these rows route to no-canonical.
3. Seed-era mismatches: large parts of the historical chain still resolve to seed-42-era reference ranges; all rows carry seed-era and low-power flags in the ledger.
4. Max-nodes silent exclusions: several topups had no current-code rows for builder-node-large graphs until the D4 runner guard made exclusions loud.
5. Scorer demotion gate: `quality_identical_raw` was gated by canonical-reference status; r76 ledger corrects fidelity-identical to raw OR exploratory.

## What r76 shipped
Merged fixes on develop include:
- `fe239cc` GEM update-budget fix.
- `00fbe41` Sugiyama x-network-simplex performance.
- `7a54b9d` / `aff2a0e` SFDP disconnected and connected parity fixes.
- `630bc2f` UMAP float32 schedule and local numba-kernel parity.
- `9d3cfa4` SFDP point-unit disconnected pack fix.
- `7a33573` Sugiyama mincross/crash stack.
- `08a6003` igraph Sugiyama ordinal conflict quirk.
- `ce4562d` D4 ledger/oracle invariant infrastructure.

## Named-Cause Registry
| Cause | Rows | Evidence | Quality note |
|---|---:|---|---|
| igraph Sugiyama GLPK degenerate-LP basis selection or residual BK/dummy detail | 222 | r76_IMPL_igraph_NOTES | Mode-B far igraph rows; vendoring GLPK excluded. |
| Graphviz neato exact CG/drand48/component packing residual | 55 | r75_PROBE_tails_RESULTS | Mode-B far neato rows; r75 tails identified neato CG/RNG/packing residual. |
| graphviz sugiyama stages B-D plus aux minlen half-width residual | 20 | r76_IMPL_mincross_NOTES A4b/A4c/A4d | Mode-B far graphviz rows; deterministic quality parity varies by label/cluster graph. |
| sfdp spline-box polyomino occupancy in packSubgraphs | 11 | r76_IMPL_sfdp_disc_NOTES C4d | 12 full-chain rows; C4d official 10-row cluster says encoder/label/random_dag_50 better-or-parity, kitchen_sink and multi_component about 5% worse; random_dag_200 remains full-chain divergent. |
| mds eigenspace/disconnected basis residual outside formal dossier set | 9 | r76_FLOOR_DOSSIERS by analogy; explicitly evidence-thin | Evidence-thin rows outside the formal MDS floor table are flagged. |
| mds proven member of reference equivalence class | 6 | r76_FLOOR_DOSSIERS | Equivalent Gram eigenspace membership; metric direction varies. |
| Graphviz FDP/neato packing residual | 3 | r75/r76 fdp and packer notes | FDP rows retain distinct packing behavior; random_dag_200 reference source is flagged. |
| OGDF stress disconnected optimization-path residual | 3 | r75_PROBE_tails_RESULTS + r76_PROBE_fairness | Quality mixed by metric; r75 tails/fairness evidence. |
| igraph DrL float32/density-grid stochastic residual | 3 | r75_mds_tails findings; coarsen evidence-thin flag retained | Two documented floor rows plus one evidence-thin coarsen row flagged. |
| OGDF FMMM MAAR/disconnected packing tie-break residual | 2 | r76_IMPL_fmmm_disc_NOTES + r76 state MAAR final | MAAR rows are equal-or-better on random_dag_50; full-chain random_dag_200 remains named residual. |
| umap spectral eigenspace floating-point chaos floor | 1 | r76_FLOOR_DOSSIERS | Quality parity shown in r76_FLOOR_DOSSIERS. |

## Gate Verdicts
- `gate_1_positive_mode_a`: passed=True; scored=39
- `gate_2_positive_mode_b`: passed=True
- `gate_3_negative`: passed=False; scored=20
- `gate_4_chance`: passed=True
- `gate_5_quality_identical_laundering`: passed=True; three_q_count=0/40
- `gate_6_reference_self_split_positive`: passed=True; scored=1

Gate 3 remains the pre-existing negative-control calibration failure; this matches the r75 report pipeline note and is not a gate-5 laundering failure.

## D4 Integrity Validator
The validator was run over the same 26-directory r76 scoring chain and exited nonzero only for the adjudicated historical families:

| Family | Failures | Adjudication |
|---|---:|---|
| `ogdf_gem` | 103 | stale-runner archaeology; fresh-scored rows resolve to `r76_refs`, `r76_refs2`, or `r76_refs3`; 27 large rows still resolving to `seeded_refs` are `STALE_REFERENCE` aggregate rows |
| `ogdf_fmmm` | 94 | stale-runner archaeology; same routing rule as GEM |
| `igraph_mds` | 21 | clamp-equivalent/no-material-param reference, backed by the MDS floor dossier |
| `ogdf_stress` | 15 | r75 runner-binary parity plus convergence/no-material-output evidence; residual rows remain named, not no-canonical |

Whitelisted param-equivalent hits were `graphviz_sfdp=99` and `umap_graph=1`. No new validator family appeared.

## Adjudications
- OGDF GEM/FMMM: 584 OGDF combos resolve to fresh r76 references; 27 large rows still resolve to `seeded_refs` and are flagged `STALE_REFERENCE` plus routed to aggregate/stale-reference.
- igraph MDS: default and fidelity legitimately share one reference because there are no material params; connected residuals use the equivalence-class floor dossier.
- OGDF stress: r75 proved stress runner binary parity; r76 treats param identity as convergence/no-material-output where observed and keeps residual maxent rows named rather than no-canonical.
- SUPERIOR_DISTINCT: r75 reference-bug rows are removed by rescoring; SFDP param-noop variants route to no-canonical; fair rows keep caveat labels.

## Aggregate, Low-Power, And Era Limits
- Aggregate/stale-reference rows: 27.
- Aggregate insufficient 2000/5000 rows: 24.
- Low-power `n < 100` rows: 645.
- Seed-42-era rows: 2052.
- Evidence-thin rows retained explicitly: 10.

The population/aggregate tier is metadata-only: rows without enough paired seeds are not per-seed pass/fail claims. The report records the family-level BH-corrected scoring method inherited from the r75/r76 definitive report pipeline; no new benchmarks were launched.

## Key Artifacts
- `eval_output/fidelity_definitive/per_combo_r76.jsonl`
- `eval_output/fidelity_definitive_r76/DEFINITIVE_FIDELITY_REPORT.md`
- `eval_output/fidelity_definitive_r76/FOUR_TIER_CATEGORIZATION.md`
- `eval_output/fidelity_definitive_r76/OFFICIAL_R76_LEDGER.md`
- `eval_output/fidelity_definitive_r76/OFFICIAL_R76_LEDGER.json`
- `.project-context/research/sprint_rng_matching/r75_findings/r76_FLOOR_DOSSIERS.md`
- `.project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_fairness.md`
- `.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_*_NOTES.md`

## Honest Limitations
- The official report renderer was run with `--no-strict` to reuse the existing deferred OC-simulation cache; the controls were run separately and gate 5 is green.
- Gate 3 remains false as a known calibration issue; gate 5 is the anti-laundering gate and is clean.
- Evidence-thin rows are not hidden: see `OFFICIAL_R76_LEDGER.md` for the explicit list.
- Stale-reference OGDF large rows are not per-seed verdicts.

## Knowledge
- The final overlay resolved 8,804 combos; 5,405 would have era-mixed under old union semantics.
- Fresh OGDF refs are `r76_refs`, `r76_refs2`, and `r76_refs3`; remaining stale OGDF rows are large aggregate-only rows.
- Mode-B deterministic row quality must be read from `d_R`, not the Mode-A rung vocabulary.
