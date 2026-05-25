<task>
R33 Quality metric verdict gates.

Current fidelity report (`eval_output/fidelity_report_100seed_final/report.md`) bases verdicts solely on Procrustes RMSD + TOST equivalence test. Quality metrics (aspect_ratio, dag_consistency, edge_length_cv, edge_straightness_mean_deg, depth_spearman_rho, overlap_count, sampled_stress, crossing_rate) are computed and surfaced in sidecar CSVs but DON'T contribute to verdict tier.

## Your job

1. Read `scripts/fidelity_analysis.py` and `scripts/generate_fidelity_report.py` to understand:
   - How current verdicts are computed (TOST tiers from Procrustes)
   - How quality metrics are collected
2. Design + implement a "quality-tier verdict gate" that adds quality criteria to the verdict tier:
   - **strong_equivalent**: existing TOST 0.5x AND no quality metric regresses by >X% vs reference
   - **weak_equivalent**: existing TOST 1x AND quality metrics within Y% of reference
   - **partial_match**: existing fallback
3. Update `generate_fidelity_report.py` to surface a per-variant quality-metric delta vs reference.

## Read

- `scripts/fidelity_analysis.py`
- `scripts/generate_fidelity_report.py`
- `eval_output/fidelity_report_100seed_final/data/per_graph_detail.csv` (sample sidecar shape)

## Implementation

This is a SCRIPT update, not a layout-engine change. Commits via commit-safe wrapper:
- `feat(eval): round 33 fidelity report -- quality metric verdict gate`

## Verify

Generate a fresh report from existing data (no benchmark re-run needed):
```bash
python scripts/fidelity_analysis.py --input eval_output/benchmark_100seed_final --output eval_output/algo_fidelity/round_33/quality_gates/data
python scripts/generate_fidelity_report.py --input eval_output/algo_fidelity/round_33/quality_gates/data --output eval_output/algo_fidelity/round_33/quality_gates/report.md
```

Compare new verdicts vs original fidelity_report_100seed_final/report.md.

## Output
`eval_output/algo_fidelity/round_33/quality_gates/SUMMARY.md` with verdict comparison table.
</task>

<completeness_contract>
Update report generator + produce sample new report. Document verdict-shift count.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
