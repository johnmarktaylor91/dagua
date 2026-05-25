<task>
R35 statistical robustness verdict check.

The fidelity report's verdicts (`strong_equivalent`, `weak_equivalent`,
`partial_match`) depend on TOST tests over ~100 seeds. If subsampling 10-20
seeds changes verdicts, the report is noisy. If verdicts hold under
subsampling, they're robust.

## Your job

1. Read `eval_output/benchmark_100seed_final/results.json` (the full 100-seed
   data; 858MB).
2. For each variant, simulate verdict under subsamples:
   - Sample 5 random subsets of 30 seeds (out of 100)
   - Compute TOST verdict per subset using the existing aggregation logic
   - Record verdict frequency (how often each verdict tier is reached)
3. Per variant, classify as:
   - **robust**: same verdict for >=4 of 5 subsets
   - **borderline**: verdict varies across subsets
   - **noisy**: verdict varies wildly

## Implementation

Write `scripts/r35_robustness_check.py` that:
- Loads results.json + positions.h5
- For each variant, subsamples seeds and re-runs the verdict computation
  (use `scripts/fidelity_analysis.py` machinery if importable, OR
  reimplement TOST verdict logic locally for the subsample)
- Outputs CSV with per-variant robustness classification

Run it, write `eval_output/algo_fidelity/round_35/robustness/SUMMARY.md` with
verdict-robustness counts per tier.

## Scope
- Use commit-safe wrapper.
- DO NOT TOUCH: existing fidelity_report_100seed_final/* outputs
- READ ONLY: results.json + positions.h5

## Output
- `scripts/r35_robustness_check.py` (committable script)
- `eval_output/algo_fidelity/round_35/robustness/SUMMARY.md`
- `eval_output/algo_fidelity/round_35/robustness/robustness_per_variant.csv`
</task>

<completeness_contract>
At least the per-variant CSV + summary count.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
