You are independently auditing the EVALUATION/FIDELITY pipeline of dagua (a graph-layout engine) -- NOT
the layout algorithms. A parallel Anthropic/Opus auditor is doing the SAME axis blind to you; find what
it misses. READ-ONLY: do NOT edit code, do NOT run benchmarks that write to eval_output (a re-bench is
running), do NOT touch git. You MAY read code + per_combo data and compute metrics on EXISTING stored
layouts to trace ground truth.

CONTEXT: fidelity scores each (engine,graph) combo into rungs: 1 bit-exact, 2/3 stat-equivalent-by-LAYOUT
(positions), 3Q quality-identical (quality METRICS equivalent regardless of position), 4 divergent. 574
are rung-4, many asserted "FP floor" (different basin from rounding chaos).
HYPOTHESIS (project lead, sound): genuine FP-floor (same algo/params, different basin from rounding) MUST
be quality-indistinguishable from the reference IN AGGREGATE over 100 seeds -- rounding can't
systematically reduce quality; it averages out. So any quality-metric difference is EITHER a metric
ARTIFACT (mismeasuring) OR a REAL non-trivial algorithmic difference (more work). Treat "FP floor" as
UNPROVEN -- the project rule is floor needs FP-chaos EVIDENCE, never assertion. Be skeptical and curious.
KEY FILES: dagua/eval/distributional_fidelity.py, dagua/eval/equivalence_metrics.py, scripts/
definitive_fidelity_analysis.py, scripts/definitive_fidelity_report.py. Data: eval_output/
fidelity_definitive_r73/per_combo.json (battery_* / *_p_tost / *_margin / stress_D/R_mean / cross_D/R /
np_D/R / mean_W_R / final_rung as STRING). Stored layouts: eval_output/benchmark_100seed_*/positions[.h5]
-- NOTE pipeline loads NEWEST-mtime per combo (load_results_multi); a combo's current reimpl may be in a
later dir (e.g. gem_realfix, r73_fixes), NOT escalation_final -- resolve newest-dir-per-combo or you will
fabricate regressions.
GUARDRAIL: any reclassification must still pass the anti-laundering controls (0/40 chance+negative). We
are NOT laundering -- we are finding whether the metric mismeasures or there is a real cause. Rigor.
OUTPUT: write full findings to the named /tmp file; return <=400-word summary with file:line evidence,
artifact-vs-real verdict, traced numbers, the fix/experiment, confidence. Concrete and quantitative.
