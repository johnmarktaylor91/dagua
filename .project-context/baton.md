NEW SESSION: Read this file first, then CLAUDE.md and AGENTS.md.

## Mission

AUTONOMOUS MODE. Complete the fidelity hardening sprint. A re-benchmark is running.
When it finishes, run fidelity analysis, verify results, generate report, commit.
User directive: "for every single graph/algo, either perfect match or good reason
we didnt run it."

Read .project-context/autonomous_gate.json for exit criteria.

## Context

The fidelity analysis pipeline compares dagua's reimplemented layout algorithms against
their original reference implementations. 510K benchmark evals, 97 algorithm families,
105 test graphs, 9104 variant/graph pairs.

### Verdict Breakdown BEFORE This Session

**Family level (97):** 56 strong, 35 partial, 6 divergent

### Verdict Breakdown AFTER Verdict Logic Fixes (recompute only, no re-bench)

**Family level (97):** 88 strong, 2 weak, 7 partial, 0 divergent

### What's Running Now

Re-benchmark dispatched via `./scripts/dispatch.sh rebench-all`:
- 94,500 total runs, ~32K cached, ~62K to run
- 8 SGD2 multi reimpl variants (lr 1.0->0.01, grad_clamp 20->5)
- 5 t-SNE reimpl variants (random init, LR floor 50, no 4x late)
- 6 NeuLay original variants (new reference wrapper installed)
- 1 FA2 linlog original (installed fa2 package with linLogMode)
- Monitor: `tail -f .project-context/tasks/rebench-all.log`
- Status: `cat .project-context/tasks/rebench-all.status`

### After Re-Benchmark Completes

1. Run full fidelity analysis (~6 hours):
   ```bash
   PYTHONUNBUFFERED=1 python -u scripts/fidelity_analysis.py \
     --input eval_output/variant_bench_full \
     --output eval_output/fidelity_report/data \
     --skip-metrics
   ```
2. Run recompute for metrics + verdicts (~12 min):
   ```bash
   python scripts/fidelity_recompute_verdicts.py --data eval_output/fidelity_report/data
   ```
3. Audit results: every family must be strong_equivalent or have documented reason
4. Generate PDF: `python scripts/generate_fidelity_report.py ...`
5. Commit everything

### Root Cause of Each Category

**Insufficient data (4,433 per-graph, drives all 35 partial_match families):**
The 35 partial_match families ALL have paired=0 and rmsd=NaN at family level.
Every per-graph row for these families is `insufficient_data`. The reason:

- These variants have orig_seeds=1, reimpl_seeds=30 (or 0/30 for NeuLay, fa2_linlog)
- The original engine (igraph/networkx/etc.) was benchmarked with seed=None (deterministic)
  producing 1 result, while our reimpl ran 30 seeds
- The fidelity script requires MIN_STOCHASTIC_SEEDS=10 on BOTH sides for stochastic variants
- With only 1 orig seed, every per-graph pair gets `insufficient_data`

Affected families: FMMM (3), GEM (3), SFDP (5), Sugiyama (5), stress_maj (3),
maxent_stress (5), pivot_MDS (4), NeuLay (6), fa2_linlog (1)

**Divergent (6 families, 50 per-graph):**
- tsnet_default, tsnet_perp5, tsnet_perp50, tsnet_steps200, tsnet_steps2000
- sgd2_multi_with_crossing
- These have real paired data (73-85 graphs paired) with RMSD 0.21-0.28
- Within-vs-between Procrustes ratio ~1.19-1.25 (measurably higher than within)

### Key Files

| File | What |
|------|------|
| scripts/fidelity_analysis.py | Main analysis (~2500 lines) |
| scripts/fidelity_recompute_verdicts.py | Fast verdict recomputer (~230 lines) |
| scripts/fidelity_add_metrics.py | Second-pass quality metrics |
| scripts/generate_fidelity_report.py | LaTeX report generator |
| eval_output/fidelity_report/data/ | CSVs: algorithm_summary, per_graph_detail, per_seed_detail, pairwise_similarity |
| eval_output/fidelity_report/report.pdf | Current report |
| eval_output/variant_bench_full/results.json | 510K benchmark records |
| eval_output/variant_bench_full/positions.h5 | 406K position tensors (1.4GB) |
| .project-context/knowledge/fidelity_verdict_pitfalls.md | Lessons learned this session |
| dagua/eval/engines/ | All engine implementations (orig + reimpl) |
| dagua/eval/variants.py | VARIANT_REGISTRY with variant definitions |

### Verdict Logic Summary

**Stochastic path** (in finalize_group_row):
- Needs MIN_STOCHASTIC_SEEDS=10 on both sides
- Runs TOST equivalence test on quality metrics at 4 margin levels
- strong_equivalent: TOST passes at 1x, no MW anomaly
- weak_equivalent: TOST passes at 1.5x, explainable anomalies
- partial_match: TOST passes at 2x
- divergent: TOST fails at 2x

**Deterministic path:**
- identical: max_displacement < 1e-4
- strong_equivalent: no unexplainable anomalies
- partial_match: max_displacement <= 1.0
- divergent: max_displacement > 1.0

**NaN handling:** When orig==reimpl distributions (zero pooled SD), NaN p-values
are treated as pass (identical distributions trivially satisfy equivalence).

**Explainable anomalies:** mirror_match, structural_note, scale_ratio_out_of_range,
runtime_ratio_outlier, runtime_ratio_warning -- these don't block strong_equivalent.

### Fixes Already Applied This Session

1. Removed ThreadPoolExecutor (GIL hang)
2. Removed HDF5 pre-load (unnecessary in serial mode)
3. Added progress logging
4. mirror_match treated as explainable
5. NaN TOST from identical distributions = pass
6. mirror_match doesn't block strong_equivalent for stochastic
7. scale_ratio and runtime_ratio are explainable

### Recompute Pipeline (FAST -- ~11 min)

To recompute verdicts after code changes:
```bash
find scripts -name '__pycache__' -type d -exec rm -rf {} +
PYTHONUNBUFFERED=1 python -u scripts/fidelity_recompute_verdicts.py \
  --data eval_output/fidelity_report/data
```

To recompile PDF:
```bash
python scripts/generate_fidelity_report.py \
  --data eval_output/fidelity_report/data \
  --output eval_output/fidelity_report
```

Full re-analysis (SLOW -- 6 hours, only if CSV data must change):
```bash
PYTHONUNBUFFERED=1 python -u scripts/fidelity_analysis.py \
  --input eval_output/variant_bench_full \
  --output eval_output/fidelity_report/data \
  --skip-metrics
```

## Task Plan

### Phase 1: Launch 6 Investigation Agents

Launch these IN PARALLEL (3 Claude subagents + 3 Codex via dispatch.sh):

#### Agent 1 (Claude): Investigate Partial Match / Insufficient Data
Prompt: Exhaustively investigate why 35 algorithm families get partial_match verdict
(all driven by insufficient_data at per-graph level). The root cause is that original
engines produced only 1 seed (deterministic) while reimpl produced 30 seeds, and the
analysis requires MIN_STOCHASTIC_SEEDS=10 on both sides.

Investigate:
- Are these originals truly deterministic? Check each engine in dagua/eval/engines/
- Should the stochastic flag be changed for some variants? Check dagua/eval/variants.py
- Can the analysis handle asymmetric seed counts (1 orig vs 30 reimpl)?
- What would a correct comparison look like for 1-vs-many?
- Can we re-benchmark the originals with multiple seeds?
- Which originals CAN'T produce multiple seeds (truly deterministic)?
- For NeuLay and fa2_linlog: why orig_seeds=0? Missing engine? Bug in benchmarking?

Deliverable: Detailed report with specific recommendations per family, code snippets
showing where changes are needed, and a concrete action plan.

#### Agent 2 (Codex): Code-level audit of insufficient data path
Spec: Read scripts/fidelity_analysis.py, dagua/eval/variants.py, and dagua/eval/engines/.
For each of the 35 partial_match families, trace the EXACT code path from benchmark
result to verdict. Identify:
- Where the insufficient_data decision is made
- Whether the stochastic flag is correct per variant
- Whether MIN_STOCHASTIC_SEEDS=10 is appropriate
- Whether a 1-vs-N comparison mode should exist
- Specific code changes to fix each family
Produce a patch or detailed diff for each proposed change.

#### Agent 3 (Claude): Investigate Divergent Families
Prompt: Deep-dive into the 6 divergent families (5 t-SNE + sgd2_multi_with_crossing).
These have real paired data with RMSD 0.21-0.28 and within/between ratio 1.19-1.25.

Investigate:
- Read the t-SNE reimplementation code in dagua/eval/engines/ or dagua/layout/
- Read the original t-SNE engine code
- Compare algorithm details: perplexity handling, learning rate, early exaggeration,
  momentum, Barnes-Hut approximation, initialization
- Check pairwise_similarity.csv for patterns: which graphs diverge most? Is it
  graph-size dependent? Topology dependent?
- For sgd2_multi_with_crossing: what's the crossing loss doing differently?
- Is 0.21-0.28 RMSD actually concerning or expected for t-SNE-family algorithms?
- What specific code changes would improve fidelity?

Deliverable: Root cause analysis per family, comparison tables, specific code fixes.

#### Agent 4 (Codex): Line-by-line t-SNE/SGD2 reimpl audit
Spec: Read the t-SNE reimplementation and the reference code side by side. Produce
a line-by-line comparison highlighting every difference. Check: initialization,
gradient computation, learning rate schedule, momentum, early exaggeration,
perplexity computation, Barnes-Hut theta, convergence criteria. Same for
sgd2_multi_with_crossing vs the other sgd2 variants that pass.
Produce specific patches for any fidelity issues found.

#### Agent 5 (Claude): Investigate Verdict Logic Robustness
Prompt: Review the entire verdict pipeline for remaining edge cases and correctness.

Investigate:
- Is the family-level aggregation too conservative? All-or-nothing vs percentile?
- The 4,433 insufficient_data rows: is the 10-seed threshold right?
- Are there per-graph "divergent" rows in otherwise strong_equivalent families?
  What's causing those 50 divergent per-graph rows?
- Check the pairwise_similarity.csv within-vs-between ratios for ALL families
- Is TOST the right test? Should we use the Procrustes ratio instead?
- Review BH correction: is it too aggressive? Too lenient?
- Check for any remaining NaN/inf edge cases in the verdict logic

Deliverable: Verdict logic audit report with specific recommendations.

#### Agent 6 (Codex): Verdict logic hardening
Spec: Read finalize_group_row(), family_summary_rows(), apply_bh_correction(),
and all verdict-related code. Find and fix:
- Edge cases where NaN/inf values produce wrong verdicts
- Family aggregation logic improvements
- Any remaining cases where identical data produces non-equivalent verdicts
- Type coercion issues (CSV strings vs Python types)
- The `bool("False") == True` problem for reflected field from CSV
Produce specific patches.

### Phase 2: Synthesis & Adversarial Review

After all 6 agents return:
1. Read ALL findings carefully
2. Synthesize into a prioritized action plan
3. Classify each action: code fix | config change | re-benchmark needed | won't fix
4. Dispatch the plan to adversarial Codex for critique
5. Iterate until Codex approves
6. Implement all fixes

### Phase 3: Targeted Rerun

Based on what changed:
- If only verdict logic changed: run recompute_verdicts.py (~11 min)
- If stochastic flags changed: need full re-analysis (~6 hours)
- If reimpl code changed: need re-benchmark for affected variants (hours)
- MINIMIZE rerun scope -- be surgical

### Phase 4: Final Report

1. Recompile PDF
2. Produce final verdict breakdown
3. Update .project-context/knowledge/fidelity_verdict_pitfalls.md
4. Update memory
5. Commit everything

## Git State

- Branch: feat/bench-and-aesthetics
- Uncommitted changes: fidelity_analysis.py (ThreadPool removal, pre-load removal,
  progress logging, verdict fixes), fidelity_recompute_verdicts.py (new file)
- Should commit before starting new work

## Critical Rules

- DO NOT rerun the full 6-hour analysis unless absolutely necessary
- The recompute script is the fast path for verdict changes (~11 min)
- Always clear __pycache__ before running scripts after edits
- Check `free -h` before heavy tasks
- Progress logging on ANY loop > 10 seconds
- Read .project-context/knowledge/fidelity_verdict_pitfalls.md before touching verdicts
