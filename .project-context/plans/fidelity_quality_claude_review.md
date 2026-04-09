# Claude Explore Adversarial Review

## Verdict: SHIP_WITH_FIXES

## CRITICAL Issues

**C1: Plan cites non-existent constant `MAX_PROCRUSTES_SEEDS_PER_SIDE = 10` (line 223)**
- Plan section: Group D (D1)
- Actual: line 58 has `PAIRWISE_SAMPLE_SIZE = 10`. No `MAX_PROCRUSTES_SEEDS_PER_SIDE` exists.
- Increasing it affects pairwise comparison sampling globally, not just procrustes.
- Cite: fidelity_analysis.py:58, 1769, 1774

**C2: Plan misframes Procrustes TOST (lines 44, 107)**
- Procrustes TOST is missing, but TOST infrastructure (`tost_pvalue`) and metric TOST columns exist (lines 1546-1555).
- Reword A2: "Add Procrustes-specific TOST equivalence test (metric TOST already implemented)."

**C3: BH correction location is line 1986, NOT ~2617 (lines 135, 170)**
- Actual BH correction loop at `apply_deferred_bhcorrection()` line 1970-1990.
- Line 2617 is CSV writing, not statistics.
- Plan repeatedly cites wrong line. Codex will waste time searching.
- Affects A4, B2, B3.

**C4: Plan doesn't address one-sided test backwards logic explicitly (line 2164-2172)**
- Current verdict logic uses `wb_pval >= 0.05` to mark "strong_equivalent"
- This is ABSENCE of evidence as EVIDENCE OF ABSENCE (backwards)
- A5 must explicitly DELETE these lines, not augment
- If A5 lands without deletion, old heuristic still contaminates verdicts

## HIGH Issues

**H1: C1 deterministic comparator should test 3 levels**
- (1) torch.equal() on aligned positions
- (2) torch.allclose(atol=1e-6) as "strong_equivalent"
- (3) max_displacement < 1e-4 as secondary
- Current code only uses (3). See compare_reimpl_vs_original.py:277 for reference.

**H2: Quality/runtime metric recomputation lacks memory budget**
- Plan says "must recompute from positions" but doesn't address:
  - Will 67k position tensors fit in RAM?
  - Streaming HDF5 reads per (graph, engine) fallback?
  - sampled_stress OOM on large graphs?
- QR-1 needs explicit memory budget + xlarge graph fallback

**H3: Graph-relative ranking has partial-data instability risk**
- Plan adopts Codex normalization but doesn't surface coverage filter
- Risk: only small graphs succeed → ranking biased toward small-graph engines
- Codex design line 434-436 has the filter (graphs_covered >= 3, coverage_ratio >= 0.5) but plan buries it in open question 8
- Should be EXPLICIT in QR-1 spec

## MEDIUM Issues

**M1: B3 metric expansion creates FID-2 scope ambiguity**
- B3 says "yes add metrics" but fidelity_analysis.py currently doesn't load positions for metrics
- Adding sampled metrics requires loading position tensors → memory + I/O spike + 2-3x runtime
- FID-2 is now 600-900 LOC OR 1500+ LOC depending on B3
- Move B3 to open questions; if yes, split FID-2a/FID-2b

**M2: E1 needs schema specification for error_message/skip_reason**
- Where populated? load_layout? process_group?
- Plain string or structured?
- What enum values? (too_few_seeds, corrupt_positions, metric_load_error, ...)

**M3: D3 normalization may break for non-Euclidean coordinate spaces**
- coord_range works for magnitude but not algorithm-specific conventions
- Some output [0,1], some output 1000s of units
- Recommend: per-family percentiles, not absolute thresholds

## LOW

- L1: pdflatex duplicate claim (F4) needs verification
- L2: merge_fidelity_csvs.py exact behavior to confirm
- L3: 90% vs 85% family threshold conflict in G1
- L4: Consider FID-2a (deterministic+metrics+cleanup) / FID-2b (report) split

## Open Questions That Can Be CLOSED

**Q1: Graph families** — use dagua/eval/graphs.py:2065-2080 + manifest tags
**Q2: Algorithm families** — use dagua/eval/variants.py:129-150
**Q3: Deterministic engines marked?** — Yes: manifest stochastic_engines list (run_benchmark.py:1962-1965); not in list = deterministic
**Q4: Metric recomputation cost** — 30-50 min feasible (95×235×3×9 = 600k calls × ~5ms)

## Things Plan Missed From Sources

**S1**: Claude Explore audit didn't explicitly call out backwards "absence of evidence" logic (claude_fidelity_audit.md:58-62 mentions heuristic fallback but not the inversion).

**S2**: Codex explicitly says "Do NOT reuse `dagua.eval.report.generate_placement_summary_artifacts()`" (codex_qr_design:30-32). Plan mentions at line 550-551 but should be CRITICAL warning in QR-1 spec.

**S3**: Codex's representative-seed policy is NEW design choice, not existing code. Plan should say "implement from scratch" explicitly.

**S4**: Codex pseudocode for graph-relative scoring (lines 277-305) should be quoted verbatim in QR-1 dispatch spec, not just referenced.

## Executive Summary

Plan is fundamentally sound. Synthesis is genuine and valuable. Architecture for both pipelines is correct, procrustes fix is critical, quality/runtime design is well-reasoned. 3 implementation risks: (1) citation errors cost 30-60 min Codex search time; (2) verdict logic refactor (A5) must explicitly DELETE backwards one-sided test, not augment; (3) B3 scope ambiguity could inflate FID-2 by 50-100%. Fix citations, clarify A5 deletion, defer B3 to open questions, add memory/timeout safeguards to QR-1. Core synthesis is excellent.
