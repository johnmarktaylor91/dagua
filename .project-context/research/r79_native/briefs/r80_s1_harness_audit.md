# r80-S1: Adversarial audit of the r79 native-algo eval harness (READ-ONLY)

You are auditing the measurement system used to score dagua's native layout algorithm
against external engines. Past sprints found FIVE real oracle/harness bugs (a directedness
oracle handing out ~44 free points, a hash-dependent graph generator, etc.), so the prior
is that more exist. Your job: try to BREAK the harness, not to bless it. Report findings;
do NOT fix anything. Zero file modifications.

Repo: /home/jtaylor/.claude/worktrees/dagua-native (branch r79/native). Python:
.venv/bin/python in that dir. You may run read-only scripts/snippets.

## Hypothesis classes to attack (check each explicitly)

1. **Rescore-path divergence.** scripts/r79_baseline.py supports full runs, --dagua-only,
   and --rescore-only. Verify the dagua rows and the frozen external rows go through the
   IDENTICAL metric code and composite selection (composite_auto vs composite_large,
   row-level `composite` vs metrics.composite_score, tie-band). If frozen external rows
   carry scores computed by an OLDER metric version while dagua rows are rescored fresh,
   the comparison is unfair -- check whether --dagua-only rescores externals from stored
   positions or reuses stored composites.
2. **Directedness oracle.** dagua/eval/graphs.py is_semantically_directed(): sample ~15
   graphs across tags (undirected, dag, scc, weighted, r79_ext) and verify the tag matches
   actual structure (reciprocal edge fraction, acyclicity). Look for default-bias (what
   happens when tags are missing?).
3. **Metric determinism + sampling.** dagua/metrics: crossings/stress are sampled. Verify
   the seed is fixed and that repeat evaluation of the same positions gives identical
   composites. Check whether sampling density differs by N in a way that could flip
   W/L near the 0.5 tie band.
4. **Composite construction.** Read composite_auto: term weights (length/overlap/cross/
   angle/cluster), whether any term can exceed its cap, degenerate-layout exploits (e.g.
   does collapsing everything to a tiny bbox or a line game any term? does zero-edge or
   single-node input crash or score 95?). Try 2-3 pathological position sets on a small
   graph and see if the composite rewards garbage.
5. **Frozen store integrity.** eval_output/r79_baseline/: results.json rows vs
   positions/*.pt -- count mismatches, missing positions for scored rows, stale graph
   names not in the current corpus, timeout/error rows silently counted as losses or
   dropped. Verify the W/T/L counter in the report code matches a hand recount from
   results.json (write a tiny read-only script; print both).
6. **External engine fairness.** For graphviz_dot/sfdp/neato + nx_spring rows: what params
   were they run with (iterations/seed/size handling)? Are node sizes passed to externals
   the same as dagua uses (label-size boxes)? If externals get size-blind runs but are
   scored with size-aware overlap counting, that biases either way -- state which.
7. **Tie-band + best-or-tied accounting.** TIE_BAND = 0.5 composite points. Check the
   comparison is symmetric (dagua >= best_external - 0.5 vs strictly greater etc.) and that
   "best external" excludes errored engines properly.

## Output contract
Markdown report to /tmp/r80_s1_harness_audit_REPORT.md with:
- Verdict line: SOUND / SOUND-WITH-CAVEATS / BROKEN.
- Findings ordered by severity (CRITICAL/HIGH/MEDIUM/LOW), each with file:line, a minimal
  repro snippet or command, and expected-vs-actual.
- A "checked and clean" list (hypothesis classes you attacked that held up).
- Your hand-recount of W/T/L from results.json vs the reported 56/8/29 + 8/2/5.
Return the full report text as your final message too.
