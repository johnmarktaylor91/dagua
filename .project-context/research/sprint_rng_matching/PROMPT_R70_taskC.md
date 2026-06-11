<task>
Implement Task C (report + aggregation) of the r70 definitive fidelity analysis for the
dagua repo (you are in /home/jtaylor/projects/dagua).

AUTHORITATIVE SPEC -- read it FIRST, in full:
  .project-context/research/sprint_rng_matching/SPEC_definitive_fidelity_analysis.md
(version 6, APPROVED). Implement EXACTLY; the SPEC wins over this prompt. Tasks A and B
are committed: dagua/eval/distributional_fidelity.py (stats core; call bh_fdr/assign_rung
from it) and scripts/definitive_fidelity_analysis.py (runner; READ its row schema from its
code and from any existing /tmp/r70_smoke.jsonl or
eval_output/fidelity_definitive/per_combo.jsonl rows). Do not modify either.

CREATE EXACTLY ONE FILE (touch nothing else):
  scripts/definitive_fidelity_report.py

It implements spec sec. 10 Task C -- a CLI that consumes runner outputs and produces the
permanent record:

1. FULL-RUN FDR PASS (spec sec. 7): global BH q=0.05 within families {p_track} and
   {p_tost}; p_typ at RAW alpha=0.05 (no BH; report expected false-atypicality count);
   p_diff annotation-only. REFUSE to apply full-run FDR unless the completeness assertion
   passes: every combo in failing_map_final.json appears in per_combo.jsonl exactly once
   with matching spec_version (controls are exempt -- they use their own local families).
   Then assign final rungs via Task A's assign_rung.

2. ACCOUNTING PARTITION (spec sec. 9, exact order TIMEOUT -> ERROR_NO_DATA -> ESCALATION
   -> REF_NO_DATA -> RUNG_0, first match wins; timeout = status 'timeout' OR error text
   /timeout/i in eval_output/benchmark_5seed_final/results.json; scope = 64 escalating +
   39 bit-identical engines; the 8 deterministic engines use refresh-dir verdicts; 4
   NO_REFERENCE + 3 UNVERDICTED_OTHER appendix-only). Assert per-engine bucket sums equal
   universe size; assert all 118 variants (eval_output/fidelity_report_final/triage_final.md
   inventory) appear in exactly one place; assert no engine has mixed reference modes.

3. DOMAIN MAP (spec sec. 9): hierarchy-requiring = {sugiyama*, reingold_tilford*, rt*};
   per-graph has_hierarchy from GENERATOR-FAMILY SEMANTICS in dagua/eval/graphs.py source
   (dag_*/tree_*/dependency/pipeline families = hierarchical; anything passed through
   _undirected_to_dag from an undirected generator = NOT) -- emit the full per-graph
   classification table into the report for CC review. DOMAIN_MISMATCH combos reported
   separately, excluded from headline denominators.

4. AGGREGATION + HEADLINES (spec sec. 9 EXACTLY): per-engine and per-family rung
   percentages (full-universe AND escalation-only), size-bin degradation curves
   (N <= 50 / 51-200 / 201-1000 / >1000 via num_nodes), runtime ratios, dispersion/
   tracking diagnostics, twin-rank deciles, Mode B mean(W_D)/sqrt(2) distributions with
   over-guard and [0.7,0.85] band counts; headlines DISTRIBUTIONALLY_MATCHED /
   SEED_FAITHFUL / REF_COMPATIBLE / BIT_EXACT / UNDETERMINED with the min-denominator
   rules (>=10 usable on-domain, >=50% escalation usable (vacuous at zero), >=10
   informative for REF_COMPATIBLE).

5. CONTROLS GATE EVALUATION (--controls mode; spec sec. 8): read
   eval_output/fidelity_definitive/controls/*.jsonl, evaluate gates 1-4 with
   CONTROLS-LOCAL families, emit controls/gate_results.json with per-gate PASS/FAIL +
   measured values, and exit nonzero if any gate fails.

6. INVARIANCE SPOT-CHECK (spec sec. 7): sample up to 200 combos
   (sha256("r70::spotcheck") over sorted keys) from {rung-4 or tracking-fail} x
   {symmetric-or-disconnected graphs} (symmetric = nontrivial automorphism group via
   equivalence_metrics toolkit, computation capped/try-except; disconnected flag from
   runner rows); re-score diagonal/B-column with toolkit invariance distance; report
   would-flip counts. Report-only.

7. OUTPUTS (spec sec. 9): eval_output/fidelity_definitive/DEFINITIVE_FIDELITY_REPORT.md
   and FOUR_TIER_CATEGORIZATION.md with ALL pre-registered content: methodology summary +
   decisions log (q95, 90%, FDR choices incl. no-BH-for-conformal rationale, margins,
   Mode B design + per-engine single-draw provenance table (derive from
   dagua/eval/variants.py + dagua/eval/competitors/ source: igraph-no-RNG vs
   binary-at-default-seed), plain-Procrustes + free-aspect exception), controls results,
   per-engine tables, headlines, degradation curves, domain table, Mode B disclosure +
   seeded-reference follow-up recommendation, invariance spot-check, expected
   false-atypicality accounting, INSUFFICIENT_DATA listings with reasons, deterministic
   sub-verdicts incl. the Tier-1-adjacency note, rung0-reverify results
   (stale_rung0_failed_reverify flags), appendices (NO_REFERENCE fr_kk/kk_fr; fcose
   no-port; the 3rd UNVERDICTED variant -- check triage), SUPERSESSION STATEMENT
   enumerating: WHERE_WE_STAND.md group tables, ALLGRAPHS_SUMMARY.md, fidelity_report_r69,
   fidelity_report_final per-variant verdicts. Also write per_combo.json (consolidated,
   rungs + q-values included) and oc_simulation.json (call Task A's run_oc_simulation).
</task>

<completeness_contract>
Done means: file exists; --help works; ruff clean; a DRY RUN on whatever jsonl exists
(/tmp/r70_smoke.jsonl or a synthetic 20-row file you construct matching the runner schema)
renders both .md outputs without crashing, with every assertion implemented (assertions may
legitimately FAIL on smoke data -- provide --no-strict to render anyway, default strict).
Print the rendered report's section headers at the end. Do NOT git commit.
</completeness_contract>

<verification_loop>
Iterate the dry run until both documents render with all sections present. Mark any
interpretation with "SPEC-INTERPRETATION:" comments.
</verification_loop>

<action_safety>
Read-only except the one new file, eval_output/fidelity_definitive/ outputs, and /tmp.
Never invoke layout engines.
</action_safety>

<default_follow_through_policy>
Most reasonable low-risk interpretation; stop only for genuine walls. Do not modify
Tasks A/B.
</default_follow_through_policy>
