<task>
r77-E1: firm up the 10 EVIDENCE-THIN rows (research/probe ONLY; no repo code changes; one
output file). The r76 official ledger (eval_output/fidelity_definitive_r76/
OFFICIAL_R76_LEDGER.md + .json) flags 10 rows whose named-cause dispositions inherit
r75-era evidence that was never experimentally confirmed (drl/neato/maxent families).
JMT directive: every option-3 disposition carries receipts. Produce them.

Repo: /home/jtaylor/projects/dagua (develop, read-only). Identify the 10 rows from the
ledger (search for the evidence-thin flag). For EACH row/cluster:
1. FIRST-DIVERGENCE SUMMARY: instrument dagua's pipeline for that engine on the row's graph
   (1 seed) and compare stage-by-stage against the reference adapter's run (offline
   invocation via dagua/eval/competitors/* is the sanctioned path; OGDF source at
   /home/jtaylor/tools/ogdf-src for reading; ogdf_stress/maxent via scripts/ogdf_runner).
   Name the first diverging stage/quantity.
2. 1-ULP PERTURBATION: nudge one initial coordinate by 1 ULP in dagua's run; show whether
   final-layout divergence magnitude reproduces the dagua-vs-reference gap (the
   chaos-amplification test from r76_FLOOR_DOSSIERS.md -- follow its method exactly).
3. QUALITY PARITY: D-vs-R metric means from per_combo_r76.jsonl.
VERDICT per row: evidenced floor (chaos proof) | portable op difference (name it + effort)
| reference/oracle issue (name it). If a PORTABLE op difference emerges, say so loudly --
that row goes back on the fix queue.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r76_THIN_ROW_DOSSIERS.md
-- per-row tables, verdicts, commands. ASCII only. Budget ~60-90 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one row blocks, document and continue.
</default_follow_through_policy>
