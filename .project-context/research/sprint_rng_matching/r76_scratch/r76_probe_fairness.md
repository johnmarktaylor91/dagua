<task>
r76-F1: SUPERIOR-DISTINCT FAIRNESS TRIAGE (research/probe ONLY; no repo code changes; scratch
in /tmp; write only the results file). r75 classified 79 rows as quality_superior_distinct:
dagua's layouts beat the reference's on the failing legs while being distributionally
distinct. JMT's standing rule: superior-distinct is only an acceptable disposition if it is
FAIR -- same params, same seeds, honest reference. Your job: audit all 79 and bucket them.

Repo: /home/jtaylor/projects/dagua (develop, read-only). Data:
eval_output/fidelity_definitive/r75_final.jsonl and per_combo_r75.jsonl -- rows with
quality_superior_distinct=true. Positions overlay (freshest dir wins per combo):
eval_output/benchmark_100seed_{escalation_final,seeded_refs,drlref_realfix,umap_realfix,
gem_realfix,r72_fixes,fmmm_r3,fdp_fix,r73_fixes,r75_fixes,r75_mds_topup,r75_topup2,r76_refs,
r76_gem_fix,r76_umap_refs,r76_umap_refs2}.

CONTEXT -- TWO FRESH ORACLE BUGS mean "superior" claims are suspect until audited (READ:
.project-context/research/sprint_rng_matching/r76_final_sprint_STATE.md iteration log,
2026-07-03 entries): (1) the stale ogdf_runner ignored iteration params for months; (2)
TODAY: umap_competitor.py returned seeded torch.randn for graphs with <=3 nodes -- dagua
"beat" a reference that never ran the algorithm (fixed, develop 7d1f090). Any superior-
distinct row could be the same genre: dagua-does-the-algorithm vs reference-does-something-
degenerate.

AUDIT EACH ROW (tables with numbers; cluster rows sharing engine/graph/cause):
1. ORACLE SANITY per engine/graph cluster: does the reference actually respond to its
   params on this combo (compare reference positions/metrics across that engine's variants
   on the same graph -- bit-identical across param variants = ALARM, name it)? Does it
   respond to seeds? Any degenerate/fallback path in the adapter for this graph class
   (tiny graphs, disconnected, self-loops, multi-edges -- read the relevant
   dagua/eval/competitors/*.py adapter code path)?
2. PARAM PARITY: were reference rows generated with params mirroring the dagua variant
   (iters/steps/rounds/neighbors)? Check the __for__ engine naming + any adapter clamping.
3. MAGNITUDE + DIRECTION: by how much is dagua "better" per failing leg (stress/cross/np,
   D vs R vs margin)? Superior by a hair inside margin noise vs superior by 2x are
   different dispositions.
4. MATCH-THE-WORSE-REFERENCE FEASIBILITY: for genuinely-fair rows, is the reference's
   worse behavior PORTABLE (a documented algorithmic choice dagua could mirror under
   fidelity_mode, e.g. a known inferior default/tie-break), or is it emergent? Recommend:
   keep-superior-distinct (fair, non-portable) | port-the-worse-behavior (name the op +
   effort) | reference-bug (name it -> refs regen needed) | reclassify (margin artifact).
5. ROI ordering of any recommended fixes/regens.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_fairness.md --
commands, per-cluster tables, verdicts, recommendations. ASCII only. Budget ~60 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one row/cluster blocks, document and continue.
</default_follow_through_policy>
