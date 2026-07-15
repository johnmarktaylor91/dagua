<task>
r78-R1: RESIDUAL MOP -- close or floor the three small leftover clusters (research/probe +
evidence; no engine code changes; one output file). From the r77 ledger registry
(eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md):

CLUSTER A -- 8 sfdp disconnected label-box residual rows: the r77 pack2 fix landed but the
residual was narrowed, not proven. Using the pack2 method (instrumented gv 7.0.5 pack.c in
/tmp/gv750-mop, doSplines/label-box dumps; READ r76_IMPL_sfdp_disc_NOTES.md "Pack2"
section), measure per-row WHAT still differs (polyomino cells? margins? placement order?)
on 2 representative graphs. Verdict per row: portable-op-remaining (name it precisely for a
follow-up) OR proven residual (show the measured bound).

CLUSTER B -- 5 graphviz FDP-family rows (classic_fmmm_graphviz_fdp_fidelity): these were
routed out of sfdp triage in r76 and NEVER probed. First-divergence probe vs installed fdp
7.0.5 (fdp -v; lib/fdpgen/*): initial layout, grid/force iterations, xLayout/prism, pack.
Name the first diverging stage per graph. Verdict: portable (name the op + effort) or
floor-candidate (what experiment would prove it).

CLUSTER C -- 2 evidence-thin sgd2 rows (era-rescore low-power crossing residual): run the
decisive same-process probe at full power (100 seeds both sides, single script, params
matched) and produce either an equivalence verdict, a named first divergence, or a
perturbation floor proof.

Repo: /home/jtaylor/projects/dagua (develop, read-only for code; /tmp scratch). Same-process
both-sides generation for ALL probes (hash-determinism lesson -- graphs are now
deterministic but keep the discipline). VERSION PINS: graphviz via `git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r78_RESIDUAL_MOP.md --
per-cluster tables, verdicts, precise follow-up specs for anything portable. ASCII.
Budget ~90 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one cluster blocks, document and continue.
</default_follow_through_policy>
