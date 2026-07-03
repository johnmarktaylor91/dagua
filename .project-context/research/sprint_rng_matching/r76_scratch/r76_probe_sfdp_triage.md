<task>
r76-C4: triage dagua's remaining divergent sfdp/fdp-family combos (44 sfdp rows from r75 +
3 fdp-family rows transferred from the r76 fmmm triage). RESEARCH/PROBE ONLY: no repo code
changes; scratch in /tmp; write only the results file. A rival-lab (Anthropic) architect will
adversarially review your verdicts before any fix is dispatched -- be precise, show numbers.

Repo: /home/jtaylor/projects/dagua (develop, read-only for code). Graphviz reference source
pinned: `git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>` -- NEVER the
working tree. Installed dot/sfdp 7.0.5 binaries are the runtime reference.

READ FIRST:
- .project-context/research/sprint_rng_matching/r75_findings/r75_sfdp_codex.md AND
  r75_sfdp_sonnet.md (dual-lab r75 research -- what was already tried/known)
- r75_findings/r75_ADVERSARIAL_VERDICTS.md (sfdp entries -- which hypotheses were
  REJECTED/NEEDS-EXPERIMENT; do not re-litigate rejected ones without new data)
- r75_findings/r76_PROBE_fmmm_triage.md (context for the 3 fdp-family transfer rows)
- .project-context/research/sprint_rng_matching/r75_RESULTS.md (note: 35 sfdp rows already
  FLIPPED in r75; these 44 are the residue. Also note no-canonical tier: sfdp ignores
  theta/maxiter graph attrs -- those rows are already dispositioned, EXCLUDE them.)

DATA: eval_output/fidelity_definitive/r75_final.jsonl -- rows with engine containing sfdp or
fdp, quality_identical_raw=false, no_canonical_reference!=true. Positions: per-combo
FRESHEST-DIR overlay across eval_output/benchmark_100seed_* dirs (escalation_final,
seeded_refs, r72_fixes, fdp_fix, r73_fixes, r75_fixes, r75_mds_topup, r75_topup2, r76_refs,
r76_gem_fix -- freshest dir wins per combo).

QUESTIONS (answer ALL with numbers):
1. LEG BREAKDOWN: cluster all rows by failing-leg pattern (battery_stress/cross/np
   *_direct_equivalent), D vs R vs margin, disconnected flag, graph size.
2. HAIRLINE vs STRUCTURAL: per-combo Procrustes RMSD dagua-vs-reference (5-seed sample) from
   saved positions. How many <0.01 (near-match: crossings integer discreteness / margin-power
   fails) vs genuinely apart?
3. FIRST-DIVERGENCE HYPOTHESIS per cluster: which sfdp stage diverges first -- initial layout,
   multilevel coarsening (match ordering), spring-electrical iterations, Barnes-Hut octree
   force approximation, or the shared disconnected packer? For ONE representative small
   graph/seed per major cluster, run a stage-by-stage comparison (instrument dagua's pipeline
   ops; for the reference infer from 7.0.5 source + sfdp -v output). Name the first diverging
   quantity per cluster. JMT RULING (binding): NO "FP-chaos floor" label is permitted until
   first-divergence bisection stops finding OP differences. Floor claims need perturbation
   evidence (1-ULP nudge reproduces the divergence pattern).
4. The 3 fdp-family rows: same treatment (these came out of fmmm triage as fdp-routed).
5. RECOMMEND: minimal close path per cluster -- (fix: name the op + expected rung) |
   (floor-evidence: name the bisection endpoint + perturbation experiment) |
   (aggregate-tier candidate). Effort estimate per path. ROI ordering.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_sfdp_triage.md --
commands, tables, per-cluster verdicts + recommendations. ASCII only. Budget ~60 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one question blocks, document and continue with
the rest.
</default_follow_through_policy>
