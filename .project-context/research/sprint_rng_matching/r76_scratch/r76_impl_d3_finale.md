<task>
r76-D3 FINALE: generate the OFFICIAL r76 ledger, report, and results -- the closing act of a
two-month fidelity campaign. Every engine family has been fixed or dispositioned this
sprint; your job is ONE authoritative scoring pass + the official tiered ledger + the
results document, under the corrected ledger rules. READ FIRST (mandatory, in this order):
1. .project-context/research/sprint_rng_matching/r76_final_sprint_STATE.md -- the COMPLETE
   iteration log (esp. all 2026-07-03 entries: every rule below has its evidence there).
2. .project-context/research/sprint_rng_matching/r75_RESULTS.md -- the r75 results doc
   (structure template + baseline numbers).
3. r75_findings/r76_FLOOR_DOSSIERS.md, r76_PROBE_fairness.md, r76_IMPL_*_NOTES.md -- the
   disposition evidence you will cite.

Repo: /home/jtaylor/projects/dagua (develop). Work on branch r76/ledger (create off
develop; you may work in the MAIN checkout since all benches are done -- verify with
`pgrep -f run_benchmark` first; if any bench is running, STOP and report).

STEP 0 -- PRECONDITIONS (updated after D4 landed, merged ce4562d):
- Determinism: VERIFIED GREEN (D4 self-check, 12 combos, deterministic verdicts). Proceed.
- The D4 validator run found 234 param-sensitivity FAILURES on HISTORICAL dirs (ogdf_gem
  103, ogdf_fmmm 95, igraph_mds 21, ogdf_stress 15) + 241 seed-era warnings. ADJUDICATE,
  do not suppress: (a) the ogdf_gem/fmmm failures are the archaeological stale-runner
  artifacts -- for EVERY ogdf combo in the universe, verify the freshest-wins overlay
  resolves its REFERENCE side to a fresh param-sensitive dir (r76_refs/refs2/refs3); any
  combo whose refs still resolve to a stale param-noop dir gets an explicit
  STALE-REFERENCE flag and routes to the aggregate tier or insufficient -- NEVER a
  per-seed verdict against a param-noop reference. (b) igraph_mds: the fairness probe
  says default/fidelity legitimately share one reference (no material params) --
  clamp-equivalence candidate; verify and add to the whitelist WITH evidence. (c)
  ogdf_stress: r75 proved stress outputs byte-identical between runner binaries; if its
  param-identity reflects genuine convergence (extra iterations no-op on those graphs),
  whitelist with evidence; else flag. Document all adjudications in the report.
- KNOWN pre-existing unrelated test failure (do not chase, note only):
  tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border.

STEP 1 -- AUTHORITATIVE SCORING PASS: definitive_fidelity_analysis.py --mode full over the
FULL combo universe (derive from eval_output/fidelity_definitive/per_combo_r75.jsonl combo
ids) with the COMPLETE freshest-wins dir chain in this order:
escalation_final, seeded_refs, drlref_realfix, umap_realfix, gem_realfix, r72_fixes,
fmmm_r3, fdp_fix, r73_fixes, r75_fixes, r75_mds_topup, r75_topup2, r76_refs, r76_gem_fix,
r76_refs2, r76_umap_refs, r76_umap_refs2, r76_umap_fix2, r76_maar_bench, r76_sfdp_fix,
r76_sfdp_refs, r76_sfdp_fix2, r76_sfdp_fix3, r76_sugiyama_topup, r76_igraph_fix, r76_refs3
(all under eval_output/benchmark_100seed_*). Output:
eval_output/fidelity_definitive/per_combo_r76.jsonl (use the new overwrite-or-fail
semantics). Batch by engine family if memory requires; workers <=6; OMP_NUM_THREADS=1.

STEP 2 -- TIERED LEDGER under the CORRECTED RULES (every rule has STATE-log evidence):
a. fidelity-identical := quality_identical_raw OR quality_identical_exploratory (the
   canonical-reference gate is reported as metadata, NEVER used to demote fidelity tiers).
b. Mode-B (deterministic) rows tier by d_R: <1e-9 bit-exact; <0.01 identical-dist;
   <0.1 close; else divergent -- map onto the official report's existing tier names
   (read how the r75 official report generator tiers mode-B rows and extend consistently).
c. NO-CANONICAL tier additions: ALL classic_sfdp p_neg2/theta04/theta08/steps200 rows
   (fresh-ref param-noop proof, STATE 2026-07-03 ~19:55); umap parallel_multiedge rows are
   NOW SCOREABLE (adapter fixed) -- they are bit-identical, count them.
d. NAMED-CAUSE divergent dispositions (each cites its dossier; each states per-row quality
   parity D-vs-R):
   - MAAR packing tie-breaks: random_dag_50 fmmm steps10/100/200 + gem iters2000
     (r76_IMPL_fmmm_disc_NOTES attempt-2; quality equal-or-better).
   - sfdp spline-box polyomino occupancy: 10 rows / 5 graphs (r76_IMPL_sfdp_disc_NOTES
     C4d; quality flags: 2 clusters ~5% worse -- STATE THEM PLAINLY).
   - igraph GLPK degenerate-LP basis selection: the igraph-family far tier
     (r76_IMPL_igraph_NOTES; vendoring excluded).
   - graphviz sugiyama stages B-D (labels/clusters) + aux minlen 1pt half-width: the
     graphviz-fidelity far tier (r76_IMPL_mincross_NOTES A4b/A4c dossiers).
   - umap spectral eigenspace floor: random_dag_50/200 nn5 (r76_FLOOR_DOSSIERS, 1-ULP
     proof, quality parity).
   - mds connected: "proven member of reference equivalence class" (r76_FLOOR_DOSSIERS).
   - drl/neato/maxent (8 rows): verify their r75 dispositions carry adequate named-cause
     evidence; if any row lacks it, run the cheap missing probe (perturbation or
     first-divergence summary) or flag it explicitly as evidence-thin -- do NOT silently
     inherit.
e. SUPERIOR-DISTINCT relabels per r76_PROBE_fairness.md: remove the label from the 8
   reference-bug rows (now rescored honestly); 13 sfdp param-noop variant rows ->
   no-canonical; keep fair rows with the probe's caveat labels.
f. ERA/POWER flags: any row scored with matched n<100 gets an explicit low-power flag
   naming the seed ranges; rows whose only reference is seed-42-era get listed.
g. POPULATION/AGGREGATE TIER (D2, S3 design: metadata-only, BH-corrected across the
   family): apply to 2000/5000-node rows that are insufficient for per-seed scoring;
   document the method in the report. No new benches.
STOP CRITERIA CHECK: every combo ends with EXACTLY ONE disposition; ZERO rows left as bare
"divergent" without a named cause + documented why-not-closed. Print the full disposition
count table and the list of any row that fails this -- the sprint does not close with
unnamed rows.

STEP 3 -- GATES: run the official gate suite (the r75 report pipeline's gates; gate_5
laundering must be 0/40; document every gate verdict). Determinism: re-run the scorer
self-check on a 12-combo sample from the new per_combo_r76.jsonl chain -- must be clean.

STEP 4 -- OUTPUTS:
- Official report dir: eval_output/fidelity_definitive_r76/ (mirror the r75 official
  report structure).
- .project-context/research/sprint_rng_matching/r76_RESULTS.md -- structured like
  r75_RESULTS.md: headline disposition table (counts per tier, r75 -> r76 deltas), "What
  r76 discovered" (the FOUR oracle bugs: umap randn fallback, sfdp param-noop attrs,
  seed-era mismatches, max-nodes silent exclusions; the scorer demotion gate; the fix list
  per engine w/ commit shas), "What r76 shipped" (all merged branches + commits), the
  named-cause registry (each residual cluster w/ dossier ref + quality flags), key
  artifacts list, honest limitations (2000/5000 aggregate tier, low-power rows,
  evidence-thin rows if any).
- Commit everything on r76/ledger (conventional commits, NO AI attribution, no push/merge).

DELIVERABLES: r76_RESULTS.md + official report + per_combo_r76.jsonl + gate verdicts +
the disposition count table in your final summary. ASCII only. This report will be read by
JMT as the campaign's closing document -- honest, precise, zero laundering.
</task>
<completeness_contract>
Done = scoring pass complete over the full universe, every combo carries exactly one
disposition (zero bare divergent), gates green (gate_5 0/40), r76_RESULTS.md + official
report committed. If ANY precondition fails (running bench, red determinism, validator
failure outside the whitelist) -- STOP and report rather than shipping a compromised
ledger.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/ledger only. Never modify engine/pipeline/scoring code
(the infra branch owns tooling; you CONSUME it). Never launch new benches. The analysis
output writes are the only eval_output writes allowed.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity or precondition failure.
</default_follow_through_policy>
