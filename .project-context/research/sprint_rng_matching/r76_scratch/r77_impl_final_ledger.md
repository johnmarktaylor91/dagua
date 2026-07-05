<task>
r77 GRAND FINALE: regenerate the OFFICIAL ledger, report, and results with EVERY r77 fix
and correction consolidated. This supersedes the r76 official ledger. READ IN ORDER:
1. .project-context/research/sprint_rng_matching/r76_final_sprint_STATE.md -- the COMPLETE
   r76+r77 iteration log (all 2026-07-03/04 entries; every correction below has evidence
   there).
2. .project-context/research/sprint_rng_matching/r76_RESULTS.md + the r76 official ledger
   (eval_output/fidelity_definitive_r76/) -- the baseline being superseded.

Repo: /home/jtaylor/projects/dagua (develop, HEAD ~6f141e1+). Branch r77/final-ledger off
develop, work in the MAIN checkout (verify `pgrep -f run_benchmark` empty first,
self-match-aware). PRECONDITIONS: scorer determinism was verified green (D4); the D4
validator tripwires exist -- run the validator and adjudicate per the r76 finale's
whitelist decisions.

STEP 1 -- AUTHORITATIVE FULL-UNIVERSE SCORING PASS: same combo universe as r76 finale
(per_combo_r75.jsonl ids), full freshest-wins chain = the r76 finale's 26-dir chain PLUS
(in this order, after r76_refs3): r77_mds2, r77_sfdp_pack2, r77_sugiyama_a5b,
r77_sugiyama_final, r77_sugiyama_wired, r77_igraph_bk (if exists -- check), r77_maar (if
exists), r77_randomdag, r77_era_refs (all under eval_output/benchmark_100seed_*; verify
each exists, skip absent ones with a note). Output:
eval_output/fidelity_definitive/per_combo_r77.jsonl (overwrite-or-fail semantics).
KNOWN SHAPE-MISMATCH HAZARD: random_dag_50/200 old-era rows have different node counts
than the new canonical realization -- the freshest-wins overlay resolves BOTH sides of
those combos to r77_randomdag (verify; if any random_dag combo resolves a side to an
old-era dir, exclude that side's dir for that combo and note it). If the scorer crashes on
a shape mismatch elsewhere, exclude that combo, flag it, continue -- report all such rows.

STEP 2 -- TIERED LEDGER: same corrected rules as the r76 finale (identical := raw OR
exploratory; mode-B d_R tiers; p_neg2/theta/steps no-canonical; fairness relabels; era
flags NOW mostly cleared -- recount; population/aggregate tier for 2000/5000 rows) PLUS the
r77 CORRECTIONS (each with its dossier):
- MAAR packing cluster: RETIRED -- all rows identical/equivalent on same-realization data
  (r77_randomdag.jsonl); the named cause was oracle bug #5, not MAAR ties.
- umap "eigenspace floor" random_dag rows: RETRACTED -> identical (same bug); AMEND the
  r76_FLOOR_DOSSIERS reference in the report text accordingly. Connected-mds equivalence-
  class floor REMAINS (r77 era rescore at full power confirms statistical distinguishability
  exactly as the eigensign dossier predicts; scoring-registration reflection note from M5).
- sfdp disc: named cause CORRECTED spline->LABEL-BOX (measured, fixed, r77); remaining sfdp
  residual recount from fresh data.
- sugiyama: use r77_sugiyama_wired.jsonl tiers for <=300 rows (igraph 141 bit-exact; labels
  91x); graphviz far-tail named cause = recursive cluster rank-collapse (A9 dossier);
  igraph far-tail = GLPK-solved-then-BK-residual recount from wired data.
- ba_500 gem full-power divergents + any era-rescore stragglers: bucket honestly (existing
  gem float32 dossier if applicable; else flag EVIDENCE-THIN explicitly).
- 3 era-rescore honesty corrections (closed->divergent at full power): apply, list them.
STOP CRITERIA unchanged: every combo exactly one disposition; zero bare divergent; print
the full table + the r76->r77 delta table.

STEP 3 -- GATES: full gate suite; gate_5 0/40; scorer self-check on a 12-combo sample.

STEP 4 -- OUTPUTS: eval_output/fidelity_definitive_r77/ (official report dir);
.project-context/research/sprint_rng_matching/r77_RESULTS.md (structured like r76_RESULTS:
headline table w/ r76->r77 deltas; "What r77 discovered" -- oracle bug #5 + the corrected
causes + the retracted floors + the 4 delegation/inference lessons; "What r77 shipped" --
all merged branches/commits: GLPK, BK, half-width, labels, clusters, wiring, sfdp label-box,
mds chain, graph determinism, crash+cache fixes; the honest-residual registry w/ dossier
refs; limitations). Commit on r77/final-ledger (conventional, NO AI attribution, no
push/merge). If pre-commit blocks on generated artifact size, note it and commit the
non-generated files normally -- do NOT bypass hooks yourself; leave oversized artifacts
untracked with a note.
</task>
<completeness_contract>
Done = full-universe pass complete, every combo one disposition, zero bare divergent,
gates green (gate_5 0/40), delta table r76->r77 printed, r77_RESULTS.md + official report
committed. STOP and report rather than ship a compromised ledger (precondition failure,
crash storm, gate red).
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/final-ledger only. Never modify engine/scoring code; never
launch benches. Analysis outputs are the only eval_output writes.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity or precondition failure.
</default_follow_through_policy>
