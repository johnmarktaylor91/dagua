# BATON: r78 fidelity endgame -- resume here (2026-07-10 16:25)

## ONE-LINE STATE
Final ledger assembly, ~90% done. All benches/rescores landed EXCEPT sgd2 refs (running
detached, resume-safe) + a killed fmmm bisection agent. Merged per-combo file is current;
named-cause registry down to 43 (from 249 at r77). Remaining: fold sgd2, adjudicate ~6
rows, build final ledger (must exit zero), rerun gates, write close-out to 3 places.

## GROUND TRUTH FILES (all under eval_output/fidelity_definitive/)
- `per_combo_r78_merged.jsonl` (3955 rows) = THE working per-combo file. r77 base + every
  r78 rescore folded in (707 initial + 59 neato + 18 C1 + 18 sugiyama + 15 wave2). This is
  the input to the final ledger build.
- `causes_r78.json` (43 rows) = named-cause sidecar. Each remaining divergent row maps to a
  documented floor cause. Retired 206 causes across the round (sugiyama 145->0, neato 54->0).
- Preview ledger dir: `eval_output/fidelity_definitive_ledger_r78_preview/` (STALE -- rebuild).

## RESUME CHECKLIST (in order)

### 1. Wait for sgd2 refs bench, then rescore + fold
- Bench: `eval_output/benchmark_100seed_r78_sgd2_refs`, pid file /tmp/r78_sgd2_refs.pid,
  log /tmp/r78_sgd2_refs.log. Was ~82% when session dropped; resumed detached ~16:22.
  Check: `grep Done: /tmp/r78_sgd2_refs.log | tail -1` (want "Done: 2030 total ... 0 errors").
- When done, rescore the 58 sgd2 combos. Combo list: all `*::classic_sgd2_multi_with_crossing`
  rows currently INSUFFICIENT in the merged file. Command pattern (append the new ref dir LAST):
  ```
  DIRS=$(awk '{printf "--data-dir %s ", $0}' /tmp/r78_rescore_dirs.txt)
  python3 scripts/definitive_fidelity_analysis.py --mode full $DIRS \
    --data-dir eval_output/benchmark_100seed_r78_sgd2_refs \
    --combos-file <58 sgd2 combos> --workers 4 --overwrite \
    --output eval_output/fidelity_definitive/per_combo_r78_sgd2fix.jsonl
  ```
  Then fold into per_combo_r78_merged.jsonl by combo_id (new wins), same as prior waves.
  EXPECT: mostly DIST_EQ or POSITIONAL (sgd2_multi is stochastic; per r77 it was distributional).

### 2. Adjudicate the residual DIVERGENT_UNEXPLAINED rows (write into causes_r78.json)
Rebuild preview (step 3 cmd) to get the current unexplained list, then:
- **sfdp disconnected rows** (parallel_cycles_4x5 x{default,graphviz_fidelity,p_neg2},
  multi_component_80, disconnected_encoder_residual, disconnected_label_cycle_collage,
  kitchen_sink_platform_graph ::classic_sfdp_p_neg2): VERIFIED per-component Procrustes = 0.00
  (parallel_cycles) / 1.4e-3 (multi_component); global distance is pure INTER-COMPONENT
  PACKING. Cause string: "Graphviz SFDP disconnected component-packing residual (per-component
  layout matches to <=1.4e-3; only inter-component pack arrangement differs, outside the
  finite-graph-distance stress construct)". Evidence in prior session; re-verify one row with
  the per-component script pattern if desired (scipy.sparse.csgraph.connected_components +
  per-comp Procrustes).
- **fmmm genuine-divergence rows** (from insufficiency+fmmm agent, cluster D): powerlaw_2000::
  steps10 has fixed_m15_dist_equivalent=TRUE -> adjudicate DIST_EQ-at-m15. grid_50x50/rgg_2000::
  steps10 are borderline (percentile 0.983-0.988, cross-dist < within-spread) -> quality/near.
  small_world_2000::steps10 + hub_and_spoke_3x20::fdpfid + random_dag_200::fdpfid = genuine
  trajectory drift with partial seed correspondence. The KILLED fmmm bisection agent's last
  note: "random_dag_200 is multi-component (19 comps) -- must compare per component". So:
  re-run per-component Procrustes on random_dag_200::fdpfid; if per-comp matches, it's a
  packing row like sfdp. For the truly-chaotic ones (hub_and_spoke, small_world_2000):
  cause = "fmmm/fdp trajectory-chaos floor: RNG/init matches (diag<offdiag) but chaotic
  force dynamics amplify float differences; drawing-quality equivalent (quality_equivalent_raw)".
  DECISION NEEDED: JMT bar accepts quality-indistinguishable as a pass -> these are QUALITY_EQUIVALENT
  tier via the ledger builder, NOT divergent, IF quality_identical_raw is set. Check each.
- **MDS deterministic rows** (org_chart_1_5_4_8, wide_3_50_3 ::classic_classical_mds_default):
  deterministic, era-stable to 12 digits, dist_equivalent=False -> MDS eigenspace floor
  (already a cause family). Add to causes_r78.json with the eigenspace cause string.
- **fam2 neato rows**: RESOLVED already (folded via neato mode-A rescore; the 5 mis-moded ones
  were in the neato59 list). Verify none remain unexplained after rebuild.

### 3. Build final ledger (MUST exit 0)
```
python3 scripts/build_definitive_ledger.py \
  --per-combo eval_output/fidelity_definitive/per_combo_r78_merged.jsonl \
  --causes eval_output/fidelity_definitive/causes_r78.json \
  --output-dir eval_output/fidelity_definitive_ledger_r78
```
If DIVERGENT_UNEXPLAINED > 0 it exits nonzero and lists them -> go back to step 2, adjudicate
each with evidence (NEVER blanket-adjudicate; each row needs per-component or bisection proof
or a named floor). --allow-unexplained only for preview inspection, never for the final.

### 4. Rerun control gates (must both be 100%)
Gates live in scripts/definitive_fidelity_report.py (evaluate_gate_negative ~2480,
evaluate_controls). Post-b15d08b + regenerated controls, gate2=39/39 and gate3=20/20.
Re-verify with the assign_rung path on controls_full data (pattern used earlier this session).

### 5. Close-out deliverable (JMT directive -- THREE places)
Full spec in FINAL_RESCORE_RUNBOOK.md "Deliverable on completion". Honest tier table
(counts + percents), every remaining floor row with evidence class (mathematical/economic/
pending), what changed this round, monster-bench note. Deliver:
  1. iMessage: ~/.claude/scripts/send-to-jmt.sh "..." (ASCII ONLY -- app drops non-ASCII)
  2. Vault: interfaces/queries/2026-07-10-dagua-fidelity-closeout/ + file-for-review.sh
  3. This chat: full text.

## KEY NUMBERS (preview, pre-sgd2, will improve)
Scoreable ~3653 (excl 302 no-canonical). Bit-exact ~332, positional-identical ~1650,
distributional ~1400, quality-only ~30-45, superior ~12, named floors ~43, unexplained
target 0, insufficient ~58 (the sgd2 rows, closing now).

## SYSTEMIC FINDINGS THIS ROUND (for the write-up + follow-ups)
- `--seed-refs` is INERT: never adds engines. Every r78 micro-bench produced ZERO reference
  rows until we put `<ref>__for__<engine>` in `--engines`. FOLLOW-UP (post-ledger, don't
  perturb in-flight): add a run_benchmark.py guard that ERRORS when --seed-refs names a base
  engine whose __for__ variants aren't selected. (run_benchmark.py:1104-1200 select_engines.)
- Overlay clobber: later dir with 1 ok row evicts an earlier dir's 100 seeded rows wholesale
  (definitive_fidelity_analysis.py:530-591). Caused the 54 neato false-divergents. Not a bug
  to fix (anti-era-mixing is deliberate) but a footgun -- document.
- MDS zero-collapse: FIXED + committed 6dfccfa (dsyevr m=0 on degenerate spectra -> full eigh).
- size_policy._SIZE_AWARE_EXTERNALS defaults True (r80-P6, the OTHER tab's work): graphviz ref
  re-benches at HEAD now inject node sizes -> different refs than historical. Wasn't needed this
  round (seeded neato refs already existed) but watch for it in any future graphviz ref bench.

## COMMITS THIS ROUND (develop)
6dfccfa mds dsyevr fix | a83a5db rescore/preview docs | (earlier) b15d08b rung 2'/2'w split +
controls regen | b795710 seed_tracking bypass removal | d528741 variant_param_names allowlists +
torchlens guards | plus docs commits. All AI attribution stripped per JMT rule.

## GIT STATUS
Branch develop. Working tree clean except eval_output/ (gitignored data) + the causes/merged
jsonl (also under eval_output, gitignored -- these are DATA, regenerable, not committed).
The build_definitive_ledger.py + tests are committed (6afec9c). classical_mds fix committed.
