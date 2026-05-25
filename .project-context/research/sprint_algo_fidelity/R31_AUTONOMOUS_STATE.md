---
run: r31_close_all_gaps
created: 2026-05-24T17:35:00-04:00
state: PHASE_1_RESEARCH
last_updated: 2026-05-24T17:35:00-04:00
user_directive: "research every last way of improving fidelity, integrated plan, implement, rerun. Leave no stone unturned. NOTHING deferred."

phase_1_research_targets:
  drl:               {codex: pending, claude: pending}
  umap:              {codex: pending, claude: pending}
  tsnet:             {codex: pending, claude: pending}
  graphopt:          {codex: pending, claude: pending}
  lgl:               {codex: pending, claude: pending}
  davidson_harel:    {codex: pending, claude: pending}
  infra_recovery:    {codex: pending, claude: pending}
  minor_tightening:  {codex: pending, claude: pending}

phase_2_integrate: pending
phase_3_implement: pending
phase_4_focal_rerun: pending

# DO NOT DELETE -- these are the canonical outputs from the just-completed 100-seed run
existing_outputs_preserve:
  - eval_output/benchmark_100seed_final/results.json (858MB)
  - eval_output/benchmark_100seed_final/positions/   (138MB raw .pt files)
  - eval_output/benchmark_100seed_final/positions.h5 (5.4GB consolidated)
  - eval_output/benchmark_100seed_final/manifest.json
  - eval_output/benchmark_100seed_final/summary.md
  - eval_output/fidelity_report_100seed_final/report.md
  - eval_output/fidelity_report_100seed_final/data/   (sidecar CSVs)
  - eval_output/quality_runtime_report_100seed_final/ (in progress, do not touch)

post_r31_rerun_plan:
  use: --resume against existing eval_output/benchmark_100seed_final
  affected_engines_only: yes (focal, not exhaustive)
  re_aggregate_fidelity: fidelity_analysis.py reads the updated results.json
---

# R31: close every fidelity gap

## Source data

`eval_output/fidelity_report_100seed_final/report.md` final verdicts:

### Strong equivalent (no action needed -- 60+ variants)
classical_mds, kk, pivot_mds, rt, spectral, sugiyama, fa2 (10 of 11), fmmm,
fr, gem, maxent_stress, neato, sfdp, stress_maj

### Weak equivalent (5 graphopt mass/charge + 5 lgl + 4 stress_sgd)
Pass at looser margin. Real room to tighten to strong.

### Partial match (~20 variants)
drl 5, umap 6, tsnet 5, graphopt 4, davidson_harel rounds50/100,
fa2_dissuade_hubs

### Insufficient data (~15 variants)
neulay 6, sgd2_multi 8, davidson_harel_rounds200 -- 0 OK samples each.
R30 dagua-internal bug fixes (CUDA OOM + neulay/tsnet grad_fn) already
committed (commits 5168b9d, 07b6d62). These should run cleanly post-fix.

## Phase 1 research targets (8, parallel)

For each: read source code line-by-line, reference impl line-by-line, identify
every fixable divergence, write ranked plan to ROUND_31_PLAN_<family>.md.

NO IMPLEMENTATION in Phase 1 -- pure research.

Adversarial dispatch: each codex told a claude is doing same target;
each claude told a codex is doing same target. Diverse outputs.

## Phase 2: integrate

I (architect) consolidate the 16 research outputs into a single ROUND_31_PLAN.md
ordered by ROI. Resolve disagreements between codex+claude reports per family.

## Phase 3: implement

Dispatch implementation codexes per family, in parallel where independent.
Each must include regression tests.

## Phase 4: focal rerun

For each modified engine family:
```bash
python scripts/run_benchmark.py --seeds 100 --variants \
    --engines <affected_engines> \
    --output-dir eval_output/benchmark_100seed_final \
    --resume
```
Then re-run fidelity_analysis.py to refresh report.md.

## Case routing (wake-up)

- A. Any phase 1 codex/claude running: yield, ack briefly
- B. All phase 1 done: integrate findings -> phase 2 plan -> dispatch phase 3
- C. Phase 3 codex running: yield
- D. Phase 3 done: focal rerun -> phase 4
- E. Phase 4 done: re-run fidelity_analysis -> iMessage user with delta

## Anti-flail

3 failed rounds on same family -> mark principled_residual_after_max_effort,
move on. Document blocker.
