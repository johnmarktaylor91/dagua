NEW SESSION: Read this file first. Then read CLAUDE.md, AGENTS.md,
and .project-context/knowledge/. The knowledge files contain durable
project understanding. This baton file contains live session state.
After reading everything, your first action should be: dispatch 10
audit agents (5 Claude + 5 Codex) for the definitive fidelity run
as described in the "Immediate Next Steps" section below.

## Goal

Launch the DEFINITIVE fidelity benchmark run: 60 seeds, 600s timeout,
all 97 algorithm families, completely from scratch. This replaces all
previous 30-seed runs. The run will take ~1 week. While it runs, we
work on other things.

## Completed This Session

- Fixed ALL identified code differences across 4 reimplementation files:
  - neulay.py: 5 fixes (cKDTree repulsion, spring dedup, lr, step budget, numpy seed, manual GCN)
  - sgd2_multi.py: 8 fixes (centering, CrossingDetector, BFS neighborhood, angular res, vertex res, aspect ratio, scheduler offset, cyclic sampling)
  - fa2.py: 2 fixes (LinLog double-division, outboundAttCompensation)
  - tsnet.py: already matched, no changes needed
- Fixed analysis methodology:
  - Within-vs-between Procrustes (Mann-Whitney one-sided) as primary verdict signal
  - Scale-invariant metrics only (removed edge_length_mean, overlap_count)
  - Proportion-based family aggregation (90% threshold, not all-or-nothing)
  - Mirror-aware Procrustes, PValueBucket.add fix
- 30-seed results: 57 strong, 6 weak, 34 partial, 0 divergent
- All code verified by 8 independent audit agents + adversarial Codex critic + independent Codex reviewer
- Committed as 3af2d2e on feat/bench-and-aesthetics
- Previous results archived to eval_output/archives/*_20260331_132440*

## In Progress

Nothing currently running. The definitive run needs to be set up and launched.

## Immediate Next Steps

The user's explicit 7-step plan for the definitive run:

1. **Dispatch 10 agents (5 Claude + 5 Codex)** for final line-by-line sweep of
   ALL 97 families. Also audit the full pipeline for easy iteration (targeted
   reruns, adding seeds, etc.). Ensure analysis computes BOTH within-vs-between
   Procrustes AND TOST -- everything we could want.

2. **Write combined plan** from all 10 agents' findings. Iterate with adversarial
   Codex until zero objections.

3. **Dispatch Codex workers** to implement all remaining fixes.

4. **Launch definitive benchmark**: 60 seeds, 600s timeout, fresh from scratch,
   maximum parallelism. Keep "skip after 3 consecutive timeouts" behavior.
   Save previous results safely (already done -- archived).

5. **Report format**: per-algorithm breakdown showing % of graphs passing at each
   threshold. Clear failure point identification. Adversarial Codex critiques
   report format for rigor/clarity/readability, iterate until satisfied.

6. **Documentation**: careful notes for the ~week-long run so nothing gets lost.

7. **Lessons learned**: apply everything from this iteration process.

## Context the New Instance Needs

- Read memory files: feedback_iteration_lessons.md, feedback_fidelity_verdicts.md,
  project_definitive_run.md -- these contain critical lessons from this session
- The H5 file at eval_output/variant_bench_full/positions.h5 is rebuilt and clean
  (370K keys, perfect sync with results.json). But for the definitive run we start
  fresh.
- The report script (generate_fidelity_report.py) needs updating -- it still
  expects edge_length_mean and overlap_count columns that were removed. The Codex
  worker in step 3 should fix this.
- Reference engines are NOT auto-included by --variants flag on run_benchmark.py.
  Must be run explicitly or recovered from .pt files.
- The safe_purge_variants.py script uses exact engine_name matching for results.json
  but the H5 keys contain engine names as substrings -- be careful with substring
  matching (it deleted reference positions in this session).
- consolidate_positions_hdf5.py opens H5 with mode "w" (destructive replace).
  Always write to temp file and rename atomically.

## Promises to User

- Definitive run with 60 seeds, 600s timeout, all engines, from scratch
- 10-agent audit before launching
- Adversarial critique of both the plan and the final report format
- Careful documentation so the ~week-long run doesn't get lost
- Apply all lessons from this iteration process

## Git State

- Branch: feat/bench-and-aesthetics
- Uncommitted: temporary scripts (scripts/_final_run.py, _final_run_v2.py,
  _overnight.py, _purge_h5.py, _rebuild_h5.py) -- these are one-off helpers,
  not needed going forward
- Latest commit: 3af2d2e feat(fidelity): match all reimplementations to references + fix analysis methodology
- Previous commit: ae2365d feat(fidelity): complete fidelity hardening sprint

## Running Processes

None.

## START HERE

Dispatch 10 audit agents (5 Claude + 5 Codex) covering all 97 algorithm
families and the full pipeline. Split by algorithm group:
- Agents 1-2: FA2 variants (11 families)
- Agents 3-4: SGD2 + stress_sgd variants (12 families)
- Agents 5-6: NeuLay + t-SNE + UMAP variants (16 families)
- Agents 7-8: FR + KK + spectral + MDS + other classics (30+ families)
- Agents 9-10: Pipeline infrastructure + report format audit
Each agent reads both dagua code AND reference code line by line.
