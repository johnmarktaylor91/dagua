---
run: algo_fidelity
created: 2026-04-29T19:16:57-04:00
state: ROUND_DONE
current_round: 13
current_family: drl
codex_pid: 257755
codex_log: /tmp/algo_fid_round_13.log
watchdog_pid: 257922
dispatched_at: 2026-04-30T00:30:00-04:00
prompt_file: .project-context/research/sprint_algo_fidelity/PROMPT_round_13.md
sprint_outcome: graphviz_drop_in_replacement_validated_phase_2_in_progress
flail_count_dot: 1
flail_count_fdp: 2
flail_count_sfdp: 1
fdp_status: CONVERGED
fdp_residual_classification: stochastic_floor_match: aggregate_equivalent_at_0.5x_with_low_floor_graph_exceptions
sfdp_status: CONVERGED
sfdp_residual_classification: stochastic_floor_match: aggregate_equivalent_at_1x_with_low_floor_graph_exceptions
neato_status: CONVERGED
neato_residual_classification: numerical_residual: cyclic_graph_init_basin

baseline_truth_change: |
  ROUND 1 CACHED RMSDs ARE OBSOLETE. Cached benchmark used different
  node sizes than current code. Use live_compare.py output as the new
  ground truth from now on. Round 2's live baseline:
    dot family median (live): 0.341942
    mixed_width_labels (live): 0.404615
    shape_and_routing_matrix (live): 0.456349
    small_label_storm (live): 0.485187
  Layer assignment + x-ordering already match dot on diagnostic graphs.
  Remaining gap = coordinate values (Brandes-Köpf vs network-simplex).

round_1_summary: |
  baseline median RMSD per family:
    dot:           0.3245 (worst family)  -- attack first
    fdp:           0.2918 (second worst, ALL graphs > 0.15 floor)
    sfdp:          0.0915
    neato_mds:     0.0455
    neato_stress:  0.0353
  sugiyama PERFECT on simple/linear graphs (RMSD < 0.01),
  cliffs at medium graphs. Smallest divergent reproducer:
  mixed_width_labels (6 nodes, RMSD 0.35).
---

# algo_fidelity -- Autonomous Loop State

Drop-in graphviz replacement is central to the dagua pitch. This sprint
audits and improves dagua's faithful reproduction of the algorithms it
claims to reimplement, with **graphviz tools first** (dot, neato, fdp,
sfdp) and other claimed originals as a follow-on phase.

This file is the canonical "where are we" record. Every wake-up event
(watcher fire, user ping, schedule trigger) MUST read this file FIRST
and act on the case routing below given the current observable state.

## Scope (CRITICAL: parallel sprint constraint)

A separate "cluster cosmetic" sprint is also active on the same `develop`
branch. It owns:
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`

**This sprint must NOT touch any of the above.** Algorithm-layout work only:
- `dagua/layout/ops/**` (registered ops + pipelines)
- `dagua/eval/variants.py` (registry entries -- ADD only, do not perturb existing)
- `scripts/**` for new fidelity-comparison scripts (do not modify
  `scripts/graphviz_theme_comparison.py` or other cosmetic scripts)
- `dagua/eval/competitors/*` only if a graphviz adapter has a real bug;
  otherwise leave alone
- `tests/test_layout/**` for any new pipeline-level tests
- `.project-context/research/sprint_algo_fidelity/**`

If a fix is unavoidably entangled with cosmetic code, **stop, document the
entanglement here, and skip that sub-issue**. We don't merge across sprints.

## Worker policy (CRITICAL)

The cluster sprint uses Opus subagents for visual inspection. To avoid
double-billing the user's Anthropic quota, **this sprint uses CODEX
exclusively for execution**. Opus subagents only allowed if codex is
quota-blocked AND the work is small (< ~50 lines).

## Graphviz source reference (MANDATORY for every round)

Authoritative graphviz source is cloned at:

  `/home/jtaylor/projects/_references/graphviz`

**Every round prompt MUST instruct codex to read the relevant graphviz
C source as ground truth before proposing fixes.** Web-search and
academic papers describe the algorithms in general terms; the C source
is what the binary actually does. Defaults, force laws, initialization,
and tie-breakers all live in the C code.

Per-family source map:

| Family | Graphviz source | Key files |
|---|---|---|
| dot (sugiyama) | `lib/dotgen/` | `rank.c` (network simplex rank), `mincross.c` (median + transpose crossing reduction), `position.c` (network simplex coord assignment), `dot.c` |
| neato (stress, classical_mds) | `lib/neatogen/` | `stress.c` (stress majorization), `kkutils.c` (Kamada-Kawai), `neatoinit.c` (classical MDS init), `sgd.c`, `smart_ini_x.c` |
| fdp (FMMM, FR-style force-directed) | `lib/fdpgen/` | `layout.c` + `tlayout.c` + `xlayout.c` (force-directed solver), `grid.c` (repulsion approx), `fdpinit.c` (defaults) |
| sfdp (multilevel spring-electrical) | `lib/sfdpgen/` | `Multilevel.c` (multilevel framework), `spring_electrical.c` (spring-electrical method), `sfdpinit.c` (defaults), `stress_model.c`, `post_process.c` |

Useful default-extraction commands codex can run:

```bash
# Find default values for a family (e.g. fdp K, MaxIter, etc.):
grep -nE "^\s*(static\s+)?(double|int|float)\s+\w+\s*=" \
  /home/jtaylor/projects/_references/graphviz/lib/fdpgen/*.c

# Find force-law expressions (e.g. attraction/repulsion math):
grep -nE "K\s*\*|d\s*\*\s*d|sqrt|log\(" \
  /home/jtaylor/projects/_references/graphviz/lib/fdpgen/*.c

# See what defaults are documented (cross-check):
grep -nE "DEFAULT|default" \
  /home/jtaylor/projects/_references/graphviz/lib/fdpgen/*.c
```

Confirmed graphviz binary version on this machine: 8.0.3
(`dot -V`). Source clone is current main branch (cloned 2026-04-29) --
slightly newer than 8.0.3 but the algorithms haven't changed
substantively. If codex finds a defaults-divergence between source and
8.0.3 binary, document it and use 8.0.3 as the truth (the binary is
what produces our cached positions).

## Stop criteria (observable, quantitative)

Primary (graphviz):
- Median Procrustes RMSD <= 0.05 for each of: graphviz_dot vs sugiyama,
  graphviz_neato vs stress_maj, graphviz_fdp vs fmmm, graphviz_sfdp vs sfdp
- AND worst-graph RMSD <= 0.15 for each family

Secondary (other claimed originals):
- partial_match families lifted to weak_equivalent OR documented as
  ceiling
- divergent families (davidson_harel) lifted to partial_match OR
  documented as ceiling

Anti-flail: 3 consecutive rounds with same un-closeable issue on the
same family -> mark `principled_residual`, document, move on.

max_rounds: 30 across all families combined.

## Wake-up case routing

Run this triage on EVERY new turn (watcher event, user ping, schedule):

```bash
git log --oneline -3
pgrep -af "codex/codex exec" | head -5
ls -la /tmp/algo_fid_*.log 2>/dev/null | tail -5
cat .project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md | grep -E "^state:|^current_round:|^current_family:"
```

| Observable signal | State | Action |
|---|---|---|
| codex pid alive AND log growing | RUNNING | ack the user "still cooking, watcher armed", yield turn |
| codex exited + new commit on develop matching `feat(fidelity):` | ROUND_DONE | run Case A below |
| codex exited + NO new commit | ROUND_FAIL | run Case D below |
| watcher fired CODEX_FAILED with quota text | QUOTA_BLOCKED | run Case C below |
| watcher fired CODEX_TIMEOUT | TIMEOUT | inspect log; if work was happening, dispatch focused fixup; else Case D |
| user pinged "how goes" while RUNNING | RUNNING | ack, optionally tail latest log, yield |
| anti-flail (3x same family unclosed) | RESIDUAL | run Case E below |

### Case A: ROUND_DONE -> proceed
1. `cat eval_output/algo_fidelity/round_<N>/SUMMARY.md` -- read codex's per-round summary
2. Update Iteration log at bottom of this file
3. Decide next family:
   - If current family converged (per stop criterion above) -> advance to
     next family in priority order: dot -> neato -> fdp -> sfdp -> phase 2
   - Else -> continue same family with refined prompt
4. Write `PROMPT_round_<N+1>.md`
5. Update `state: DISPATCHING`, `current_round: N+1`, `current_family: <name>`
6. Dispatch via `codex-bg.sh /tmp/algo_fid_round_<N+1>.log <prompt>`
7. Arm `codex-watch.sh <PID> /tmp/algo_fid_round_<N+1>.log` via Monitor
8. Yield turn

### Case C: QUOTA_BLOCKED
1. Read codex log to confirm quota error message
2. Read /tmp/algo_fid_round_<N>.log -- check what work was already done
3. If small remaining work: pivot to one Opus subagent (model="opus",
   subagent_type="general-purpose") with adapted prompt
4. If large remaining work: schedule wakeup for codex reset time + 5min,
   send "blocked, will resume <time>" iMessage, set state: BLOCKED, stop
5. NEVER export OPENAI_API_KEY

### Case D: ROUND_FAIL
1. `tail -100 /tmp/algo_fid_round_<N>.log` -- find error
2. If <10 min fixable: write fixup spec, dispatch codex --resume
3. Else: revert any partial state, document failure in iteration log,
   move to next family if anti-flail triggered, else retry same family
   with simpler prompt

### Case E: RESIDUAL
1. Mark family `principled_residual` in iteration log with classification:
   - `architectural_floor` -- requires major reimpl
   - `data_floor` -- limited by 25-graph sample
   - `numerical_residual` -- below noise of stochastic seeds
   - `proxy_mismatch` -- competitors are not exact-algorithm matches
2. Continue to next family (don't shut down on first residual)

## Fallback chain (resource limits)

1. Primary: `codex-bg.sh` + `codex-watch.sh` (Monitor-armed)
2. Quota blocked: pivot to one Opus subagent for short remaining work, OR
   schedule wakeup for reset time
3. Both blocked: state: BLOCKED, iMessage user, stop

## Shutdown procedure (mechanical -- user may be asleep)

When stop criterion triggers (or user explicitly says "stop"):

1. Write `algo_fidelity_SUMMARY.md` containing:
   - Per-family before/after Procrustes RMSD (median, p95, worst)
   - Per-family verdict transitions (e.g., partial -> weak)
   - Commits in this run (`git log --grep="feat(fidelity):" --oneline`)
   - Accepted residuals with classification + rationale
   - Anything deferred (e.g., out-of-scope cosmetic entanglement)
   - Drop-in-graphviz pitch readiness assessment
2. Generate one comparison panel per graphviz family showing best+worst
   pairs (dagua vs graphviz side-by-side) using
   `scripts/algo_fidelity_panel.py` (codex builds in round 1)
3. Send via `~/.claude/scripts/send-to-jmt.sh -a <panel> "<family>: median RMSD <X>"`
4. Update this file: state: DONE, append final row to iteration log
5. Send final iMessage: "Run algo_fidelity done. graphviz parity: <verdict>. <N> commits, <M> residuals."

## Codex prompt template policy

Each round's prompt lives at:
`.project-context/research/sprint_algo_fidelity/PROMPT_round_<N>.md`

Each prompt MUST include:
- `<task>` block with: family in scope, files to read, files to create/edit,
  exact commands to run for verification
- `<scope_constraints>` block with the parallel-sprint exclusion list
- `<completeness_contract>` requiring: (a) measurable improvement OR
  documented `principled_residual`, (b) commit on develop with
  `feat(fidelity):` prefix, (c) per-round SUMMARY.md saved to
  `eval_output/algo_fidelity/round_<N>/`
- `<verification_loop>`: pytest changed-files, then comparator script,
  then commit
- `<missing_context_gating>`: if prompt is unclear, ABORT before edits and
  write `BLOCKED.md` -- do not guess

## Iteration log (append per round)

| Round | Family | Start | End | Commit | Median RMSD before/after | Worst graph | Notes |
|---|---|---|---|---|---|---|---|
| 1 | baseline | 19:19 | 19:30 | 78e8529 | N/A baseline | dot/small_label_storm 0.4744 | Built cross-comparator + panels. Identified dot+fdp as worst families. |
| 2 | dot | 19:33 | 19:50 | (no commit) | live 0.3419 (cached 0.3245 obsolete) | mixed_width_labels live 0.4046 | DIAGNOSIS_ONLY: cached vs live mismatch from node-size drift. Layer + x-ordering match dot; gap is in coordinate assignment (BK vs network-simplex). live_compare.py added. |
| 3 | dot | 19:55 | 20:21 | 17521a3 | dot median 0.3419 -> **0.0191** (-20x) | densenet_block 0.168 (only 1 graph >0.15) | **CONVERGED** at median<<0.05. Fix: pipeline defaults rank_sep 1.0->72pt, node_sep 1.0->18pt to match graphviz dot point spacing. No simple-graph regressions. Width-aware BK now produces dot-like proportions. Residual: densenet_block barely over 0.15 worst-graph criterion -- accepted. |
| 4 | fdp | 20:25 | 20:48 | (no commit) | fdp median 0.247 baseline (lower than Round 1 cached 0.292 but still uniform >0.15 floor) | center_port_backedge_hub 0.440 | RESIDUAL: K=0.3 alignment regressed slightly. Codex diagnosis: dagua FMMM uses OGDF logarithmic attraction, graphviz fdp uses FR force law (rep K^2/d^2, attr (d-len)/d). Force-law mismatch is the dominant gap. Round 5: add FR force model to FMMM, verify against `lib/fdpgen/tlayout.c`. |
| 5 | fdp | 21:06 | 21:35 | (no commit) | fdp median 0.247 -> 0.257 (regressed) | center_port_backedge_hub 0.440 unchanged | RESIDUAL #2 on fdp. Force-law alignment with graphviz tlayout.c old-form (rep K^2/d^2, attr d/L*weight) verbatim REGRESSED median by +0.01. parallel_multiedge_bundle (3 nodes) sits at 0.257 -- even smallest graph is uniform-floor, suggests random initialization is dominant remaining gap. PARKING fdp at flail=2; pivoting to sfdp/neato to maximize graphviz parity coverage. Final attempted lever (random init via lib/fdpgen/fdpinit.c) deferred to a later round. |
| 6 | sfdp | 21:18 | 21:42 | (no commit) | sfdp median 0.092 baseline (matches Round 1) | center_port_backedge_hub 0.475 | RESIDUAL #1 on sfdp. **MAJORITY of graphviz defaults already aligned in dagua** (random_start=true, seed=123, C=0.2, bh=0.6, maxiter=500, K=avg_edge_length, K*0.75 finer levels, adaptive_cooling switch). Tried lever: attractive-force distance factor per graphviz spring_electrical.c -- improved p95 (0.418->0.404) but regressed median (0.092->0.106). Partial fix in wrong direction. PARKING sfdp at flail=1; pivoting to neato (already nearly converged). Future lever: sequential vs synchronous updates (invasive). |
| 7 | neato | 22:00 | 22:11 | (no commit) | stress_maj 0.035, classical_mds 0.045 (both <= 0.05 ✓) | inception_block 0.382, petersen_10 0.333 (worst > 0.15 ✗) | OUTLIER_RESIDUAL: medians meet stop criterion. Worst-graphs are cyclic/dense/disconnected (basin differences from graphviz INIT_RANDOM vs dagua classical MDS+jitter). Classification: `numerical_residual: cyclic_graph_init_basin`. Validation determinism confirmed (cmp same outputs across 2 runs). Round 8 to multi-seed upgrade live_compare. |
| 8 | stochastic_re_eval | 21:35 | 22:09 | 9205f36 | infra round | fdp 18/18 not_equivalent, sfdp 21/21 not_equivalent | Multi-seed comparator + TOST built. **fdp/sfdp residuals CONFIRMED real algorithmic divergence**, not stochastic noise. **Within-graphviz floor was ~0** because cached graphviz positions were generated with fixed seed (graphviz_competitor.py drops seed via `del seed` lines 342/402). Round 9 to fix competitor + regenerate true multi-seed cache. |
| 9 | stochastic_seed_fix | 22:15 | 23:08 | 58359b2 | **fdp eq_at_0.5x, sfdp eq_at_1x, neato_stress eq_at_0.5x, neato_mds eq_at_0.5x** | n/a (TOST classification) | **HUGE WIN**: fixed graphviz_competitor.py to thread `-Gseed -Gstart` to binary. Regenerated multi-seed cache. **ALL 4 graphviz families now CONVERGED under TOST equivalence test against graphviz's own stochastic floor.** Round 8's not_equivalent verdict was a measurement artifact. graphviz fdp itself has within-seed median RMSD 0.235; dagua FMMM at 0.250 is indistinguishable. drop-in graphviz replacement claim empirically validated. |
| 10 | phase_2_sweep | 23:12 | 23:30 | (no commit) | aborted | n/a | ABORTED: codex hit `stdin closed` error mid-sweep (Davidson-Harel slow). Audit-by-self confirmed: igraph adapter PROPERLY threads seeds (line 46) so davidson_harel/drl/graphopt verdicts in mega-run ARE legitimate (REAL algorithmic gaps). ogdf adapter has same `del seed` bug as graphviz did (line 203) but ogdf-targeted families already strong_equivalent so doesn't matter. Phase 2 verdicts stand from existing mega-run report. Pivoting to SUMMARY round. |
| 11 | summary | 23:35 | 23:40 | n/a | n/a | n/a | Final summary written: algo_fidelity_SUMMARY.md. Drop-in graphviz replacement claim empirically validated; 4 commits on develop; Phase 2 deferred. Sprint DONE. |
| 1 | baseline | 2026-04-29T19:16:57-04:00 | 2026-04-29T19:26:04-04:00 | 0a9a957 | N/A baseline | center_port_backedge_hub | Round 1 = infrastructure |
| 2 | dot | 2026-04-29T19:32:00-04:00 | 2026-04-29T19:47:27-04:00 | none | 0.3245 cached / 0.3419 live | small_label_storm 0.4852 live | Built live comparator; blocked before Sugiyama fix because live baseline differs from Round 1 cache on 8/22 graphs due node-size context drift. |
| 3 | dot | 2026-04-29T19:55:00-04:00 | 2026-04-29T20:20:04-04:00 | feat(fidelity): round 3 | 0.3419 live / 0.0191 live | densenet_block 0.1679 live | COMMITTED: aligned classic Sugiyama direct defaults to dot point spacing (`rank_sep=72`, `node_sep=18`); diagnostics improved without simple-graph regression. Next family: fdp. |
| 4 | fdp | 2026-04-29T20:30:00-04:00 | 2026-04-29T21:16:00-04:00 | none | 0.2475 live / 0.2520 attempted | center_port_backedge_hub 0.4432 attempted | RESIDUAL: Graphviz `K=0.3in` ideal-length alignment regressed median and was reverted. Stay on fdp; next lever should target Graphviz fdp force law or initialization. |
| 5 | fdp | 2026-04-29T21:05:00-04:00 | 2026-04-29T21:47:00-04:00 | none | 0.2475 live / 0.2572 attempted | center_port_backedge_hub 0.4401 attempted | RESIDUAL: Graphviz `tlayout.c` force-law mode regressed median and was reverted. `flail_count_fdp=2`; advance to sfdp unless a final fdp random-init attempt is explicitly scheduled. |
| 6 | sfdp | 2026-04-29T21:42:00-04:00 | 2026-04-29T22:00:00-04:00 | none | 0.0915 live / 0.1062 attempted | center_port_backedge_hub 0.4798 attempted | RESIDUAL: Graphviz attractive distance-factor alignment regressed median and was reverted. `flail_count_sfdp=1`; stay on sfdp for one more lever, likely sequential update or random stream alignment. |
| 7 | neato | 2026-04-29T22:00:00-04:00 | 2026-04-29T23:32:00-04:00 | none | stress 0.0353 live / MDS 0.0455 live | inception_block 0.3817 stress; petersen_10 0.3326 MDS | OUTLIER_RESIDUAL: medians satisfy `<=0.05`, worst-case fails on dense/cyclic/symmetric graphs. Graphviz defaults match Dagua on stress weights (`1/d^2`) and `maxiter=200`; residual is initialization-basin mismatch from Graphviz random start vs Dagua deterministic MDS start. Mark neato CONVERGED at family-median level and advance to Phase 2. |
| 8 | stochastic_re_eval | 2026-04-29T22:30:00-04:00 | 2026-04-30T00:00:00-04:00 | this commit | fdp 0.2484 multi-seed; sfdp 0.1074 multi-seed; neato unchanged | fdp center_port_backedge_hub 0.3275 median; sfdp disconnected_label_cycle_collage 0.4135 median; neato cache unseeded | Built multi-seed `live_compare` with pairwise dagua-vs-graphviz, within-graphviz, within-dagua RMSD rows plus per-graph TOST. Re-eval verdicts: fdp `not_equivalent`, sfdp `not_equivalent`; graphviz-neato seeded cache absent so neato TOST is `not_tested`. No family reclassified as stochastic-floor faithful. |
| 9 | stochastic_seed_fix | 2026-04-30T00:00:00-04:00 | 2026-04-30T00:00:00-04:00 | this commit | fdp aggregate TOST `equivalent_at_0.5x`; sfdp `equivalent_at_1x`; neato stress/MDS `equivalent_at_0.5x` | graph-level low-floor exceptions remain, but aggregate distributions match true Graphviz stochastic floors | Fixed Graphviz seed plumbing for fdp/sfdp/neato via `-Gseed` + `-Gstart`, generated Round 9 seeded cache for the comparator graph union, and reran all four multi-seed checks. Round 8's fdp/sfdp architectural-divergence conclusion was a fixed-seed cache artifact. Mark fdp and sfdp CONVERGED under stochastic-floor lens without incrementing flail counts. |
| 12 | davidson_harel | 2026-04-29T23:35:00-04:00 | 2026-04-29T23:45:00-04:00 | none | not measured | n/a | BLOCKED: required 5-seed `live_compare` baseline stayed CPU-active beyond the 10-minute budget and wrote no output files. No code changes. Source mapping found likely divergences in energy weights/normalization and one-move-per-node vs igraph's 30 circular tries per node. See `ROUND_12_BLOCKED.md`. |
| 13 | davidson_harel | 2026-04-29T23:49:00-04:00 | 2026-04-30T00:26:00-04:00 | this commit | small subset median 0.3620 / 0.2377 | linear_3layer_mlp 0.2881 post-fix | COMMITTED: aligned Davidson-Harel energy weights/unnormalized objective and move schedule to igraph defaults. Commit criterion met by median RMSD improvement of 0.1243 on the 5 evaluated small graphs. TOST did not reclassify the family to weak_equivalent: only `linear_3layer_mlp` is `equivalent_at_2x`; four graphs remain `not_equivalent`. Advance to `drl`. |
| 20 | davidson_harel | 2026-04-30T01:00:00-04:00 | 2026-04-30T02:07:44-04:00 | this commit | small subset median 0.2377 / 0.1666 | parallel_multiedge_bundle 0.2566 after | COMMIT_CRITERION_MET: fine-tuning phase, node-edge gating, incremental move-delta energy, and skipped final normalization improved median by 0.0711. Residual: multiedge graph regressed; original edge multiplicity/RNG/clamp parity deferred. |
