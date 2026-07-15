# r74 Cluster O5 -- INSUFFICIENT_DATA Triage (257 combos)

Source: `eval_output/fidelity_definitive_r73/per_combo.json` (3,955 combos; 257 insufficient).
Benchmark rows: `eval_output/benchmark_100seed_escalation_final/results.json` (default analysis dir).
READ-ONLY survey. All counts verified against the data fields named in the brief.

## Headline reframe (vs thread-6 / MEMORY)

Two prior claims are REFUTED by the actual benchmark `error` strings:

1. **"davidson/stress/neato/maxent are CRASH/BUG (divide-by-zero, empty subgraph)"** -- FALSE.
   Every one of their 257-cluster failures carries `error == "watchdog: future exceeded timeout"`
   or `"worker layout timeout exceeded"`. There are ZERO exception/crash strings for these
   engines. Thread-6 inferred "crash" from "0 reimpl seeds on small graphs" WITHOUT reading the
   error field. They are TIMEOUTS (slow pure-Python pipelines), not bugs.
2. **"sgd2_multi_with_crossing 81 = no reference by design / STRUCTURAL_NA"** -- FALSE.
   A reference exists and ran: `sgd2_multi_ref__for__classic_sgd2_multi_with_crossing` has
   3,692 OK rows in results.json. The 4 sibling variants (default/batch8/with_aspect/stress_only)
   are ALL rung-1 bit-exact (332 combos). The crossing variant is INSUFFICIENT only because its
   benchmark run was never completed: merged store shows ref `status_running: 5500` (rows stuck
   mid-flight) + 2,406 errors, and reimpl `status_error: 6164 + status_skipped: 1928`. This is a
   compute/provenance gap, NOT a structural absence.

Also flag: the brief suggests using the `disconnected`/`degenerate` flags to classify crashes.
**These flags are None for ALL 257 insufficient combos** -- they are computed only AFTER mode
classification (analyze_payload line 767, after the early return at 765). They carry no signal
for this cluster. Crash/timeout discrimination must come from the benchmark `error` strings.

## Complete error taxonomy across the 257 (reimpl-side, results.json)

| error string | count | meaning |
|---|---|---|
| `watchdog: future exceeded timeout` | 12,168 | worker future never returned within 600s |
| `BrokenProcessPool ... terminated abruptly` | 1,000 | REAL CRASH (worker killed) -- all `classic_umap_nn30` |
| `worker layout timeout exceeded` | 210 | scaled per-job timeout exceeded |
| `maximum recursion depth exceeded` | 3 | REAL BUG -- `sugiyama_graphviz_fidelity / small_world_2000` |

So: ~99% timeouts; 1,000 umap_nn30 crashes; 3 sugiyama recursion crashes.

Timeout config (`scripts/run_benchmark.py`): DEFAULT_TIMEOUT=120s scaled by graph size
(MIN 30s for <500 nodes, full at 500); WATCHDOG_TIMEOUT=600s per future;
CONSECUTIVE_FAILURE_SKIP_THRESHOLD=3 (3 consecutive timeouts -> remaining seeds `skipped`,
which is why one slow graph zeroes out all 100 seeds = `status_skipped: "skipped after 3
consecutive errors"`).

---

## SUBGROUP A -- RECOVERABLE-BY-COMPUTE (166 combos)

Reasons: matched_seeds_lt_30 (146) + reimpl_seeds_lt_30 (13) + ref_seeds_lt_30 (7).
In matched_seeds_lt_30, reference is healthy in 145/146 (`n_ref_seeded_ok=100`); the gap is
reimpl-side (`n_reimpl_ok=0` in 108/146). All failures are timeouts.

Split by graph size (the actionable distinction):

### A1 -- SMALL-GRAPH PERF BUG (82 combos, N<=300) -- HIGH ROI, this is the prize
These engines time out on graphs a fast layout should finish in <1s. Proof:
`davidson_harel_rounds50` takes **9.0s for a 42-node graph** and times out completely above
~42 nodes (regular_4_40, sierpinski_42 at the cliff). `drl_coarsest` times out on
`regular_3_30` (30 nodes). This is super-linear cost in pure-Python scalar loops, NOT a
legitimate big-graph budget.

| engine | A1 combos | smallest failing graph |
|---|---|---|
| classic_drl_coarsest | 23 | regular_3_30 (30) |
| classic_drl_default | 17 | sierpinski_42 (42) |
| classic_davidson_harel_rounds50 | 10 | triangular_lattice_36 (36) |
| classic_drl_coarsen | 9 | wide_single_layer_1_50_1 (52) |
| classic_neato | 7 | real_lesmis_77 (77) |
| classic_stress_maj_iter500 | 6 | real_lesmis_77 (77) |
| classic_maxent_stress_{alpha2,default,entropy} | 6 | sbm_5x50 (250) |
| classic_fmmm_graphviz_fdp_fidelity | 2 | sbm_5x50 (250) |
| classic_davidson_harel_rounds100 | 1 | sparse_pair_50 (50) |
| classic_pivot_mds_50 | 1 | heavy_tail_weights_50 (50) |

Root cause confirmed for davidson_harel: `dagua/layout/ops/pipelines/davidson_harel.py` runs
nested Python loops `for round in range(rounds+fineiter) -> for node in permutation -> for
move_try in _MOVE_TRIES` with per-node `float()` scalar conversions (lines ~352-390). drl
(`drl.py`) similarly runs 750 force-iterations (200+200+200+50+100 across coarsening levels) in
Python. These are source-faithful ports of C engines, hence slow.

**Recovery action (per engine, choose one):**
- BEST: vectorize the inner energy-delta / force loop in torch so cost stops being O(rounds *
  nodes * tries) of Python scalar ops. Gets these <1s, all 82 recover, fidelity unchanged (the
  algorithm is identical; only the host-language loop changes). Effort: medium per pipeline,
  ~1-2 engines per Codex run. drl + davidson_harel + stress_maj cover 56 of 82.
- CHEAP STOPGAP: raise per-job timeout (e.g. `--timeout 1200 --watchdog-timeout 7200`) and
  re-benchmark ONLY these engine x graph pairs. Recovers the N<=120 graphs at ~9s->minutes, but
  N=200-300 graphs at this scaling cost ~hours/seed -> NOT viable for all 82. Use only to
  unblock the N<=80 subset while the perf fix lands.
- Compute cost of a targeted re-benchmark of just A1: 82 combos x 100 seeds = 8,200 layouts; at
  the (post-fix) ~1s each = ~2.3 CPU-hours; pre-fix it does not finish.

### A2 -- LARGE-GRAPH timeouts (84 combos, N>300) -- MIXED, mostly legitimate floor
These are big graphs where the cost may be real. Engines:
fr_steps500 (7, +the 7 ref_seeds combos), fmmm_graphviz_fdp_fidelity (7), sfdp_* (29 across
theta08/default/graphviz/p_neg2/theta04/steps200), gem_iters100 (6, on ba_5000/grid_50x50/
powerlaw_2000/rgg_2000 -- genuinely O(N^2)/iter, FLOOR), drl_refine/coarsest/default large (12),
sugiyama_* large (8, ba_5000/rgg_2000/sbm_8x100/small_world_2000), maxent/stress_maj (4),
classical_mds (2, er_2000), neato (2).

**Recovery action:**
- gem on N>=2000, fr_steps500 on N>=2000, sfdp on powerlaw_2000/rgg_2000: legitimately
  compute-heavy. These should be RELABELED `COMPUTE_FRONTIER_NA` (not INSUFFICIENT) with the
  measured per-seed runtime as evidence -- or recovered only with a long-wall-clock dedicated
  re-benchmark (`--watchdog-timeout 7200`, few workers). Low fidelity yield; do last.
- The N=300-500 band (fmmm_fdp, sfdp on 500-node, sugiyama on small_world_500) likely recovers
  with a modest timeout bump + the same perf work as A1 (drl/fmmm). Medium ROI.

### ref_seeds_lt_30 (7, all classic_fr_steps500) -- the REFERENCE times out
For fr_steps500 the reimpl mostly succeeds (n_reimpl_ok 10-67) but the REFERENCE
`nx_spring__for__classic_fr_steps500` times out (`drop_ref status_error: 91`, `watchdog`
on ba_2000 etc.). networkx spring_layout at 500 iterations on 2000-node graphs is slow.
Recovery: re-run the fr_steps500 REFERENCE with longer wall-clock, OR accept as compute floor.
sbm_8x100 is the near-miss (n_reimpl_ok=67, n_ref_seeded_ok=25 -- just 5 ref seeds short of 30).

---

## SUBGROUP B -- MISSING-REFERENCE (91 combos) -- reframed

### B1 -- sgd2_multi_with_crossing (81) -- NOT structural; INCOMPLETE BENCHMARK -- HIGH ROI
Reason `no_reference_rows`, but a reference exists (3,692 OK rows). The crossing-min loss is
O(E^2), so both ref and reimpl runs were abandoned (merged store: ref `status_running: 5500`,
reimpl `status_error: 6164`). n_reimpl_ok is 0 for 75/81, 1-3 for 6/81.
**The 4 sibling sgd2_multi variants are 100% rung-1 bit-exact (332 combos).** The crossing
variant shares the same RNG path, so it will very likely score rung-1/2 once it completes.
**Recovery: re-benchmark `classic_sgd2_multi_with_crossing` + its reference to completion** with
adequate per-seed wall-clock (crossing loss is the slow part; consider fewer seeds, e.g. 30, to
hit the MIN_MODE_SEEDS=30 threshold cheaply). Effort: low (no code change, just a scoped
benchmark run). This is the single biggest count-recovery in the cluster (81 -> measurable).
Quick win, but compute-gated by the O(E^2) crossing cost on big graphs (ba_500, citation_dag_300).

### B2 -- umap_nn30 (10) -- DOUBLE-SIDED CRASH (belongs in C) -- QUICK-WIN code fix
Reason `no_reference_rows`, but the real cause is a CRASH on BOTH sides:
reimpl `classic_umap_nn30` AND reference `umap_graph__for__classic_umap_nn30` both die with
`BrokenProcessPool` (1,000 + 1,000). The 10 graphs include weighted_chain_20 (N=20 < 30!),
regular_3_30 (N=30), weighted_clusters_3x10 (N=30), weighted_karate_34 (N=34).
**Bug class: n_neighbors(=30) >= n_samples** in kNN graph construction (umap/pynndescent
crashes when nn >= N). random_dag_200 (N=383) also crashes -> possible secondary
pynndescent/memory issue, but the small-graph pattern dominates.
**Recovery: clamp n_neighbors to min(30, N-1) in BOTH the reimpl pipeline
(`dagua/layout/ops/pipelines/umap.py`) and the reference adapter (umap competitor).** Effort:
low (2-3 line guard each side) + re-benchmark these 10. High ROI quick win.

---

## SUBGROUP C -- CRASH-BUGS (13 reimpl_seeds_lt_30 + the umap_nn30 crashes from B2)

### C1 -- sugiyama on big graphs (11 of the 13) -- TIMEOUT not crash (mostly)
sugiyama_{default,passes4,passes48,tight,wide,graphviz_fidelity} on ba_5000/rgg_2000/sbm_8x100/
small_world_2000/small_world_500. Reference is DETERMINISTIC (`has_ref_deterministic=True`,
`n_ref_seeded_ok=0`) -> Mode B, needs >=30 reimpl seeds. Reimpl times out (`watchdog`) so
n_reimpl_ok ranges 0-29. Several are near-miss: passes48/sbm_8x100=25, tight/ba_5000=28,
default/ba_5000=29, wide/ba_5000=29 -- just 1-5 seeds short.
**Recovery: longer wall-clock re-benchmark of sugiyama on these few big graphs** (the per-pass
ordering loop is slow at N=5000). Near-misses recover trivially.
EXCEPTION: `sugiyama_graphviz_fidelity / small_world_2000` shows 3 `maximum recursion depth
exceeded` -- a REAL stack-overflow bug (recursive cycle-removal/DFS at N=2000). **Fix: convert
the recursive sugiyama cycle-break/ranking to iterative, or raise sys.setrecursionlimit.**
Quick code fix.

### C2 -- classical_mds (2) -- TIMEOUT
classical_mds_{default,igraph_fidelity} on er_2000 (`watchdog`, n_reimpl_ok=0). Deterministic
ref. Full eigendecomposition / double-centering on 2000-node dense distance matrix is slow.
Recovery: longer timeout re-run, or accept as compute floor (N=2000 dense MDS).

### C3 -- pivot_mds_50 / heavy_tail_weights_50 (1) -- 5 reimpl OK, ref deterministic
n_reimpl_ok=5, 95 status_error in merged store (no error string in escalation_final dir alone ->
the errors came from a different overlay dir). Needs the merged-store error inspected; likely a
weighted-graph timeout. Low priority (single combo).

---

## ROI-ORDERED RECOVERY PLAN

| # | action | combos recovered | effort | compute | quick win? |
|---|---|---|---|---|---|
| 1 | umap_nn30: clamp n_neighbors=min(30,N-1) both sides + rerun 10 | 10 | low (code) | tiny | YES |
| 2 | sgd2_multi_with_crossing: rerun ref+reimpl to completion (30 seeds ok) | 81 | low (rerun) | medium (O(E^2)) | YES (count) |
| 3 | sugiyama recursion fix (iterative cycle-break) + rerun small_world_2000 | ~1-3 | low (code) | tiny | YES |
| 4 | perf-vectorize davidson_harel + stress_maj + drl inner loops; rerun A1 | ~56-82 | med (per pipeline) | low after fix | partial |
| 5 | sugiyama/classical_mds/fr-ref big-graph: `--watchdog-timeout 7200` rerun | ~20 | low (rerun) | high | no |
| 6 | gem/sfdp/fr N>=2000: RELABEL COMPUTE_FRONTIER_NA w/ runtime evidence | ~30 | low (label) | none | label-only |

Quick wins (1-3): ~92-94 combos recoverable with small code fixes + scoped reruns, almost no
compute. The perf-vectorization (4) is the big structural lever (~56-82) but needs real
pipeline work. The large-graph tail (5-6) is low-yield; relabel rather than recover.

## Thread-6 claims I could NOT verify / REFUTED
- REFUTED: davidson/stress/neato/maxent are "crash/bug (divide-by-zero, empty subgraph)" --
  they are all timeouts (no exception strings exist).
- REFUTED: sgd2_multi_with_crossing is "structural / no reference by design" -- reference exists
  (3,692 OK rows); the variant's run was never completed (5,500 rows still `status_running`).
- PARTIALLY CONFIRMED: thread-6's DRL "timeout not crash" + gem "O(N^2) compute frontier" are
  correct. Its umap_nn30 "add reference variant" framing is wrong -- the reference EXISTS but
  crashes (BrokenProcessPool); the fix is an n_neighbors clamp, not a new adapter.
- COULD NOT verify thread-6's exact combo counts (it cited 60/12/12/8/8/11 per engine vs my
  verified per-reason data) -- my numbers come straight from per_combo.json reason fields and
  supersede the thread-6 estimates.
- NOTE: per_combo `status_running`/`status_error` reflect the MERGED multi-dir store
  (load_results_multi, last-dir-wins); escalation_final ALONE shows some of these as OK (e.g.
  ba_500 sgd2-crossing 100 OK), so the final verdict depends on which overlay dir clobbered
  them. The exact r73 --data-dir list was not located in the state files; this provenance gap
  is worth confirming before the sgd2-crossing rerun.
