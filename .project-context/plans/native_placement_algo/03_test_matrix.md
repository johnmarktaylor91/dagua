# Test Matrix

Three disjoint graph sets plus a rolling-generator protocol. Every sprint
respects these roles.

## Roles

| Role | Used for | Regenerated? |
|------|----------|--------------|
| Iteration suite | Daily iteration, iterate_native.sh, sweep | Fixed |
| Held-out suite | Sprint-exit validation only | Fixed |
| Rolling set | Extra anti-overfit signal at sprint exit | Regen per sprint |

## Iteration suite (fixed, ~25 graphs)

Small enough to fit a <60-second loop; large enough to cover the 14 structural
tags from eval/graphs.py. Stored in `dagua/graphs/iteration/`. Per-graph budget:
size <=500 nodes so total suite runs <= 30 seconds in aggregate.

- `chain_10`, `chain_100`, `chain_500`
- `binary_tree_127` (depth 7 complete tree)
- `wide_parallel_50`, `wide_parallel_200` (fan-out topology)
- `skip_light_30`, `skip_heavy_60` (residual + densenet-style)
- `diamond_40`, `diamond_nested_100` (U-shape and nested)
- `cluster_shallow_80`, `cluster_deep_180` (2-level and 5-level nested)
- `mixed_width_40` (labels from "ReLU" to very long)
- `self_loops_20`
- `multi_edge_30`
- `grid_10x10`, `grid_20x10` (lattice DAGs)
- `random_dag_200` (density 2.5)
- `near_clique_20` (density 5)
- `disconnected_3x40`
- `long_chain_1000` (stress init)
- `star_40` (one hub, 40 leaves)
- `undirected_karate_club`
- `undirected_les_mis`

The exact list is frozen at Sprint 0 exit. Additions to this suite must be
approved by the user and bump `iteration_suite_version`.

## Held-out suite (fixed, at least 30 graphs -- OPAQUE, NEVER iterated against)

Revised per adversarial review 2026-04-22: bumped from 15 to a minimum of
30 graphs (preferred 42) with >=2 representatives per priority family and
enforced small/medium split per family. One sample per tag is insufficient.

Held-out opacity: topology and adjacency hashes are committed; the actual
graphs are generated at Sprint 0 from a SECRET SALT stored at
`.project-context/private/holdout_salt` (gitignored), combined with a public
seed string. This prevents the "clever engineer inspects held-out and tunes
for it" failure mode flagged in adversarial review.

Storage layout:
```
dagua/graphs/holdout/MANIFEST.json         # tag, family, size, topology_hash
.project-context/private/holdout_salt      # 32-byte secret, gitignored
dagua/graphs/holdout/.opaque               # marker file; forbid direct open
```

Inspection policy: a sprint-exit run of `dagua.eval.benchmark` with
`--mode=holdout` regenerates graphs from salt + seed, runs metrics, and
deletes graph tensors from memory. No persisted on-disk unsealed graph files.
If an engineer opens the held-out directory directly, it fails validation.

Sprints 1-8 are FORBIDDEN to inspect held-out topologies. Sprint 9 exit may
inspect to debug regressions. A separate "post-release audit suite" is
generated from a second salt that Claude does not have access to during
plan execution; user provides it after Sprint 9.

Held-out-large (100K+) is in a separate manifest since its generation is
slow; same opacity rules apply.

## Rolling set (regenerated per sprint, salt-derived)

Revised per adversarial review: seed is NOT `sha256(public_tag)[:8]`, which
is predictable. Seed is now `sha256(secret_salt || sprint_tag)[:8]` using
the same salt as held-out. Set size: 10 graphs drawn from the 14-tag
generator pool, disjoint from held-out (enforced by topology hash check).

`dagua/eval/benchmark.py` gains `--seed-strategy={fixed,rolling,holdout}`,
`--sprint-tag=<string>`, and `--salt-path=<path>` CLI args in Sprint 0.
Missing salt file -> run fails; rolling/holdout mode is salt-required.

## Generator overhead (new, from adversarial review)

The rolling-set generator and graph_generator.py module do NOT exist today.
Sprint 0 Task 0.7.1 measures generation overhead per graph tier; if >15%
of an 8-minute sprint-exit run, generation is precomputed at sprint tag
emission rather than on demand.

## Graph families (priority order)

| Priority | Family | Examples |
|----------|--------|----------|
| P0 | Directed DAG | ResNet, chain, diamond, random DAG |
| P0 | Tree | Binary tree, decoder tree |
| P0 | Nested cluster | Transformer, ViT, U-Net with labels |
| P1 | Undirected sparse | Les Mis, social graph, grid lattice |
| P1 | Near-clique | Dense subgraph, small complete graph |
| P1 | Disconnected | 3 chains side-by-side, multi-component |
| P2 | Self-loops | RNN unrolled, feedback |
| P2 | Multi-edge | Multi-arg op, parallel edges |
| P3 | Pathological | Star, long chain, 50-branch diamond |

P0 must improve at every sprint exit; P1 must not regress; P2 and P3 must
not crash.

## Scale ladder

| Tier | Nodes | Priority | Notes |
|------|-------|----------|-------|
| Micro | 10 | P0 | Sanity only, metrics meaningless |
| Small | 100 | P0 | Default iteration size |
| Medium | 1000 | P0 | Sprint exit gate |
| Large | 10K | P0 | Sprint exit gate |
| Huge | 100K | P1 | Sprint 2 exit gate |
| Mega | 1M | P1 | Sprint 8 exit gate |
| Ultra | 10M | P2 | Sprint 8 exit gate |
| Beyond | 100M+ | deferred | See 09 Q7 |

Runtime budgets per tier (target for Sprint 9 exit):

| Tier | Default budget | Stretch |
|------|----------------|---------|
| Micro | <=0.5 s | <=0.2 s |
| Small | <=2 s | <=1 s |
| Medium | <=10 s | <=5 s |
| Large | <=60 s | <=30 s |
| Huge | <=6 min | <=3 min |
| Mega | <=8 min | <=4 min (GPU) |
| Ultra | <=45 min | <=30 min (GPU) |

Memory budget: peak RSS <=120 GB at Ultra tier. Autograd 3-4x multiplier
accounted for per scaling principles doc.

**Caveat (from adversarial review):** the ops pipeline today uses
`LossGroup(backward_mode="combined")` and lacks per-loss backward,
checkpointing, and hybrid device fallback that live in legacy `_layout_inner`.
Sprint 1 ports those memory features; until it does, Mega and Ultra budgets
are STRETCH targets, not exit criteria. See 02_sprint_map.md Sprint 1 exit.

## Scale validation protocol (revised)

The "3 runs per tier" rule in the draft only measures repeatability on a
fixed topology. Adversarial review: that is not variance coverage. Revised:

At each exit tier, run a `topology cross-product`:
- Sizes: Huge (100K), Mega (1M), Ultra (10M)
- Topology classes: sparse-wide, sparse-deep, higher-E/N (dense)
- Seeds: 3 per cell (repeatability)
- Total: 3 sizes x 3 classes x 3 seeds = 27 runs at Sprint 8 exit.

For Sprints 2 and 8, the exit gate requires: no OOM on any cell, no composite
regression >5% on any cell vs prior sprint (per-class), and cell-level
runtime within 1.5x the per-tier budget.

CPU fallback envelope (24 GB host): documented as sparse-wide only up to Huge
(100K). Anything denser or larger requires GPU. This replaces the vague "CPU
fallback documented" language from the draft.

## Non-regression bars (per aesthetic metric, revised)

Per-sprint tolerance in the original draft accumulated to ~-27% across nine
sprints. Adversarial review flagged this as not a real non-regression bar.
Revised per-sprint + cumulative bars below use CODE field names from
`dagua.metrics.composite`:

Per-sprint bar (vs prior sprint exit held-out):
| Metric | Tolerance (relative) |
|--------|----------------------|
| overlap_count | 0 (hard) |
| crossing_rate | +3% |
| dag_consistency | -2% |
| edge_node_crossing_rate | +5% |
| edge_length_cv | +10% |
| angular_res_mean_deg | -5% |
| cluster_mean_sep_ratio | -10% |
| composite | -3% per sprint |

Cumulative bar (vs Sprint 0 baseline):
| Metric | Cumulative floor |
|--------|------------------|
| composite | -5% absolute (no free waiver) |
| per-family composite | see family veto table in 04_evaluation_rubric.md |

Breaching per-sprint tolerance is a standard exit block. Breaching cumulative
bar requires an explicit user waiver on iMessage before exit; Claude does NOT
silently approve.

Tighter bars apply on P0 families. If a sprint trades quality for speed, the
trade is explicitly declared in the sprint exit note; the user confirms.

Family veto bars for P2/P3 (near_clique, disconnected, pathological) live in
04_evaluation_rubric.md. These prevent weighted-mean washout of hard cases.

## Mid-sprint vs exit-sprint

Mid-sprint: run the iteration suite only. Do NOT touch held-out. Rolling set
not required for quick feedback.

Sprint exit: full chain. Iteration suite + held-out + rolling set. All three
must pass the non-regression bars. Adversarial Codex review must PASS.

## Generator harness (for rolling + held-out)

New module: `dagua/eval/graph_generator.py`. Pure Python, stdlib + torch only.
API:
```
make_suite(seed_bytes: bytes, size: int = 10, families: list[str] = None) -> list[TestGraph]
```
Samples topologies from each requested family with sizes drawn per the scale
ladder weights. Registered in Sprint 0. Deterministic for a given seed.

## Competitor benchmark at every sprint exit (new)

Iteration + held-out evaluations are head-to-head vs the FULL 16-variant
authoritative matrix in 11_competitor_weaving.md (single source of truth).

Competitor results are cached; refresh at Sprint 0.5, Sprint 5, Sprint 9
plus any hash-change trigger per 11's refresh protocol. Running Dagua
head-to-head costs only Dagua's wall-time.

Per-sprint Pareto gates (from 10_iteration_loop.md):

| Sprint | Pareto-optimal share required on iteration suite |
|--------|--------------------------------------------------|
| 1 | 20% |
| 2 | 30% |
| 3 | 40% |
| 4 | 50% |
| 5 | 55% |
| 6 | 65% |
| 7 | 70% |
| 8 | 80% |
| 9 | 90% (plus >=80% on held-out) |

Pareto-optimal means no competitor beats Dagua on both (composite, runtime)
axes on that graph.

## Mandatory records at sprint exit

Commit under `eval_output/native_algo/sprint_<N>_exit/`:
- `metrics.json` (all three sets, all metrics)
- `adversarial_review.json` (findings, verdict)
- `visual_audit/` (one image per held-out graph + key iteration graphs)
- `regression_notes.md` (any tolerated regressions + reason)
- **New** `pareto_vs_competitors.json` (per-graph classification + deltas)
- **New** `iteration_log.jsonl` (full within-sprint log; see 10)
- **New** `extractions_log.md` (which competitor techniques landed this
  sprint, which failed to land, with one-line reason each; see 11)
