# Native Placement Algorithm -- Overview

Status: DRAFT. Awaiting user approval before Sprint 0.
Branch plan: one feature branch at a time. Sprint 0 on a new `feat/native-algo-sprint-0`.
Generated: 2026-04-22.

## Vision

`dagua.layout(g)` with no algorithm arg should produce **the best default
layout of any open-source graph layout library, across a wide graph variety
and 10 -> 10M+ node scale, competitive or better than the authoritative
16-competitor set frozen in 11_competitor_weaving.md (Graphviz dot/sfdp/
neato/fdp, ELK, dagre, igraph sugiyama/fr/kk, NetworkX spring/kk,
sgd2_multi_ref, gephi_yifanhu, fa2_ref, ogdf_fmmm, cytoscape_fcose) on
BOTH quality and runtime. The full 16-variant list with adapter modules
is in 11_competitor_weaving.md; that file is the single authority.**

Method: modern optimization as the default toolkit (autograd, differentiable
loss, GPU vectorization), with clean hybrid classical steps where they
measurably win. Every best idea from every competitor is a candidate for
extraction into a registered op; see 11_competitor_weaving.md for per-sprint
extraction targets. The engine stays composable: every step is a registered
op, and the top-level pipeline has zero inline helpers.

The work is organized as an iterative quality+runtime improvement loop (see
10_iteration_loop.md): each sprint picks the graphs where Dagua is weakest
vs competitors, forms a hypothesis, implements, measures delta on both axes,
and either keeps the change or reverts. Success is measured as Pareto-optimal
share on the iteration and held-out suites, with explicit gates per sprint.

## Success Criteria (binary, revised post-round-2 review)

1. `dagua.layout(g)` with no kwargs uses a single op pipeline. The legacy
   `_layout_inner` path is deleted or demoted to frozen reference.
2. **Competitive-or-better on quality**: on the held-out set, Dagua is
   Pareto-optimal (quality, runtime) vs the 16 authoritative competitor
   variants named in 11_competitor_weaving.md on >=80% of graphs.
3. **Family floors met at Sprint 9** (per 10_iteration_loop.md revised
   per-family floors): directed DAG large/medium >=80%, nested cluster
   >=85%, undirected sparse >=85%, small DAG/tree >=50% (parity), etc.
   No single family caps the program.
4. **Competitive-or-better on runtime**: same-device comparison only;
   per-family runtime envelopes documented at Sprint 0.5. Small DAGs
   explicitly parity-only against graphviz_dot (which runtimes ~60-80 ms
   on chain_100 -- Dagua cannot be expected to beat C).
5. Scale: default produces a layout within per-tier runtime budget (see 03)
   for N in {10, 100, 1K, 10K, 100K, 1M, 10M} without OOM or NaN.
6. Features: respects pinning, flex, alignment, and user clusters end-to-end,
   including across multilevel coarsening.
7. Composability: default pipeline imports ONLY from `dagua.layout.ops`.
   Zero imports from `dagua.layout.engine` or inline helpers. Multi-op
   bundles with controlled mutation (nested cluster execution, label
   size feedback) are explicitly documented as bypasses.
8. Adversarial gate: each sprint's exit adversarial review passes (no
   unaddressed CRITICAL or HIGH findings).
9. Iteration discipline: `sprint_<N>/iteration_log.jsonl` exists per sprint
   with at least 5 entries, profile/version metadata, Pareto deltas vs
   competitors, and at least one competitor-extracted op per
   extraction-eligible sprint.
10. Sprint 9 ship checklist fully satisfied (see 02 Sprint 9 for the list).

## Non-Goals

- Outperforming Graphviz at the smallest scale where Graphviz is already clean.
  We aim for parity there, excellence where Graphviz fails.
- Supporting non-2D layouts. 2D only.
- Shipping a new rendering pass. Existing render stays.
- Replacing all 23 competitor pipelines. They stay as registered algorithms.

## Sprint Map (ordered)

| # | Name | Role | Clock budget |
|---|------|------|---|
| 0 | Pipeline Decompose + Default Flip + MVP Iteration | Prereq | 1.5-2 days |
| 0.5 | Benchmark Authority + Opaque Held-Out | Prereq | 1.5-2 days |
| 1 | Initialization + Gradient Core + Memory Port | Rebuild + port mem ops | 2-3 days |
| 2 | Multilevel V-Cycle + Hierarchy Memory Parity | Differentiable scaling | 3 days |
| 3 | Hybrid Classical Steps | Warm-starts + polish | 2 days |
| 4 | Cluster-as-Node + Hierarchical Flex | Nested cluster placement | 2 days |
| 5 | Pinning + Flex End-to-End | User constraints across coarsening | 1.5 days |
| 6 | Edge Routing (Differentiable) | Integrated edge optimization | 2 days |
| 7 | Node Size + Text Polish | Content-aware sizing + label collision | 1.5 days |
| 8 | Scale Ladder Hardening + Hybrid Force Branch | 10M+ runtime + dispatch | 2-3 days |
| 9 | Aesthetic Dial-In + Ship Checklist | Final tuning + release | 1.5-2 days |

Clock budgets are pessimistic working-day estimates. Each sprint is
independent-executable (see success criteria). Sprint 0 is a prerequisite for
all others; Sprints 1-3 are sequentially dependent; 4-7 can partially interleave;
8-9 are exit gates.

Full sprint specs in files 01_audit_and_decompose.md through the 02_sprint_map.md
addendum. One file per sprint starting at 01.

## Risks (top three, revised twice per adversarial review rounds)

1. **Small-graph parity ceiling.** Per Codex's measurement, `chain_100`
   runs ~62 ms in graphviz_dot vs ~93 ms in Dagua today (1.49x, on the
   runtime bar's boundary). Mature discrete DAG heuristics give Graphviz
   dot a structural runtime/quality advantage at small N that Dagua may
   approximate but not match. Mitigation: parity floor rather than strict
   beat for small DAG / tree families (see 10 family floors). See 08 R15.
2. **Memory-parity gap.** Ops pipeline hardcodes `backward_mode="combined"`
   and lacks checkpointing + hybrid device support. Sprint 1 MUST port
   these before Mega/Ultra budgets are exitable. See 08 R3.
3. **Competitor extraction cargo-cult.** Extractions like
   `NetworkSimplexLoss` risk copying the name without porting the
   discrete behavior. Mitigation: reclassified as multi-op bundles with
   explicit mini-specs (11). Adversarial review checks "why" preserved.
   See 08 R16.

## Open Questions for User (top three -- full list in 09_open_questions.md)

1. Legacy engine: delete or demote to `_archive/` after the default flips?
   Some downstream code may still call `_layout_inner` directly.
   (See 09 Q1.)
2. Held-out opacity: go with opaque storage (secret salt + hashes-only
   manifest) vs convention-only "don't look" approach?
   (See 09 Q13; recommendation: opaque.)
3. Bit-for-bit preservation during Sprint 0 decomposition, or accept small
   numerical differences from reordered float ops? (See 09 Q2;
   recommendation: metric-level parity, not bit.)

## Cross-cutting Infrastructure

- **Fast iteration harness** (09 Q4): `scripts/iterate_native.sh` runs the
  default on a rotating graph subset + scalar score in <=8s. Built Sprint 0.
- **Iteration loop** (10): the core work pattern -- pick weak graphs,
  hypothesize, measure quality+runtime delta, Pareto check vs competitors,
  keep or revert. Codified in 10_iteration_loop.md.
- **Competitor extraction** (11): per-sprint concrete targets from OGDF,
  Graphviz, ELK, dagre, cola.js, Gephi, cuGraph. Codified in
  11_competitor_weaving.md.
- **Head-to-head competitor benchmark** at every sprint exit: Dagua vs
  the full 16-variant authoritative matrix on iteration + held-out. Pareto
  gate calibrated at Sprint 0.5 baseline, ramping to 90% iter / 80% held-out
  at Sprint 9 with per-family floors. See 10 and 11.
- **Held-out suite**: opaque, secret-salt-derived, 30-42 graphs, never
  iterated against. See 03.
- **Rolling random seed**: per-sprint regenerated, secret-salt-derived.
- **Codex adversarial review** between every sprint; mid-sprint for any
  sprint longer than one clock day. See 06.

## Entry point map

| For... | Read first |
|--------|-----------|
| Executing Sprint N | 02_sprint_map.md, then the sprint file |
| Understanding the test matrix | 03_test_matrix.md |
| Understanding how we score | 04_evaluation_rubric.md |
| Dispatching Codex | 05_multi_agent_orchestration.md |
| Reviewing plan or code | 06_adversarial_review_protocol.md |
| Researching prior art | 07_research_targets.md |
| Managing risk | 08_risk_register.md |
| Parked questions | 09_open_questions.md |
| **Within-sprint iteration cycle** | 10_iteration_loop.md |
| **Per-sprint competitor extraction** | 11_competitor_weaving.md |

## What the Native Default IS and IS NOT (Dagua DNA)

IS:
- Differentiable by default. PyTorch, autograd, GPU-capable.
- Composable. Every stage is a registered op.
- Hybrid-friendly. Classical warm-starts and polish where they measurably win.
- Hierarchical-aware. USER-DEFINED clusters are first-class visible entities;
  SEPARATELY, internal auto-coarsening for performance is a distinct concern
  invisible to the user. These two must never be conflated.
- Pinning-correct. User-set positions and flex constraints survive coarsening.

IS NOT:
- A reimplementation of any single competitor algorithm.
- Forced to be 100% differentiable.
- Dependent on imports from `dagua.layout.engine`.
- Committed to a specific paper or library API.
- An auto-clusterer. If the user gives no clusters, the output is a flat
  hierarchy from the user's perspective; internal coarsening is just a
  performance trick.
