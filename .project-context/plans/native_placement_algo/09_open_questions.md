# Open Questions for User

Questions that require your answer BEFORE Sprint 0 can exit. The plan is
deliberately short on blocking questions -- most are parked here rather than
left undecided.

Priority: P0 blocks Sprint 0. P1 blocks later sprints. P2 is a preference
check, non-blocking.

## Q1 (P0) -- Legacy engine fate

`dagua/layout/engine.py:_layout_inner` and its ~3000 lines are the current
default path when `config.algorithm is None`. Sprint 0 flips the default to
the ops pipeline. What happens to the legacy code?

Options:
- **A. Archive.** Move to `dagua/layout/_archive/legacy_engine/` with a README
  declaring it frozen reference only. Leave an import shim at the old path
  for backward compatibility.
- **B. Delete.** Remove entirely. Break any downstream callers; they can
  pin to an older version.
- **C. Keep, demote.** Keep at current path, but stop testing and stop
  advertising. Lets Codex and TorchLens limp along.

Recommendation: A. We gain a clean default, lose nothing, and the shim is
cheap. Deleting is tempting but asks for TorchLens breakage.

## Q2 (P0) -- Bit-for-bit preservation during Sprint 0

When Sprint 0 decomposes helpers out of `engine.py` into ops or resolvers,
should the output of the new default be bit-for-bit identical to today's
output on seed=42? Or is a small numerical difference (from reordered float
ops) acceptable?

Options:
- **A. Bit-for-bit strict.** Blocks any simplification of the config prologue.
- **B. Metric-for-metric, not bit.** Composite score unchanged within 1%.
- **C. No preservation guarantee.** Declare a new default; old outputs gone.

Recommendation: B. Bit-for-bit is practically a trap and would block honest
refactor. Metric-level parity is meaningful.

## Q3 (P1) -- Cluster hierarchy when the user gives none

Sprint 4 introduces cluster-as-node. If the user provides zero clusters,
should we:

Options:
- **A. Auto-construct a hierarchy** via community detection (Louvain or
  similar) and use it as a multilevel coarsening strategy.
- **B. Do not cluster.** Only use user-supplied clusters.
- **C. Hybrid.** Auto-construct for N > some threshold (e.g., 10K) where
  coarsening is necessary for performance, but do not expose to the user as
  "their clusters."

Recommendation: C. Matches the scaling reality while respecting user intent.

## Q4 (P0) -- Fast iteration harness design

Sprint 0 Task 0.4 builds `scripts/iterate_native.sh`. Should it:

Options:
- **A. Single graph, single metric, single image.** <=8s.
- **B. Single graph, full metric dict, image + delta-vs-baseline.** 10-15s.
- **C. Rotating 5-graph subset with averaged score.** 30-45s.

Recommendation: A for daily iteration, C as `scripts/iterate_native.sh --suite`
mode for end-of-day. Both ship.

## Q5 (P1) -- Directed-vs-undirected default algorithm split

Sprint 1 picks the default initializer per graph family. Should the default
end up dispatching two entirely different pipelines for directed vs undirected,
or one unified pipeline that conditions per family?

Options:
- **A. One pipeline, family-conditional ops.** Easier to maintain; ops choose
  behavior based on `classify_graph`.
- **B. Two pipelines.** `dagua_native_dag` and `dagua_native_undirected`,
  with the default dispatcher routing. More code, clearer separation.
- **C. One pipeline, but a family-specific config resolver flips weights.**

Recommendation: A or C. B creates two entities to maintain and regress.

## Q6 (P2) -- iMessage HJ ping rate limit

Proposed default: one 3x3 grid per sprint exit; extras only if adversarial
disagreement >1 point AND the flagship graph is in scope. Is this cadence
right for you?

Options:
- **A. As proposed.** ~9-10 pings across the plan.
- **B. More frequent.** One per mid-sprint checkpoint as well.
- **C. Less frequent.** Only at Sprint 0 exit, Sprint 5 exit, Sprint 9 exit.

Recommendation: A. Balances your judgment cost with our need for ground truth.

## Q7 (P2) -- Scale-beyond-10M is in or out?

The scale ladder lists 100M+ as "deferred." Do you want Sprint 8 to treat 100M
as an explicit target (requires disk-offload coarsening work already noted
in gotchas), or is 10M the official ceiling for this plan?

Options:
- **A. 100M explicit.** Add Sprint 8.5 for offload hardening.
- **B. 10M is the ceiling.** 100M becomes a future effort.
- **C. 10M ceiling, but keep the offload scaffolding live so a future push
  is incremental.**

Recommendation: C. We do the bookkeeping now but do not chase 100M runs here.

## Q8 (P1) -- Resolver module location

Sprint 0 Task 0.1 creates a resolver module for config-time helpers. Where
should it live?

Options:
- **A. `dagua/layout/ops/pipeline_resolve.py`.** Keeps config logic near ops.
- **B. `dagua/layout/resolve.py`.** Sits next to `engine.py` (which becomes
  minimal).
- **C. Merge into `dagua/config.py`.** Simplest, but config.py gets fat.

Recommendation: B. Config logic is not ops-layer; ops are optimization
primitives. This keeps the layer boundary clean.

## Q9 (P2) -- Branch strategy and sprint branch naming

Proposed: `feat/native-algo-sprint-N`, merge to main at each sprint exit.
Alternative: a long-running `feat/native-algo` branch with intermediate
tags. Alternative 2: PR-per-sprint to `feat/native-algo` and squash at end.

Recommendation: `feat/native-algo-sprint-N` for atomic reviewable PRs, but
merging to an intermediate `feat/native-algo` branch, which is merged to
main at the end of Sprint 9. Keeps main clean.

## Q10 (P2) -- Competitor reimplementation parity

Today's 23 competitor pipelines (fr, kk, fa2, sgd2, ...) stay live. Should
they continue to ride existing test coverage only, or do we also benchmark
them against our new default for every sprint exit?

Options:
- **A. Only test them on their own fidelity.** As today.
- **B. Benchmark ALL of them at every sprint exit.** Slow, thorough, shows
  how the default stacks up.
- **C. Benchmark them at Sprint 5 and Sprint 9 only.** Middle ground.

Recommendation: C. Existing competitor benchmarks are cached, so Sprint 5 +
Sprint 9 adds minimal runtime but gives us a strong "we are better than the
median" signal.

## Q11 (P1) -- Edge routing differentiability

Sprint 6 makes edge control points learnable. Does Dagua's identity require
full-graph joint optimization of node + edge positions, or is the staged
approach (freeze nodes, then optimize edges) acceptable?

Options:
- **A. Staged.** Ship Sprint 6 as described.
- **B. Joint.** Add a Sprint 6.5 for joint optimization.
- **C. Both modes, user picks.** More surface area, more maintenance.

Recommendation: A for Sprint 6 shipped; B parked as a future "v2 default."
Joint optimization is powerful but its failure modes (edges pulling nodes
into odd positions) need their own sprint to contain.

## Q12 (P2) -- Aesthetic weighting is my problem or yours?

Sprint 9 tunes the composite's coefficients. Should Claude own this tuning
or should the user approve every weight delta?

Options:
- **A. Claude owns, user approves final set.** Standard.
- **B. User approves each weight change.** Heavyweight but maximally
  transparent.
- **C. Claude publishes weight changes to an Obsidian note; user reviews
  async.**

Recommendation: A. Claude documents every tuning decision in the Sprint 9
exit note and HJ-pings with the final grid.

## Q13 (P0) -- Opaque held-out storage (NEW, from adversarial review)

Adversarial review flagged: committing held-out graph tensors to git and
telling people "don't look" is not real isolation. Recommendation is
opaque storage: secret salt (gitignored), topology hashes committed, and
graphs regenerated on demand.

Options:
- **A. Opaque (recommended).** Secret salt at
  `.project-context/private/holdout_salt`, MANIFEST.json with hashes only,
  graphs regenerated on demand, destroyed after metric computation, pytest
  fixture enforces opacity.
- **B. Convention-only.** Keep committed graphs; rely on discipline to not
  iterate against them.
- **C. Hybrid.** Commit only topology hashes; graphs on developer machines
  only, not in repo.

Recommendation: A. The opacity cost is small (salt + regenerate) and it
removes a real overfit vector.

## Q14 (P1) -- Held-out size bump

Adversarial review: 15 held-out graphs is one sample per tag. Not enough
to catch family-conditional drift.

Options:
- **A. 30 minimum (recommended), 42 preferred.** 2-3 per priority family;
  enforced small/medium split.
- **B. 20.** Lighter, less coverage.
- **C. 50+.** More coverage, slower exit runs.

Recommendation: A with the flexibility to grow to 42 at Sprint 0 Task 0.9.1.

## Q15 (P1) -- Emergency rubric-change path

Coefficients are frozen until Sprint 9. But what if Sprint 3-7 reveals a
metric is broken?

Options:
- **A. Emergency change requires user sign-off via iMessage (recommended).**
  All prior sprint metrics re-computed under new weights, logged as
  `metrics_recomputed.json`. Documented in 08 R13.
- **B. No change allowed.** Sprint 9 only. Forces us to optimize against a
  known-bad objective for up to 6 sprints.
- **C. Silent auto-change when Claude detects breakage.** Loses user
  transparency.

Recommendation: A. Codified in 04_evaluation_rubric.md "Freeze protocol"
section.

## Q16 (P2) -- Post-release audit suite source

The plan adds a "post-release audit suite" generated from a SECOND salt not
accessible during the plan. Who generates it?

Options:
- **A. User generates it after Sprint 9 exit and hands back to Claude.**
  Clean separation.
- **B. Third-party researcher generates it.**
- **C. Generate from a public benchmark (OGDF test suite).** Less controlled.

Recommendation: A. User writes a 32-byte random value to
`.project-context/private/audit_salt` after Sprint 9 and asks Claude to
re-run metrics. Claude has not seen that salt until then.

---

Please answer at least Q1, Q2, Q4, Q13 before Sprint 0 begins. Q3, Q5, Q8,
Q11, Q14, Q15 can be answered at the corresponding sprint's entry. Q6, Q7,
Q9, Q10, Q12, Q16 are non-blocking and can be answered anytime.

## Q17 (P1, new) -- Competitor list authority

Sprint exit runs head-to-head against graphviz_dot, elk_layered,
graphviz_sfdp, igraph_sugiyama, sgd2_multi. Is this the right set?

Options:
- **A. Yes, plus Gephi FA2 for undirected (recommended).** 6 total.
- **B. Add dagre and networkx_spring** for JS / Python user parity. 8 total.
- **C. Minimal: 3 (graphviz_dot, elk_layered, sgd2_multi)** for speed.

Recommendation: A.

## Q18 (P1, new) -- Runtime normalization across devices

Comparing Dagua GPU runtime to graphviz_dot CPU runtime is apples to
oranges. The "1.5x fastest competitor" target needs a rule.

Options:
- **A. Same-device comparison only.** Dagua-CPU vs graphviz-CPU;
  Dagua-GPU vs cuGraph-GPU (when present). Honest but requires two
  competitor sets.
- **B. "Best competitor runtime across any device" as denominator.**
  Harsh: makes GPU-accelerated Dagua compete with C-based graphviz.
- **C. Family-specific.** Undirected N>=100K compared GPU; small DAGs
  compared CPU. Complex to specify.

Recommendation: A, with a documented per-family device choice in 03.

## Q19 (P2, new) -- Extraction veto

Sprint extractions are mandatory. Should the user be able to veto an
extraction target if ideologically opposed (e.g. "we will not use any
Graphviz code even as reference")?

Options:
- **A. User veto via 09 before sprint begins.** Recorded in sprint file.
- **B. No veto; extraction is always allowed.** Simpler.
- **C. Blanket policy up-front: we use everything open-source.**

Recommendation: C. All 29 competitors listed are open-source.

## Change log

2026-04-22: Initial draft with Q1-Q12.
2026-04-22: Added Q13-Q16 per adversarial review findings
(rubric-code mismatch, held-out inspectability, emergency rubric change,
post-release audit).
2026-04-22: Added Q17-Q19 per user directive "best open graph algo in the
market" (competitor list authority, runtime normalization, extraction veto).

## User resolutions (2026-04-22)

| Q | Resolution |
|---|-----------|
| Q1 | A (archive `_layout_inner` to `_archive/legacy_engine/` with shim) |
| Q2 | C (no preservation guarantee; we are iterating anyway) |
| Q3 | FLAT hierarchy when no user clusters given. Must distinguish user-defined clusters (semantic, first-class, visible) from internal auto-coarsening (performance strategy, invisible to user). |
| Q4 | A (single-graph <=8s MVP) |
| Q5 | A (one pipeline, family-conditional ops) -- Claude judgment |
| Q6 | More frequent than proposed: regular iMessage pings |
| Q7 | 10M ceiling is fine |
| Q8 | Claude judgment (going with B: `dagua/layout/resolve.py` next to `engine.py`) |
| Q9 | Stay on `feat/bench-and-aesthetics` branch; commit current changes |
| Q10 | Superseded by Sprint 0.5 authoritative matrix + hash-change auto-refresh |
| Q11 | A (staged: freeze nodes, then optimize edges) -- Claude judgment, to confirm via investigation before Sprint 6 |
| Q12 | A (Claude tunes autonomously, iMessage final set + grid at Sprint 9 for sign-off) |
| Q13 | A (opaque held-out with secret salt) |
| Q14 | A (30 minimum, 42 preferred) |
| Q15 | A (emergency rubric change via user iMessage sign-off) |
| Q16 | A (user supplies post-release audit salt, Claude re-runs metrics) |
| Q17 | 16-variant matrix (resolved during adversarial iteration) |
| Q18 | A (same-device comparison only) |
| Q19 | C (all open-source competitors fair game) |

All P0 blockers resolved. Sprint 0 cleared to begin.
