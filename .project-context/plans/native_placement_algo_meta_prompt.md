# Native Dagua Placement Algorithm -- Meta-Plan Prompt

This is the optimized prompt to paste into a fresh Claude Code session at
`/home/jtaylor/projects/dagua` to generate the meta-plan for the native placement
algorithm mega-sprint. The prompt instructs Claude Code to produce a structured set
of ten plan files under `.project-context/plans/native_placement_algo/` and halt
for user approval before any sprint begins.

Generated: 2026-04-22.

---

```
<role>
You are an architect agent working on Dagua, a GPU-accelerated differentiable graph layout engine built on PyTorch. You are NOT writing code in this turn -- you are producing the META-PLAN that will guide the mega-sprint for building Dagua's native default placement algorithm. The plan itself will drive multiple sprints spanning several days (at minimum) of subsequent work. This is the core intellectual deliverable of the Dagua project.
</role>

<context>
Read before planning. Do not plan from assumption.

Project files:
1. /home/jtaylor/projects/dagua/CLAUDE.md -- project briefing, dispatch rules, architect identity
2. .project-context/architecture.md -- full system map
3. .project-context/baton.md -- current session state
4. .project-context/autonomous_gate.json -- any active gating
5. .project-context/knowledge/gotchas.md and decisions.md -- prior traps + decisions
6. dagua/layout/engine.py -- optimization loop + pipeline dispatch
7. dagua/layout/ops/ -- 268 registered composable ops (34 modules)
8. dagua/layout/ops/pipelines/ -- 23 existing pipelines including dagua_native.py
9. dagua/eval/ + eval_output/report/*.md -- benchmark + comparison infrastructure

Session memory (read on-demand, not all at once):
~/.claude/projects/-home-jtaylor-projects-dagua/memory/ -- especially project_memo.md,
algorithms.md, loss_functions.md, project_sgd2_insights.md, project_weighted_edges.md,
eval_system.md, feedback_reimpl_fidelity.md, feedback_autonomous_iteration.md.

Prior work already landed:
- 23 competitor algorithms faithfully reimplemented and decomposed into ops
- 268 ops with full docstrings; pipelines have zero private helpers
- Rendering polished to 9.25/10 mean across 335 images
- Benchmark infra: 20k runs, 95 graphs, 235 engine variants, fidelity validation, competitor caching
- Native Dagua engine exists with a pipeline mode (use_pipeline=True in dagua_native.py)

Audit questions to answer as Step 1:
- Is dagua_native.py FULLY decomposed into registered ops with zero inline helpers? If not,
  decomposition is a prerequisite sprint (Sprint 0).
- Does the benchmark harness support rapid single-algorithm iteration (one seed, freshly
  generated random graphs, sub-minute turnaround)? If not, flag as a dependency.
- What is the current native pipeline's speed vs quality profile per graph family?
</context>

<task>
Produce a meta-plan for building Dagua's native default placement algorithm -- the algorithm
that runs when a user calls dagua.layout(g) with no algorithm specified.

Output the plan as a structured set of files under
.project-context/plans/native_placement_algo/:

- 00_overview.md -- vision, success criteria, non-goals, risks, sprint map
- 01_audit_and_decompose.md -- Sprint 0: audit + decompose if needed + baseline metrics
- 02_sprint_map.md -- ordered sprint list with entry/exit criteria each
- 03_test_matrix.md -- graph generators, scale ladder, feature matrix, anti-overfit plan
- 04_evaluation_rubric.md -- quantitative metrics + qualitative aesthetic criteria, weights, regression bars
- 05_multi_agent_orchestration.md -- when to dispatch Codex, subagents, iMessage human judgment
- 06_adversarial_review_protocol.md -- Codex adversarial review cadence, prompt templates, escalation
- 07_research_targets.md -- literature to mine, competitor code to study, reusable ideas
- 08_risk_register.md -- known risks, mitigations, abort criteria per sprint
- 09_open_questions.md -- questions requiring user input before Sprint 1 begins

Do NOT produce implementation code. Do NOT start any sprint. Produce only the plan artifact
and await approval.
</task>

<algorithmic_dna>
The plan must serve Dagua's identity:

1. Modern optimization is the default toolkit: autograd, differentiable losses tied to
   aesthetic criteria, GPU vectorization, learnable node positions.
2. Differentiability is a means, not a dogma. Where a classical non-differentiable step
   outperforms, incorporate it cleanly -- warm-start, staged hybrid, straight-through
   estimator, relaxation. The engine must absorb hybrids WITHOUT becoming a frankenstein monster.
3. Composability is non-negotiable: every algorithmic step is a registered op, reusable by
   other pipelines. No inline helpers in the native pipeline.
4. Hierarchical clustering is a first-class scaling mechanism: a cluster acts as a single
   node at its hierarchy layer. Exploit this to tame combinatorial explosion.
5. Pinning and flex are user-facing primitives: users pin some properties, set flex levels
   on others, and the rest re-routes. The native default must respect pinning end-to-end.
6. Scope of optimization: node positions, edge routing, node sizes, text placement/size,
   cluster bounding boxes. Out of scope: purely cosmetic features that do not benefit from
   optimization.
7. Graph variety: directed, undirected, skip connections, self-loops, DAGs, disconnected
   components, trees, near-cliques, long chains, grids.
</algorithmic_dna>

<what_the_plan_must_cover>

Core algorithm structure:
- Pipeline stages (coarsening, initialization, coarse-to-fine refinement, differentiable
  optimization, edge routing, cluster fitting, text placement, final polish). Which stages
  lift from existing pipelines (sgd2, fr, kk, sfdp, stress_sgd, umap, tsnet, sugiyama,
  spectral) vs invent new.
- Initialization strategy: random vs spectral vs structural vs warm-start from fast
  classical pass. Cost vs quality tradeoff.
- Coarsening/multilevel: how to build the hierarchy automatically when the user provides
  none; how to inherit user-supplied clusters when present.
- Cluster-as-node abstraction: concrete mechanics for treating a cluster as a single
  placement unit at its hierarchy layer.
- Hybrid differentiable/non-differentiable steps: which need relaxation, which need
  warm-start, which are native-differentiable.

Optimization beyond node positions:
- Edge routing: splines, bundling, crossing minimization, port selection.
- Node size/shape: content-aware sizing; text measurement feedback loop.
- Text placement: node labels, edge labels; collision-aware.
- Cluster bounding boxes: padding, shape (rectilinear vs hull), label placement.

Graph feature support:
- Directed vs undirected: when does directionality bias the layout (ranking, layering)?
- Skip connections: curved routing, reduced force weight, distinguishable rendering
- Self-loops: placement + rendering impact
- Disconnected components: placement policy
- Pathological topologies: star, complete, grid, very long chain, densely cyclic

User pinning / flex system:
- Pinned positions: gradient masking, hard constraints
- Flex levels (soft/firm/rigid) per property, translated to loss weights or constraints
- Alignment constraints: axis locking, relative ordering
- Compatibility of pinned properties with hierarchical coarsening

Scale ladder and speed targets:
- Explicit node-count tiers: 10, 100, 1K, 10K, 100K, 1M, 10M+
- Runtime budgets per tier (derived from competitor benchmarks + user expectation)
- Memory budgets (respect 3-4x autograd multiplier)
- When to drop to CPU, when to require GPU, when to coarsen more aggressively

Benchmark-guided iteration:
- Which benchmark subset drives fast per-sprint feedback (sub-minute)
- When to trigger a full benchmark run
- How to fairly compare against reimplemented competitors
- How to add new benchmark graphs without invalidating historical comparisons

Aesthetic evaluation:
- Quantitative proxies: edge crossings, node overlap, edge-node crossings, stress,
  neighborhood preservation, symmetry scores, angular resolution, aspect ratio, label overlap
- Qualitative / AI aesthetic: adversarial-agent rubric (Claude reviewer vs Codex reviewer
  vs human judge), disagreement resolution
- iMessage-based human judgment: when to ping the user with image comparisons; rate-limit
  so it stays valuable, not noise
- Weighting: how quantitative and qualitative combine into a ranking

Multi-agent orchestration:
- Codex for implementation: scope 1-3 files per task, split otherwise
- Codex adversarial review: separate from standard review, focused on breaking assumptions
- Claude Code subagents for literature scans, competitor-library code reading, alternate passes
- Parallel dispatch discipline: pre-dispatch checklist (shared files, behavioral coupling,
  known-red tests, model mix)

Non-regression and anti-overfitting:
- Held-out graph set (never iterated against)
- Rolling random-generator seed per sprint (not reused)
- Full sample-graph suite runs at sprint exit, not mid-sprint
- Explicit regression bars per aesthetic metric

Failure modes and debugging:
- NaN gradients, exploding losses, stuck local optima, bad initialization
- Determinism policy: seeds, nondeterministic ops to avoid

Literature + competitor code targets:
- Papers to mine: ForceAtlas2, SFDP, sgd^2, stress majorization variants, neural layout
  papers, GNN-based layouts, Kamada-Kawai extensions
- Implementations to read: Gephi, OGDF, Graphviz sfdp, NetworkX, igraph, cytoscape
- Extraction targets: initialization tricks, annealing schedules, edge length heuristics,
  coarsening algorithms
</what_the_plan_must_cover>

<adversarial_review>
Extensive adversarial Codex review is mandatory and must be woven into the plan's cadence:

1. Plan-level review. Once the meta-plan is drafted, dispatch /codex:adversarial-review at
   least TWICE with different focus areas BEFORE declaring the plan ready. Suggested foci:
   "attack the scaling assumptions and memory budgets", "attack the evaluation rubric for
   overfit risk", "attack the hybrid differentiable / non-differentiable integration for
   frankenstein risk". Revise based on findings.

2. Sprint-level review. Every sprint exits through an adversarial Codex review of the
   produced code AND the plan step's claimed outcomes. The review must verify: did the
   sprint meet its exit criteria, did it regress any prior sprint's gain, did it smuggle
   in assumptions not approved.

3. Mid-sprint checkpoint reviews. For any sprint longer than one day, include an
   intermediate adversarial review to prevent wasted effort on a bad branch.

4. Review prompt templates. The plan must include concrete /codex:adversarial-review
   prompt templates for each stage, using XML blocks (task / verification_loop /
   action_safety / missing_context_gating).

5. Escalation protocol. When adversarial review and implementation Codex disagree, the
   plan specifies the resolution path (fresh Codex thread with dual briefs, user
   escalation, or Claude arbitration).
</adversarial_review>

<process>
Follow this order. Do not jump ahead.

1. Read the context files above. Especially the architecture map, dagua_native.py, and
   the listed memory files.

2. Audit current state. Dispatch a subagent if useful. Answer: is dagua_native.py fully
   decomposed? What are its stages? What does benchmark data say about current strengths
   and weaknesses per graph family? What is speed vs quality today?

3. Surface forks before planning. For each major design decision under
   <what_the_plan_must_cover>, identify realistic options with pros/cons. Park the ones
   requiring the user in 09_open_questions.md. Do NOT lock architecture unilaterally.

4. Draft the ten plan files listed in <task>. Keep each file tight; push longer reasoning
   into .project-context/knowledge/research/ side notes.

5. Run two adversarial Codex reviews on the drafted plan with different focus areas.
   Revise based on findings.

6. Report back (see <report_back> below). Do not start Sprint 0.
</process>

<success_criteria>
The meta-plan is done when:
1. Any engineer (human or Codex) can execute any single sprint file without reading the
   others in full.
2. Each sprint has binary entry/exit criteria, a named execution pattern
   (Claude-plans + Codex-implements, solo-Codex, human-judgment, etc.), and a rollback plan.
3. Non-regression is explicit: each sprint names which prior gains must not degrade and
   how that is verified.
4. Overfitting risk is addressed: held-out graph suite + rolling random seed policy specified.
5. Adversarial Codex review is baked into the cadence as a gate between sprints AND at
   mid-sprint checkpoints, not as a final step.
6. Open questions requiring user input are surfaced in 09_open_questions.md BEFORE
   Sprint 1 can begin.
7. Length discipline: 00_overview.md is one page; each sprint file is three pages max.
</success_criteria>

<constraints>
- ASCII only. No unicode arrows, em dashes, smart quotes, fancy bullets. Use ->, --, *, ^2.
- Do not write implementation code in this turn.
- Do not start any sprint until the user explicitly approves the plan.
- When dispatching Codex, use the exact invocation from CLAUDE.md:
  codex exec --skip-git-repo-check --sandbox danger-full-access [--cd <workdir>] "<prompt>"
  Use run_in_background=true via Bash. Verify within 8s with pgrep -fl "codex exec".
- Respect disk-space pre-flight rules for any bulk work.
- Memory discipline: read memory files on-demand, not all at once. MEMORY.md is always loaded.
- No AI attribution in commits, PRs, or code.
- Dagua project rule: one feature branch at a time besides main unless user asks otherwise.
</constraints>

<anti_patterns>
Reject these tendencies:
- An idealized plan that reads well but has no binary exit criteria
- Designing for every graph structure equally -- the plan must rank priorities
- "We'll figure it out later" as the answer to hard design questions -- either decide or
  park in 09_open_questions.md
- Treating adversarial review as a final step rather than a gate
- Treating non-differentiable methods as second-class -- the native algo is allowed to use
  classical warm-starts in its critical path if they win
- Overfitting the plan to the current sample graph set -- the test matrix must include
  held-out and random-regenerated sets
- Unbounded sprint scope -- each sprint is bounded by clock-time and exit criteria
- Surfacing 30 open questions -- pick the ones that block architecture; defer the rest
</anti_patterns>

<deliverable_format>
- Ten markdown files under .project-context/plans/native_placement_algo/
- 00_overview.md is the entry point; all others are referenced from it
- Each sprint file follows: Goal / Entry criteria / Exit criteria / Test plan /
  Adversarial review plan / Rollback plan / Open questions
- Length: overview one page, each sprint three pages max
- ASCII only
</deliverable_format>

<report_back>
When done, reply with exactly:
1. One-paragraph executive summary of the plan
2. Top three open questions from 09_open_questions.md
3. Top three risks that shaped the plan
4. Sprint list: name + one-line goal each
5. Path to 00_overview.md
6. Literal line: "Awaiting approval before starting Sprint 0."

Do not summarize further. Do not begin Sprint 0. Wait for the user.
</report_back>

<default_follow_through_policy>
Default to the most reasonable low-risk interpretation and keep going. Only stop for
missing details that change correctness, safety, or irreversible architectural commitments.
Park deferrable decisions in 09_open_questions.md rather than pausing the plan.
</default_follow_through_policy>
```
