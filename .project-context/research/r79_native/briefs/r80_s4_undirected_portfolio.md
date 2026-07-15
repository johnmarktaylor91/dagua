# r80-S4: Undirected-class portfolio route (probe-gated), directedness plumbing + inference fix

## Why (read first)
dagua's native algorithm loses ~26 of its 34 benchmark losses on semantically-undirected
graphs (social/community/SBM/small-world/mesh/scale-free) where force engines win. Three
prior attempts routed these to a bespoke stress core and FAILED. This round does something
different: route them to dagua's OWN bit-faithful reimplementations of sfdp/neato (built in
the fidelity campaign), finish with size-aware overlap projection, and pick the best
candidate by the SAME honest composite the benchmark uses. Headroom is proven:
graphviz-sfdp positions + our overlap cleanup score +12..20 over best external on the loss
graphs (.project-context/research/r79_native/P3B2_STRESS_FORENSICS.md).

Architect-confirmed findings you build on (verified 2026-07-08, do not re-derive):
- `_infer_semantically_directed` (dagua/layout/graph_classify.py) mislabels the classic
  undirected corpus graphs as DIRECTED (karate, sbm_4x30, scale_free_ba_120,
  small_world_100, grid_5x5, r79_weighted_community_4x18, r79_weighted_small_world_120)
  because they store each undirected edge ONCE (reciprocal ratio 0) and don't trigger the
  deep-layering rule. It ALSO mislabels transformer_layer (a genuine deep DAG) as
  UNDIRECTED via that same deep-layering rule (num_layers/N >= 0.4 -> False).
- The `graph=` kwarg of classify_graph -- the only way an explicit user declaration
  `DaguaGraph.is_semantically_directed` reaches routing -- is NOT passed at the real call
  sites: dagua/layout/engine.py:1809 and dagua/layout/resolve.py:540 (check
  multilevel.py:2017 too). Declaration plumbing is dead today.
- The eval scoring oracle is tag-based: dagua/eval/graphs.py::is_semantically_directed
  (`"undirected" not in tags`). The corpus never sets the declaration on the DaguaGraph
  objects, so layout can never see it.

## Setup
Main repo worktree (DO NOT modify it): /home/jtaylor/.claude/worktrees/dagua-native
(branch r79/native). Create your own worktree + branch:
  git -C /home/jtaylor/.claude/worktrees/dagua-native worktree add \
      /home/jtaylor/.claude/worktrees/dagua-native-p2 -b r80/undirected-portfolio
Check `df -h /` first; if free space < 12GB, STOP and report. Then create a venv in p2:
  cd /home/jtaylor/.claude/worktrees/dagua-native-p2 && uv venv .venv && \
  uv pip install -p .venv/bin/python -e ".[dev]" python-igraph
(uv hardlinks from its cache so this is cheap.) Verify
`.venv/bin/python -c "import dagua,pathlib;print(pathlib.Path(dagua.__file__))"` resolves
INSIDE dagua-native-p2. All commands below use this venv from the p2 dir.

## Stage 1 -- PROBE (no product code; hard decision gate)
Write a standalone script scripts/r80_probe_undirected_portfolio.py (committed, reusable):
- Iterate every corpus graph where the eval oracle says undirected
  (`from dagua.eval.graphs import get_test_graphs, is_semantically_directed`).
- For each, produce candidate layouts using dagua's public API with our own reimplemented
  pipelines: LayoutConfig(algorithm="sfdp"), algorithm="neato", algorithm="kk" (confirm
  registry names via dagua/layout/ops/pipelines/__init__.py; they are the fidelity-campaign
  reimplementations -- they must NOT invoke external binaries; verify no graphviz
  subprocess is spawned). Seed 42, defaults otherwise.
- Apply dagua's size-aware overlap projection to each candidate (find the projector entry
  point used by native_stress.py's projection stage in dagua/layout/projection.py; use
  REAL label-size node boxes exactly as the benchmark scoring does).
- Score each candidate + the frozen dagua row + frozen externals with the IDENTICAL honest
  composite the baseline harness uses for undirected rows -- read scripts/r79_baseline.py
  and import the same functions (composite_auto(..., is_semantically_directed=False), same
  tier, same seed). Frozen rows: eval_output/r79_baseline/results.json + positions/*.pt in
  the MAIN worktree (read-only access there is fine).
- Emit a markdown table per graph: current-dagua, best-external (name+score), sfdp+proj,
  neato+proj, kk+proj, and best-candidate-vs-best-external delta. Also wall-time per
  candidate. Save to .project-context/research/r79_native/P8_PORTFOLIO_PROBE.md.

DECISION GATE: count loss-graphs (current dagua LOSS vs externals) where
max(candidates) >= best_external - 0.5. If that count < 10 of ~26, STOP after committing
the probe + report -- do not wire routing. Otherwise continue.

## Stage 2 -- plumbing + inference fix
1. Pass `graph=` through the classify_graph call sites where a graph object is in scope
   (engine.py:1809, resolve.py:540; audit multilevel.py:2017 and the pipeline-internal
   classify calls -- only plumb where a real DaguaGraph is available, no signature
   contortions).
2. Corpus declaration: in dagua/eval/graphs.py, when constructing TestGraph objects, set
   `graph.is_semantically_directed = False` for graphs whose tags contain "undirected",
   using the SAME oracle function the scorer uses (single source of truth -- do not
   duplicate the tag logic). This is honest: a real user with an undirected graph declares
   it; external force engines already ignore direction unconditionally.
3. Inference fix (fallback for undeclared graphs), in _infer_semantically_directed:
   the deep-layering rule (num_layers/num_nodes >= 0.4 -> undirected) must NOT fire when
   the layering is chain-like/meaningful. Compute the fraction of edges whose layer span
   (layer[target]-layer[source]) equals 1: if that fraction is high (>= 0.6), return True
   (directed) even with deep layering -- genuinely deep pipelines (transformer_layer) have
   mostly adjacent-layer edges, mechanically-oriented graphs do not. Add unit tests for:
   a deep chain-of-blocks DAG -> True; a reciprocal-pair graph -> False; a mechanically
   index-oriented dense graph that previously hit the 0.4 rule -> still False.
   Keep the default-True bias for ambiguous graphs (directed is the safe default for the
   layered engine).

## Stage 3 -- the portfolio route
In dagua/layout/ops/pipelines/dagua_native.py:
- `_choose_native_pipeline`: after the forced-pipeline branch, if
  structure.is_semantically_directed is False, return "undirected_portfolio" (new).
  Trees/chains keep their existing early exits ABOVE this branch. Do not remove any
  existing branch.
- New module dagua/layout/ops/pipelines/native_undirected.py implementing the route the
  same way the existing "stress" route is invoked from _run_native_problem:
  - Candidate A: the incumbent -- whatever _choose_native_pipeline would have returned if
    the portfolio branch did not exist (compute it by factoring the remainder of the
    routing logic into a helper, NOT by copy-paste), run normally. This guarantees the
    route can never do worse than today wherever selection is honest.
  - Candidate B: our sfdp pipeline on the same problem tensors, then size-aware overlap
    projection with real node boxes.
  - Candidate C (only when quality >= high per the quality knob): neato + projection.
  - Score all candidates with the same composite used in Stage 1 (undirected flavor,
    fixed metric seed, real node sizes). Select argmax; tie goes to the incumbent.
  - Respect time_budget_s and the quality knob budgets (read how native_stress.py consumes
    them). If num_nodes > 1500, skip the contest and return the incumbent (record as a
    documented cap; the probe data will tell us if a higher cap is safe later).
  - Decomposable-ops philosophy: register the scoring/selection steps as ops where
    natural; follow the existing top-level-route precedent rather than inventing a new
    private-function layer (adversarial review already flagged private-func op-work as
    debt -- do not add more than the existing pattern does).
  - Weighted graphs: pass edge weights through unchanged (sfdp/neato pipelines already
    handle them). NO weight-semantics transforms in this round.
  - Clustered graphs: if cluster info is present, let the cluster-aware sfdp driver do its
    thing; if Stage-4 sweep shows any clustered-graph regression, EXCLUDE clustered graphs
    from the route predicate and record it as a residual instead of chasing it.
- The layout-time selection composite MUST use the flag consistent with what the scorer
  will use. With Stage-2 declaration in place, declared graphs are consistent by
  construction. For undeclared graphs the fixed inference decides; that is acceptable.

## Stage 4 -- gates (in order; stop and report if one fails twice)
1. Scoped tests: files you touched + tests matching "classify or routing or portfolio or
   native" -- consult .project-context/research/r79_native/KNOWN_RED_TESTS.md and deselect
   those. NEVER bare `pytest tests/ -x` (documented whack-a-mole trap).
2. Default-path safety: for 5 DIRECTED graphs (incl. transformer_layer,
   dependency_graph_100), assert positions are bit-identical before/after your branch
   (the portfolio route must not fire on them). transformer_layer is the landmine: it was
   inferred undirected before your Stage-2 fix -- prove it now routes directed.
3. Full sweep: `.venv/bin/python scripts/r79_baseline.py --dagua-only` in YOUR worktree.
   Acceptance: ZERO WIN->LOSS flips anywhere in the corpus, and the undirected class gains
   >= +6 graphs in best-or-tied vs the frozen 56/8/29 + 8/2/5 baseline. Record the full
   per-graph before/after table.
4. `ruff check` on touched files.

## Output contract
- Commits on r80/undirected-portfolio (conventional messages, no AI attribution).
- Evidence: .project-context/research/r79_native/P8_PORTFOLIO_EVIDENCE.md -- probe table,
  routing predicate description, per-graph sweep deltas, W/T/L before/after, candidate
  win-rate stats (how often sfdp vs neato vs incumbent won the contest), wall-time impact
  on the undirected class, caps and residuals.
- Final message: W/T/L line, top-5 biggest flips, any concerns, whether the projector
  branch (r80/projector, being built in parallel by another agent) would raise your
  numbers (estimate from probe rows where candidates still had residual overlaps).

## Hard rules
- Do NOT touch dagua/layout/projection.py or native_stress.py (another agent owns them
  this round). Use the projector via its existing public entry point only.
- Do NOT modify scripts/r79_baseline.py scoring logic, the frozen store, or tags.
  (Setting the declaration field on corpus graph CONSTRUCTION per Stage 2 is the one
  sanctioned eval/graphs.py change; keep it minimal and tag-derived.)
- No runtime delegation to external layout binaries anywhere in the route.
- ASCII only. Clean up /tmp scratch. Check `df -h /` before and after big steps.
