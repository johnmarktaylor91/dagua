# Dagua Architectural Decisions

## [2024] Headless Layout Engine
Context: Should the layout engine take Graph objects or raw tensors?
Decision: Headless — engine takes `edge_index`, `node_sizes`, `groups` as tensors.
Rationale: Makes the engine independently testable, reusable without the Graph abstraction, and avoids circular deps between graph.py and layout/.
Alternatives considered: Engine takes Graph directly (simpler API, but creates tight coupling and makes the engine untestable without the full graph stack).

## [2024] Constraints as Composable Loss Callables
Context: How should layout aesthetics be expressed?
Decision: Each constraint is `(pos, graph_data) -> scalar loss`. Users compose them freely.
Rationale: Leverages PyTorch autograd. Users can write custom constraints in ~3 lines. No need for a constraint registry or plugin system.
Alternatives considered: Constraint classes with register/unregister (heavier API), fixed constraint set (less flexible).

## [2024] 5-Level Style Cascade
Context: How should styles be resolved when multiple sources conflict?
Decision: per-element > cluster member style > theme type > graph default > global default.
Rationale: Matches CSS-like specificity intuitions. Lets users set broad themes while overriding individual elements.
Alternatives considered: Flat style dict (simple but no cascade), 3-level (no cluster or global).

## [2024] Bottom-Up Default Direction
Context: Which direction should DAG flow by default?
Decision: Bottom-up (y increases upward), matching DNN forward pass convention.
Rationale: Primary use case is neural network visualization via TorchLens.
Alternatives considered: Top-down (Graphviz default), left-to-right.

## [2024] Deterministic by Default
Context: Should layout be deterministic across runs?
Decision: `seed=42` by default, `seed=None` to opt out.
Rationale: Reproducibility matters for documentation, testing, and debugging.
Alternatives considered: Random by default (more exploration), no seed control.

## [2024] Flex System for Soft Layout Targets
Context: How to express layout preferences (spacing, alignment, pinning)?
Decision: Flex system — `Flex.soft(40)`, `Flex.firm(40)`, `Flex.locked(0)`. Soft targets are loss terms, hard targets use projection.
Rationale: Unified interface for preferences of varying strength. Differentiable when soft, exact when locked.
Alternatives considered: Separate pin/align/spacing APIs (more discoverable but fragmented).

## [2024] IO in io.py, Exposed as Graph Classmethods
Context: Where should serialization and interop live?
Decision: Standalone functions in `io.py`, thin `Graph.from_*` classmethod wrappers.
Rationale: Keeps graph.py focused on orchestration, keeps io.py independently testable.
Alternatives considered: Methods directly on Graph (simpler but bloats graph.py).

## [2026-03] Auto Text Backgrounds for Patterned Fills
Context: Patterned node fills reduced label readability in gallery cards.
Decision: Use a white text background with pattern-specific opacity instead of text outlines or external labels.
Rationale: Text outlines looked blurry, and external labels changed layout. A white background with tuned opacity keeps the fill visible while restoring readability.
Alternatives considered: Text outline/stroke (blurred at small sizes), external labels (layout changes), fully opaque label boxes (hid too much of the fill).

## [2026-03] Synthetic Italic via Affine Shear
Context: Linux font resolution often lacks a distinct italic file for the chosen sans-serif stack.
Decision: Apply a synthetic affine shear when a native italic font variant is unavailable.
Rationale: Shear works with any resolved font, keeps appearance consistent across environments, and does not require shipping extra font files. The chosen 15-degree angle is visible without looking exaggerated.
Alternatives considered: Font substitution (environment-dependent and inconsistent), skipping italic styling when unavailable (visual regression).

## [2026-03] 8-Face Hub Distribution over Full Perimeter
Context: Multiple inbound arrowheads stacked on the same node face in moderate hub cases.
Decision: Bucket terminal placement across 8 discrete node faces instead of using continuous angular distribution.
Rationale: This was the most conservative improvement that did not require changing the edge router. It handles moderate hubs (roughly 3-6 edges) well enough while keeping routing behavior stable.
Alternatives considered: Continuous perimeter distribution (more flexible but coupled to router changes), router-aware fanout (better for extreme hubs but much larger scope).

## [2026-03] Curvature-Adaptive Dashing
Context: Border dashes visually merged on tight curves.
Decision: Scale dash on-lengths with curvature while keeping gap lengths constant, with a minimum scale floor of 0.4.
Rationale: Constant gaps preserve visual density, while shorter on-segments prevent neighboring dashes from merging on high-curvature sections. The 0.4 floor avoids over-shortening into visual noise.
Alternatives considered: Scaling both dashes and gaps (density drift), fixed dash pattern everywhere (merging on tight curves), unconstrained scaling (dashes disappear on extreme curvature).

## Composable Ops Wave 1 Decisions (2026-04-02)

1. **RNG backend per algorithm**: Ops match the RNG backend of their classic/ source
   (torch.Generator, numpy.random, Python random.Random). NOT unified to torch only.
   Why: Classic algos use 5+ different RNG families. Fidelity requires matching each.

2. **Force pipeline pattern**: ZeroForces -> [accumulators] -> ApplyDisplacement.
   GEMNodeTick is the exception (sequential single-node Gauss-Seidel).
   Why: Force-directed algos accumulate forces then apply; GEM processes one node at a time.

3. **loss.py split into two files**: loss_engine.py (16 ops) + loss_classic.py (12 ops).
   Why: Too large for one Codex agent; no shared helpers needed between the two.

4. **Execution strategies NOT ops**: Tiled GPU, subset GPU, hybrid CPU/GPU stay in engine.py.
   Why: These are engine infrastructure, not algorithm logic. Ops must be callable FROM them.

5. **Multiple optimizers via keyed storage**: CreateOptimizer(key="detector") stores in
   extras["optimizer_detector"]. Default key uses state.optimizer.
   Why: SGD2 needs pos optimizer + crossing detector optimizer simultaneously.

6. **Algorithm-specific state in extras dict**: "algo_field" convention (e.g. "tsne_gains").
   reads/writes metadata is advisory only for extras keys.
   Why: Adding 50+ typed fields to SolveState would be worse than the escape hatch.

## Composable Ops Completion (2026-04-04)

7. **All 23 algorithms decomposed into ops pipelines**: Wave 2 (all algorithms) and Wave 3
   (migration + fidelity validation) completed. 268 ops, 34 modules, 23 pipelines. All
   pipeline fidelity tests green (bit-identical to archive reimplementations).

8. **algorithm_params in LayoutConfig**: `LayoutConfig(algorithm="fr", algorithm_params={"cooling": 0.95})`
   passes kwargs through to the pipeline function's frozen config dataclass. Validated
   via integration tests. Keeps LayoutConfig stable while pipelines grow independently.
   Why: Adding per-algorithm fields to LayoutConfig would explode its surface area.

9. **Archive as reference, not compatibility layer**: `_archive/classic/` is reference-only.
   No backward-compat symlinks or re-exports. Tests import from pipelines, not archive.
   Why: Clean break. Maintaining compatibility would constrain ops evolution.
