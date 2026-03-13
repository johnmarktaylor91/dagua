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
