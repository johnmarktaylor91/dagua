# Wave 2 Exemplar: FR Pipeline

## Codex Prompt (ready to dispatch)

```xml
<task>
REPO: dagua (GPU-accelerated differentiable graph layout engine built on PyTorch).
BRANCH: feat/bench-and-aesthetics
FILES TO READ FIRST: CLAUDE.md, dagua/layout/AGENTS.md, .project-context/conventions.md

GOAL: Create the first Wave 2 algorithm pipeline -- Fruchterman-Reingold expressed
as a Pipeline of composable ops. This is the exemplar that validates the pattern
before we translate the remaining 23 algorithms.

FIDELITY REQUIREMENT: The pipeline MUST produce BIT-IDENTICAL output to
dagua/layout/classic/fr.py::layout_fr() given the same inputs and seed.
Not "close". Not "statistically equivalent". torch.equal(classic, pipeline) == True.

## What exists

Wave 1 delivered 140 composable ops in dagua/layout/ops/ with Pipeline, Repeat,
Conditional, and EarlyBreak composition primitives in base.py. The ops framework
has LayoutProblem (immutable inputs), SolveState (mutable working state), and
RuntimeContext (execution infrastructure).

The classic FR implementation is in dagua/layout/classic/fr.py (277 lines).
Read it FIRST -- it is the ground truth.

## Deliverables

### 1. Directory + pipeline file: dagua/layout/ops/pipelines/fr.py

Create dagua/layout/ops/pipelines/__init__.py (empty or minimal).
Create dagua/layout/ops/pipelines/fr.py containing:

```python
def build_fr_pipeline(steps: int = 50) -> Pipeline:
    """Build an FR pipeline that is bit-identical to classic/fr.py."""
    ...
```

This function returns a Pipeline of ops that, when run on properly prepared
LayoutProblem + SolveState + RuntimeContext, produces the same output as
layout_fr().

### 2. Glue ops (new, minimal)

The following gaps exist between classic FR and the current op library.
Create these in the appropriate existing op files (NOT in the pipeline file):

**Gap A -- Temperature initialization (dagua/layout/ops/anneal.py):**
FR computes: `temperature = max(x_extent, y_extent) * 0.1`
No such op exists. Create:
```python
@dataclass(frozen=True)
class InitTemperatureFromExtentConfig:
    scale: float = 0.1  # multiplier on max extent

class InitTemperatureFromExtent(Op):
    """Set state.temperature from the bounding-box extent of state.pos."""
```

**Gap B -- Combined FR force step (dagua/layout/ops/force.py):**
Classic FR computes repulsion + attraction in ONE combined einsum over the
full NxN delta matrix:
```python
displacement = einsum("ijk,ij->ik", delta, (k^2/d^2 - A*d/k))
```
The existing InverseDistanceRepulsion + UniformSpringAttraction compute these
separately (different float accumulation order -> not bit-identical).

Create a combined op:
```python
class FRCombinedForce(Op):
    """Combined FR repulsion+attraction matching classic/fr.py's einsum."""
```
This op MUST reproduce the EXACT computation from classic/fr.py lines 249-257:
- delta = pos[:, newaxis, :] - pos[newaxis, :, :]
- distance = norm(delta, dim=-1), clamped to min=0.01
- displacement = einsum("ijk,ij->ik", delta, (k*k/d^2 - A*d/k))
- Store displacement in state.forces

It needs state.extras["fr_adjacency"] (the dense NxN adjacency built during setup).

**Gap C -- FR convergence check (dagua/layout/ops/converge.py):**
Classic FR: `torch.linalg.norm(delta_pos) / num_nodes < 1e-4`
This is Frobenius norm of the full displacement matrix divided by N.
The existing DisplacementThreshold uses mean-of-per-node-L2-norms (different formula).

Create:
```python
@dataclass(frozen=True)
class FRConvergenceCheckConfig:
    threshold: float = 1e-4

class FRConvergenceCheck(Op):
    """FR-specific convergence: frobenius_norm(delta_pos) / N < threshold."""
```
This needs the actual delta_pos from the force step. Store it:
after ApplyDisplacement runs, it should store delta_pos in state.extras["last_delta_pos"].
OR: compute convergence from pos vs prev_pos (same result since delta_pos = pos_new - pos_old).
Choose whichever is simpler to implement correctly.

### 3. Adapter function: dagua/layout/ops/pipelines/fr.py

```python
def layout_fr_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop-in replacement for classic/fr.py::layout_fr().

    Same signature (minus trace_every, pos, area -- those are compatibility params).
    Returns positions with identical dtype, device, and values.
    """
```

This adapter:
1. Constructs LayoutProblem from the raw tensors
2. Creates initial SolveState (empty)
3. Creates RuntimeContext with seed
4. Builds the FR pipeline via build_fr_pipeline(steps)
5. Runs pipeline.apply(problem, state, ctx)
6. Extracts and returns final positions

### 4. Fidelity test: tests/test_pipeline_fr.py

```python
class TestFRPipelineFidelity:
    """Bit-identical fidelity tests for the FR pipeline vs classic."""

    @pytest.mark.parametrize("num_nodes,seed", [
        (0, 42),      # empty graph
        (1, 42),      # single node
        (2, 42),      # single edge
        (5, 42),      # small graph
        (5, 99),      # different seed
        (20, 42),     # medium
        (50, 7),      # larger
    ])
    def test_bit_identical(self, num_nodes, seed):
        """Pipeline must produce torch.equal() output vs classic."""
        # Build a simple graph (chain or random edges)
        # Run classic layout_fr()
        # Run pipeline layout_fr_pipeline()
        # assert torch.equal(classic_pos, pipeline_pos)

    def test_with_edge_weights(self): ...
    def test_disconnected_graph(self): ...
    def test_complete_graph(self): ...
```

Use various graph topologies: chain, complete, star, disconnected components,
self-loops, empty edge set. All must be bit-identical.

## Critical implementation notes

1. **dtype MUST be float64 throughout.** Classic FR runs entirely in float64.
   The pipeline must preserve this. RandomUniformInit with rng_backend="numpy"
   returns float64 from np.random.RandomState. All subsequent ops must NOT
   downcast to float32 until the final postprocess step (cast to float32 at the end,
   matching classic's `scaled.to(dtype=torch.float32, device=device)`).

2. **The dense adjacency matrix** is built once in classic FR and reused every
   iteration. In the pipeline, build it during setup and store in
   state.extras["fr_adjacency"]. Use BuildAdjacency or replicate the exact
   classic _adjacency_matrix() logic. The adjacency must be float64 and directed
   (not symmetrized).

3. **Postprocessing** (classic lines 270-272):
   ```python
   scaled = _rescale_layout(positions, scale=1.0)  # center + normalize to max_abs=1
   scaled = scaled * (sqrt(max(num_nodes, 1)) * 50.0)  # output scale
   final = scaled.to(dtype=torch.float32, device=device)
   ```
   Use CenterPositions + ScalePositions(method="max_abs", factor=sqrt(N)*50) if they
   produce the same result. If not, implement the exact 2-step postprocess.

4. **optimal_distance = sqrt(1.0 / max(N, 1))** -- this is what _resolve_area_k
   computes when area=1.0 (unit square). Verify the pipeline's area resolves to 1.0.
   If state.extras["force_area"] needs to be set, set it in the adapter.

5. **cooling_step = temperature / (steps + 1)** where temperature is the initial
   temperature. LinearCool with rate=None stores the initial temperature on first
   call and computes rate = initial / (total_steps + 1). Verify total_steps is
   set correctly in state before the Repeat loop.

6. **DO NOT modify any existing op files' public behavior.** New ops and new config
   options are fine. Do not break existing tests.

7. **DO NOT modify dagua/layout/classic/fr.py.** It is the reference.

## File inventory (what you will create/modify)

CREATE:
- dagua/layout/ops/pipelines/__init__.py
- dagua/layout/ops/pipelines/fr.py
- tests/test_pipeline_fr.py

MODIFY (add new ops only, do not change existing ops):
- dagua/layout/ops/anneal.py (add InitTemperatureFromExtent)
- dagua/layout/ops/force.py (add FRCombinedForce)
- dagua/layout/ops/converge.py (add FRConvergenceCheck)

NO OTHER FILES should be modified.

## Verification

Run after implementation:
```bash
pytest tests/test_pipeline_fr.py -x -v --tb=long
pytest tests/test_ops_anneal.py tests/test_ops_force.py tests/test_ops_converge.py -x --tb=short
pytest tests/ -x --tb=short -q  # full suite, nothing broken
```

ALL tests must pass. The fidelity tests must use torch.equal() (exact match),
not torch.allclose().
</task>

<completeness_contract>
Resolve the task fully before stopping.
The pipeline must pass ALL fidelity tests with torch.equal().
If a fidelity test fails, diagnose the exact floating-point divergence point
(which iteration, which computation), fix the pipeline, and re-run.
Do not stop at "close enough" -- the requirement is bit-identical.
Check edge cases: 0 nodes, 1 node, disconnected graphs, weighted edges.
</completeness_contract>

<verification_loop>
After implementing:
1. Run the fidelity tests. If ANY fail, trace the divergence:
   - Print positions after each iteration for both classic and pipeline
   - Find the first iteration where they diverge
   - Identify which op produces the wrong value
   - Fix and re-run
2. Run the existing op tests to verify no regressions.
3. Run the full test suite.
Do not declare done until all three pass.
</verification_loop>

<missing_context_gating>
Do not guess about op behavior. Read the actual source code of every op you use
in the pipeline (especially RandomUniformInit, BuildAdjacency, LinearCool,
ApplyDisplacement, CenterPositions, ScalePositions). Verify each matches the
corresponding classic FR step.

If an op has a config option that MIGHT affect the output (dtype, device,
scaling, clamping constants), read the implementation to confirm the right setting.
</missing_context_gating>

<action_safety>
Keep changes tightly scoped:
- Only create the 3 new files and add ops to the 3 existing files
- Do not refactor existing ops
- Do not modify classic/fr.py
- Do not add type stubs, docstring rewrites, or formatting changes to existing code
- Register new ops with @register_op following existing patterns
</action_safety>

<default_follow_through_policy>
Default to the most reasonable low-risk interpretation and keep going.
Only stop for missing details that change correctness, safety, or irreversible actions.
If you discover an additional fidelity gap not listed above, fix it -- do not stop to ask.
</default_follow_through_policy>
```

## Post-dispatch checklist

- [ ] Verify no file conflicts with other workers (this is solo)
- [ ] After completion: run `pytest tests/test_pipeline_fr.py -x -v`
- [ ] After completion: run full suite `pytest tests/ -x --tb=short -q`
- [ ] Review the diff for scope creep
- [ ] If fidelity fails, check float64 preservation and einsum equivalence first
