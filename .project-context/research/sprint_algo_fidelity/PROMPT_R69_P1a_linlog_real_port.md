<task>
Replace dagua's linlog fidelity path -- which currently DELEGATES to the reference --
with a genuine, independent pure-Python/PyTorch reimplementation living in the pipeline.

## The problem (delegation -- a project-critical anti-pattern)
`dagua/layout/ops/pipelines/linlog.py` lines ~165-183 contain:

    if fidelity_mode:
        from dagua.eval.competitors.linlog_competitor import (
            _layout_linlog_reference, _resolve_config,
        )
        ...
        return _layout_linlog_reference(graph=graph, config=config, seed=seed).to(...)

This makes dagua's "fidelity" output LITERALLY the reference's output -> any bit-exact
match is a meaningless tautology. dagua/layout/ops/pipelines/*.py must NEVER import or
call dagua/eval/competitors/* at runtime. (This is incident-pattern #2 of a documented
project rule; 5 prior delegation incidents.)

## What "the reference" is (good news: it's pure Python/torch, same language)
`dagua/eval/competitors/linlog_competitor.py` implements a clean paper-spec LinLog
(Andreas Noack) force-directed solver in PURE torch -- NO external library, NO binary.
The canonical entry point is `_layout_linlog_reference` (linlog_competitor.py:721-810).
Its algorithm, per step (1..config.steps):
  1. `_scheduled_exponents(step, config)` -> (attr_exponent, repu_exponent)
  2. `_energy_factors(...)` -> (repu_factor, scaled_gravity)
  3. zero `forces` [N,2] and `curvature` [N]
  4. `_weighted_barycenter(positions, repulsion_weights)`
  5. repulsion: `_add_barnes_hut_repulsion` if num_nodes > config.barnes_hut_threshold
     else `_add_exact_repulsion` (accumulates into forces + curvature)
  6. `_add_attraction(...)`, `_add_gravity(...)`
  7. `steps = forces / curvature.clamp(min=_MIN_DISTANCE)[:,None]`; cap by
     `_average_distances`; cooling = `1 - 0.95*(step-1)/max(steps,1)`; `positions += steps*cooling`
  8. final `_normalize_positions(positions)`
Setup before the loop: `_prepare_edges`, `_node_repulsion_weights`, `_initial_positions(num_nodes, seed)`.
You MUST read ALL of linlog_competitor.py (helpers `_prepare_edges`, `_node_repulsion_weights`,
`_initial_positions`, `_scheduled_exponents`, `_energy_factors`, `_weighted_barycenter`,
`_add_barnes_hut_repulsion`, `_add_exact_repulsion`, `_add_attraction`, `_add_gravity`,
`_average_distances`, `_normalize_positions`, constants `_MIN_DISTANCE`, `_LinLogConfig`,
`_resolve_config`) to port the EXACT arithmetic and ordering.

## The job
1. Reimplement the LinLog solver INSIDE the layout layer as dagua's own code (in
   `dagua/layout/ops/pipelines/linlog.py`, and/or new ops under `dagua/layout/ops/`),
   reproducing `_layout_linlog_reference` step-for-step and op-for-op so the output is
   BIT-EXACT (it is same-language torch, so identical torch ops in identical order ->
   identical floats). Match: RNG/initial-position generation, force & curvature
   accumulation ORDER, Barnes-Hut tree construction + traversal order, the clamp/cap/
   cooling math, and final normalization.
2. Route the pipeline's `fidelity_mode` (truthy / "ogdf" / whatever the linlog variants
   will use) to THIS new in-pipeline implementation. REMOVE the
   `from dagua.eval.competitors...import` block entirely.
3. Leave the existing NATIVE (non-fidelity) `LayoutProblem` torch path untouched.
4. Prefer composing from registered ops per dagua's architecture (principle #8), but a
   self-contained pipeline implementation is ACCEPTABLE if op-composition can't match the
   reference arithmetic. Add WHY comments explaining the sequential/scalar choices made
   for bit-exactness.
</task>

<constraints>
- PURE Python/PyTorch. No `subprocess`, no external binary, no new third-party dep.
  (Common libs numpy/scipy/torch are fine; linlog needs none beyond torch.)
- ABSOLUTELY NO runtime import of `dagua.eval.competitors.*` (or any reference adapter)
  from `dagua/layout/ops/`. The whole point is an INDEPENDENT implementation. Copy/adapt
  the algorithm, do not import it.
- Do not change `dagua/eval/competitors/linlog_competitor.py` (it stays the reference).
- Do not change the public `LayoutConfig`/pipeline API surface.
</constraints>

<verification_loop>
1. `python -c "import dagua.layout.ops.pipelines.linlog"` clean.
2. Confirm delegation gone:
   `grep -n "eval.competitors\|_layout_linlog_reference" dagua/layout/ops/pipelines/linlog.py`
   -> EMPTY.
3. BIT-EXACT parity test (write it as a throwaway script under /tmp and run it; you may
   also leave a permanent test under tests/): for seeds [42,43,44], for ~5 graphs spanning
   small (exact repulsion) AND large enough to trigger Barnes-Hut (> barnes_hut_threshold),
   weighted and unweighted: compute positions via (a) the new pipeline fidelity path and
   (b) `_layout_linlog_reference`. Assert `torch.allclose(a, b, atol=1e-6, rtol=0)` AND
   Procrustes RMSD < 1e-6. Report the max abs diff and max RMSD observed.
4. If not bit-exact, trace the first diverging step (compare forces/curvature after step 1)
   and fix the arithmetic-order mismatch. Iterate until < 1e-6.
</verification_loop>

<output>
1. Edit `dagua/layout/ops/pipelines/linlog.py` (and any new ops files).
2. Add a permanent parity test (e.g. `tests/test_pipeline_linlog_fidelity_parity.py`)
   asserting bit-exact vs `_layout_linlog_reference` across seeds/sizes.
3. Write a short report to `eval_output/fidelity_report_r69/p1a_linlog_port.md`:
   what was ported, the op/arithmetic-order decisions, and the measured max abs diff /
   max Procrustes RMSD from the parity test.
4. Commit to current branch `develop`:
   `feat(layout): R69 P1a -- real in-pipeline linlog port (remove reference delegation)`
</output>

<default_follow_through_policy>
Proceed autonomously. The reference is dagua-authored same-language torch, so bit-exact
is achievable -- chase it to < 1e-6. Only STOP and report if you discover the reference
itself depends on something non-deterministic or external that cannot be reproduced in
pure torch (it should not). Do NOT "solve" parity by importing the reference -- that is
the exact anti-pattern being removed.
</default_follow_through_policy>
