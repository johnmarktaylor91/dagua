<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Fix the "element 0 of tensors does not require grad and does not have a grad_fn"
errors in classic_neulay and classic_tsnet variants. Errors fire at
~0.05s runtime (initialization), not mid-layout.

## Evidence

664 errors total in `eval_output/benchmark_100seed_final/results.json`:

| Engine variant | error count |
|---|---:|
| classic_neulay_no_gcn | 99 |
| classic_neulay_radius08 | 93 |
| classic_neulay_radius02 | 88 |
| classic_neulay_lr05 | 60 |
| classic_neulay_lr001 | 39 |
| classic_neulay_default | 24 |
| classic_tsnet_steps200 | 69 |
| classic_tsnet_steps2000 | 69 |
| classic_tsnet_default | 42 |
| classic_tsnet_perp50 | 42 |
| classic_tsnet_perp5 | 39 |

All neulay variants (6/6 affected) and all tsnet variants (5/5 affected).

Top graphs affected: small/medium graphs like sbm_4x30 (120 nodes),
real_football_115 (115 nodes), scale_free_ba_120 (120 nodes),
regular_4_40 (40 nodes), triangular_lattice_36 (36 nodes).

The error means: `loss.backward()` is called but the loss tensor wasn't
produced through an autograd-tracked path. Either:
- The initial positions tensor has `requires_grad=False` and isn't being made
  trainable before the loss computation
- The loss function returns a constant (e.g., everything is masked out) so the
  graph is empty
- Some forward path detaches the tensor early

## Your job

1. Reproduce on a 36-node graph (e.g., `triangular_lattice_36`):
   ```python
   from dagua.eval.graphs import get_test_graphs
   from dagua.eval.competitors import get_competitor
   graphs = {tg.name: tg for tg in get_test_graphs()}
   tg = graphs['triangular_lattice_36']
   c = get_competitor('classic_neulay_default')
   result = c.layout(tg.graph, seed=42)
   ```

2. Trace through `dagua/layout/ops/pipelines/neulay.py` and
   `dagua/layout/ops/pipelines/tsnet.py` to find where the autograd path
   breaks. Common patterns:
   - `state.pos = state.pos.detach()` somewhere it shouldn't be
   - `state.pos = torch.tensor(...)` instead of `state.pos.clone()`
   - Init op produces a tensor without `requires_grad=True`
   - Loss is computed from tensors that don't share graph history

3. Fix per affected pipeline. Likely a 5-20 line fix per pipeline.

4. Add regression test in tests/test_layout/ for both neulay and tsnet on a
   small graph confirming `loss.backward()` succeeds.

## Scope

- DO NOT TOUCH: render/styles, cluster sprint files, benchmark scripts/output.
- Stage commits with explicit `git add <paths>`.
- Commit format: `fix(layout): neulay/tsnet -- restore autograd path on small graphs`

## Verification

- `pytest tests/test_layout/ -x --tb=short -q -k "neulay or tsnet"`
- Call `classic_neulay_default.layout()` and `classic_tsnet_default.layout()`
  on triangular_lattice_36 with seed=42. Confirm no error, position tensor
  returned with right shape.

## Output

`eval_output/algo_fidelity/round_30/neulay_tsnet_grad_fn/SUMMARY.md` with
root cause file:line and fix description.

</task>

<completeness_contract>
Fix both engines OR document why each is unfixable with a principled reason.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
