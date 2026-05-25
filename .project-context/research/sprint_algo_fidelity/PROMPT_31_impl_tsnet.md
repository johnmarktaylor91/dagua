<task>
Round 31 IMPLEMENTATION for tsnet.

Read these plans first:
- eval_output/algo_fidelity/round_31/tsnet/PLAN_claude.md
- eval_output/algo_fidelity/round_31/tsnet/PLAN_jtaylor_zmachine_20260524_174631.md
- eval_output/algo_fidelity/round_31/ROUND_31_INTEGRATED_PLAN.md (section A1)

## Implement (in order)

### TIER A1: c=4 gradient scaling (PRIMARY FIX)
sklearn's `_kl_divergence` multiplies the KL gradient by `c = 2*(dof+1)/dof = 4` for 2D embeddings (`_t_sne.py:199, 294`). Dagua's autograd-derived gradient at `dagua/layout/ops/tsnet.py:445-452` misses this multiplier.

Fix: scale the loss by `4.0` before `.backward()`, OR multiply the gradient post-hoc by 4. Either works; pick the cleaner.

### Also from both plans
- RNG init: dagua uses torch.Generator (Philox); sklearn uses NumPy MT. Use np.random.RandomState for init in fidelity_mode.
- Convergence early-stop: sklearn aborts on grad_norm <= 1e-7 or 300 iters without progress.

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/tsnet/post_impl
```

Expected: median RMSD 0.27 -> ~0.10 or lower after c=4 fix.

## Scope
- DO NOT TOUCH: render/styles, cluster sprint files, existing fidelity_report_100seed_final/* or benchmark_100seed_final/* outputs
- Stage with explicit git add. Commit format: `fix(layout): round 31 tsnet -- <terse>`.
- Multiple micro-commits OK.

## Output
`eval_output/algo_fidelity/round_31/tsnet/SUMMARY.md` with before/after medians.
</task>

<completeness_contract>
At minimum implement c=4 fix. Add other items if cheap.
Either measurable improvement OR principled_residual SUMMARY.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
