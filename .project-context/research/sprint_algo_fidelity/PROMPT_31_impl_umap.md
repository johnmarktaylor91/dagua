<task>
Round 31 IMPLEMENTATION for umap.

Read first:
- eval_output/algo_fidelity/round_31/umap/PLAN_claude.md
- eval_output/algo_fidelity/round_31/umap/PLAN_jtaylor_zmachine_20260524_174616.md
- eval_output/algo_fidelity/round_31/ROUND_31_INTEGRATED_PLAN.md (A3)

## Implement (priority order)

### D1: Per-axis [0,10] rescale (HIGHEST IMPACT, ~0.04-0.08 RMSD)
umap_.py:1188-1192 applies TWO normalizations: `noisy_scale_coords` (max-abs -> 10) then per-axis [0,10] independently.
Dagua does one global min-max into [-10,10] in `dagua/layout/ops/umap.py` finalize section.

### D2: smooth_knn_dist algorithm parity (~30 LoC)
- Reference starts `mid=1.0, lo=0, hi=∞` and doubles when hi=∞
- Dagua pre-doubles `upper`, then bisects starting at upper/2
- Reference clamps `max(sigma, sigma_min)` only at end; dagua clamps inside loop
- MIN_K_DIST_SCALE conditional floor (per-row mean for rho>0, global mean for rho=0)

### D3: Multi-component spectral init (~80-120 LoC)
Reference routes disconnected fuzzy graphs through `multi_component_layout` (spectral.py:145-260). Dagua skips connectivity check.

### D4: Eigensolver mode parity
Always use `eigsh(L, k=3, which="SM", ncv=max(7, sqrt(N)), v0=ones, tol=1e-4)`. Dagua may use dense eigh for N<512 (sign convention drift).

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_umap umap_graph \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/umap/post_impl
```

Expected: median 0.149 -> ~0.05-0.08.

## Scope
- DO NOT TOUCH: render/styles, cluster sprint, existing fidelity_report/benchmark_100seed_final outputs
- Explicit git add. Commit: `fix(layout): round 31 umap -- <terse>`.

## Output
`eval_output/algo_fidelity/round_31/umap/SUMMARY.md` with before/after.
</task>

<completeness_contract>
D1 + D2 minimum. D3 + D4 if cheap. Multi-commit OK.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
