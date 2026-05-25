<task>
R33 IMPLEMENTATION for umap leftovers (D5+D6+D9 from R31 Claude PLAN).

R31/R32 umap codex implemented D1+D2+D3+D4 (per-axis rescale, smooth_knn_dist, multi-component spectral, ARPACK). Remaining items NOT implemented:

## D5: Eigensolver determinism (~10 LoC)
Reference always uses `eigsh(L, k=3, which="SM", ncv=max(7,√N), v0=ones, tol=1e-4, maxiter=5N)`. Dagua may use dense eigh for some sizes. Force ARPACK always with these exact params (per `umap-learn umap_.py`).

## D6: tau_rand_int per-source RNG (~30 LoC)
Reference seeds a Tausworthe state per source using `embedding[:,0].view(int64)`. Dagua uses a single global `torch.Generator`. Per-source state means each thread/source produces independent bit-exact streams matching umap-learn's negative-sampling.

Sketch (per Claude R31 plan):
```python
# Per-source state: 3-int Tausworthe seeded by embedding row first-dim bits cast to int64
def _make_tau_state(emb_rows):
    return jax_style_tausworthe_state(emb_rows[:, 0].view(torch.int64))

def _tau_rand_int(state, lo, hi):
    # Tausworthe combined generator -- bit-exact match to reference
```

## D9: find_ab_params p0 default (~3 LoC)
Reference uses scipy default `p0=(1, 1)`. Dagua uses `p0=(1.93, 0.79)`. Matters for `mindist05` / `spread2` variants.

## Reference
- /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py
- /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/layouts.py (`tau_rand_int`)
- dagua/layout/ops/umap.py + dagua/layout/ops/pipelines/umap_layout.py

## Implementation

Use commit-safe wrapper: `bash scripts/commit-safe.sh -m "..."`

Apply items in order. Re-measure after each on bounded subset:
```bash
python scripts/algo_fidelity_live_compare.py classic_umap umap_graph --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,asymmetric_hourglass_hub,small_world_100,scale_free_ba_120 --output-dir eval_output/algo_fidelity/round_33/umap_leftovers/<phase>
```

(Note: extended bounded subset includes medium graphs per `scripts/larger_subset_verify.sh`.)

## Output
`eval_output/algo_fidelity/round_33/umap_leftovers/SUMMARY.md` with per-item before/after.
</task>

<completeness_contract>
Apply D5 + D6 + D9. If any regresses, revert that one and document.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
