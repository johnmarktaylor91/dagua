<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 25 STRAGGLER FIX for **umap** family (`classic_umap` vs `umap_graph`).

## Round 24 measurement

3 graphs (out of 5; 2 graphs may have failed cache load), 30 seeds:
- median RMSD: **0.399**
- max RMSD: **0.410**
- per-graph:
  - linear_3layer_mlp: dagua_vs=0.408, within_t=0.203, within_d=0.334, TOST 1x not_equivalent (margin 0.201)
  - mixed_width_labels: dagua_vs=0.410, within_t=0.194, within_d=0.302, TOST 1x not_equivalent (margin 0.186)
  - parallel_multiedge_bundle: dagua_vs=0.379, within_t=0.470, within_d=0.013, TOST 1x equivalent_at_1x (margin 0.392)

UMAP is genuinely stochastic (within_target ~ 0.20-0.47) BUT dagua is much higher than reference variance on 2 of 3 graphs. The reference (umap_graph) is the `umap-learn` package's UMAP layout.

## Round 22 + 23 prior work

- Round 22 / 23 commits: `aac3ba3` umap knn neighborhoods, `1760d31` umap sampling schedule, `465a997` umap weighted distances, `6d52627` umap raw coordinates, `a8c0e72` umap summary
- Round 22 RESIDUAL: `.project-context/research/sprint_algo_fidelity/ROUND_22_RESIDUAL_umap.md`

The Round 22/23 fixes did move umap baselines slightly but the family is still well outside the reference variance.

## Your job

1. **Identify why 2 of 5 graphs are missing** from the bounded comparison (only 3 reported).
   - Check `eval_output/algo_fidelity/round_24/umap/multi_seed_summary.json` for which graphs are present.
   - Likely cause: `umap_graph` cached target positions missing for some graphs. If so, run a one-off live UMAP layout to populate the cache before the comparison.
2. **Read upstream `umap-learn` source** to find the next concrete divergence:
   - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py` (UMAP class) — main `_fit_embed_data` flow
   - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/layouts.py` — `optimize_layout_euclidean` (the actual layout loop)
   - `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/spectral.py` — `spectral_layout` initialization (multi-component)
3. **Common UMAP divergence sources**:
   - Initialization: umap-learn defaults to `init='spectral'` with multi-component handling. Round 21 diff flagged "missing multi-component spectral init".
   - knn graph construction: umap-learn uses `pynndescent`. Dagua may use different knn (sklearn) — check Round 23 commit `aac3ba3`.
   - Edge weight semantics: umap-learn computes "fuzzy simplicial set" with sigma/rho per point.
   - Optimization: umap-learn's negative-sampling in `optimize_layout_euclidean`.
   - Random number generation: umap-learn uses `numpy.random.RandomState` seeded in a specific way.
4. **Pick ONE concrete fix** from the candidate list (the most-likely-impactful per the diff doc). Apply. Re-measure.

## Reference

- Round 21 diff: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_umap.md`
- Round 22 residual: `.project-context/research/sprint_algo_fidelity/ROUND_22_RESIDUAL_umap.md`
- umap-learn source: `python -c "import umap; print(umap.__file__)"`
- Dagua: `dagua/layout/ops/pipelines/umap.py`, `dagua/layout/ops/umap*.py` (find via `grep -l UMAP dagua/layout/ops/*.py`)

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_umap umap_graph \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_25/umap/{baseline,post_fix}
```

Required: at least one of:
- All 5 graphs measurable (you regenerated missing cache entries)
- Per-graph TOST equivalent_at_1x for at least 4/5 graphs
- Median RMSD reduction >= 0.05 across the 3 measurable graphs

## Scope constraints

- **DO NOT TOUCH**: `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`, `scripts/build_gallery_audit.py`, `tests/test_render/**`, `.project-context/research/sprint_clusters/**`, `.project-context/research/sprint_graphviz_parity/**`.
- Stage commits with explicit `git add <files>`; NO `git add -A`.
- Commit format: `feat(fidelity): round 25 umap -- <terse desc>`.

## Tests

- After each commit: `pytest tests/test_layout/ -x --tb=short -q -k "umap"`
- Final summary: `eval_output/algo_fidelity/round_25/umap/SUMMARY.md`

</task>

<completeness_contract>
- Either measurable TOST improvement on >= 1 graph OR principled_residual documentation explaining why upstream UMAP behavior cannot be matched without a wholesale rewrite.
- SUMMARY.md mandatory.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Only stop for missing details that change correctness, safety, or irreversible actions.
</default_follow_through_policy>
