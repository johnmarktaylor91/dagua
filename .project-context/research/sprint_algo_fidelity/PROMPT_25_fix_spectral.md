<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 25 STRAGGLER FIX for **spectral** family (`classic_spectral` vs `nx_spectral`).

## Round 24 measurement

5 graphs, 30 seeds:
- median RMSD: **0.150**
- max RMSD: **0.347** (mixed_width_labels)
- per-graph: linear_3layer_mlp 0.100, mixed_width_labels 0.347, nested_shallow_enc_dec 0.100, parallel_multiedge_bundle 0.111, tl_mlp_3layer 0.089
- Reference (nx_spectral) is DETERMINISTIC (within_dagua and within_target both ~ 0)
- TOST: not_tested because reference has zero variance, but dagua_vs_target is FAR larger than 1e-3 -- DIVERGENT_FROM_DETERMINISTIC_REF

**This is a real algorithmic divergence**, not a stochastic-floor classification artifact.

## Round 22 + 23 prior work

- Round 22 commit `7fc8a7a` -- "spectral -- add NetworkX fidelity mode" (added a fidelity_mode flag)
- Round 23 commits `14743c4`, `6191462` -- "spectral -- finish fidelity gaps", "summary"

**The medians did not budge after these commits** -- meaning fidelity_mode was either not enabled by default or did not capture the right divergence.

## Your job

1. **Confirm whether `classic_spectral` is being run with fidelity_mode=True.** Check `dagua/eval/competitors/classic_competitor.py` for the `classic_spectral` entry. If not, set it.
2. **Line-by-line compare NetworkX `spectral_layout` against dagua spectral**.
   - NetworkX: `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py` (`spectral_layout` function)
   - Dagua: `dagua/layout/ops/pipelines/spectral.py`, `dagua/layout/ops/embed.py` (relevant ops including BuildLaplacian, Pseudoinverse, eigenvalue solve)
3. Likely divergences (per Round 21 diff and general spectral-layout knowledge):
   - **Laplacian variant**: NetworkX uses unnormalized Laplacian for `spectral_layout` by default. Dagua might use symmetric normalized.
   - **Eigenvector selection**: NetworkX takes the 2nd and 3rd smallest non-zero eigenvalues' eigenvectors. Sign conventions may flip.
   - **Eigensolver**: NetworkX uses `numpy.linalg.eigh` (small graphs) or `scipy.sparse.linalg.eigsh` (large). Dagua might use `torch.linalg.eigh` with different LAPACK routine.
   - **Disconnected components**: NetworkX may handle each component separately; dagua may not.
   - **Output normalization**: NetworkX rescales to unit box. Dagua's spectral output scaling may differ.
4. Apply the fix. Verify across all 5 graphs.

## Reference

- Round 21 diff: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_spectral.md`
- Round 22 fix prompt: `PROMPT_22_fix_spectral.md` and corresponding SUMMARY
- Round 23 fix prompt: `PROMPT_23_full_fix_spectral.md` and corresponding SUMMARY
- NetworkX source: `python -c "import networkx; print(networkx.__file__)"` -> sibling dir for `drawing/layout.py`
- Dagua source: `dagua/layout/ops/pipelines/spectral.py`, `dagua/layout/ops/embed.py`

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_spectral nx_spectral \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_25/spectral/{baseline,post_fix}
```

Required: median RMSD reduction >= 0.05 AND all 5 graphs RMSD < 0.05 (since reference is deterministic, parity should be near-bit-exact modulo sign convention).

## Scope constraints

- **DO NOT TOUCH**: `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`, `scripts/build_gallery_audit.py`, `tests/test_render/**`, `.project-context/research/sprint_clusters/**`, `.project-context/research/sprint_graphviz_parity/**`.
- Stage commits with explicit `git add <files>`; NO `git add -A`.
- Commit format: `feat(fidelity): round 25 spectral -- <terse desc>`.

## Tests

- After each commit: `pytest tests/test_layout/ -x --tb=short -q -k "spectral"`
- Final summary: `eval_output/algo_fidelity/round_25/spectral/SUMMARY.md`

</task>

<completeness_contract>
- Reference is deterministic. Target is RMSD < 0.05 on every graph (sign-invariant Procrustes already handles eigenvector flips).
- Either measurable improvement OR principled_residual with clear classification (architectural / data / numerical).
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Only stop for missing details that change correctness, safety, or irreversible actions.
</default_follow_through_policy>
