<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 25 STRAGGLER FIX for **gem** family (`classic_gem` vs `ogdf_gem`).

## Round 24 status

The Round 23 gem codex (per ROUND_21_DIFF_gem.md) attempted to add `fidelity_mode` support including:
- `_glibc_rand_values(seed, count)` — C glibc rand() reproducer
- `_ogdf_runner_initial_positions(num_nodes, seed, device)` — OGDF-style init positions
- `fidelity_mode=True` parameter to `layout_gem_pipeline`

It committed only the consumer-side call (which broke gem entirely with `layout_gem_pipeline() got an unexpected keyword argument 'fidelity_mode'`). I (architect) hot-fixed Round 24 (commit `799454d`) by removing the orphan kwarg call so gem at least runs at baseline. The actual fidelity_mode helpers were NEVER landed.

The orphan test file `tests/test_layout/test_gem_fidelity.py` was deleted by my hotfix.

## Your job

**Properly implement gem `fidelity_mode` from scratch**, including all the helpers the Round 23 codex described.

Files in scope:
- `dagua/layout/ops/gem.py` — add `_glibc_rand_values`, `_ogdf_runner_initial_positions`, hook `fidelity_mode` through `GEMPrepareState` / `InitializeGEMPositions`
- `dagua/layout/ops/pipelines/gem.py` — add `fidelity_mode: bool = False` parameter to `build_gem_pipeline` and `layout_gem_pipeline`
- `dagua/eval/competitors/classic_competitor.py` — restore `default_params={"max_iters": 30_000, "fidelity_mode": True}` AND restore the `fidelity_mode=True` kwarg in the `ClassicGEM.layout()` call (around line 1294-1305)
- `tests/test_layout/test_gem_fidelity.py` — recreate with regression coverage for glibc rand reproducer and OGDF runner init parity

## Reference

- OGDF GEM: `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp`
- OGDF random: `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/Random*.cpp`
- Dagua GEM: `dagua/layout/ops/gem.py`, `dagua/layout/ops/pipelines/gem.py`

The glibc rand() expected behavior at seed=42 (from the deleted test file):
```python
[value % 1000 for value in _glibc_rand_values(seed=42, count=6)] == [166, 740, 881, 241, 12, 758]
```

OGDF uses linear congruential generator equivalent to glibc rand() in the runner: it's a 31-bit LCG with multiplier 1103515245, increment 12345, modulus 2^31 — see `glibc-2.x` rand_r() reference implementation.

OGDF runner init positions for num_nodes=3, seed=42 should be:
```python
[[16.6, 74.0], [88.1, 24.1], [1.2, 75.8]]
```
(glibc rand values modulo 1000 / 10.0 — interleaved x, y per node)

## Verification

Run BEFORE and AFTER:
```bash
python scripts/algo_fidelity_live_compare.py classic_gem ogdf_gem \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_25/gem/{baseline,post_fix}
```

Required:
- baseline runs without crashing (gem fix from R24 hotfix already lets it run)
- post_fix improves median RMSD by >= 0.01 OR worst-graph by >= 0.05
- regression test at `tests/test_layout/test_gem_fidelity.py` passes

## Scope constraints

- **DO NOT TOUCH**: `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`, `scripts/build_gallery_audit.py`, `tests/test_render/**`, `.project-context/research/sprint_clusters/**`, `.project-context/research/sprint_graphviz_parity/**`.
- Stage commits with explicit `git add <files>`; NO `git add -A`.
- Commit format: `feat(fidelity): round 25 gem -- <terse desc>`. Multiple micro-commits OK.

## Tests

- After each commit: `pytest tests/test_layout/ -x --tb=short -q -k "gem"`
- Final summary: `eval_output/algo_fidelity/round_25/gem/SUMMARY.md`

</task>

<completeness_contract>
- Reimplement the dropped fidelity_mode helpers fully with regression test coverage.
- Either measurable improvement OR principled_residual documentation if all OGDF-faithful changes fail to move RMSD.
- SUMMARY.md mandatory.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Only stop for missing details that change correctness, safety, or irreversible actions.
</default_follow_through_policy>
