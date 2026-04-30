<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 20 ADVERSARIAL FIX for **tsnet**.

## SPEC

Your spec is `.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_tsnet.md`.
Read it end-to-end. 7-item ranked fix list.

**Critical context**: the largest divergence (#1) is sklearn uses
**Barnes-Hut sparse nearest-neighbor P** (`method="barnes_hut"`,
default), while dagua uses **dense exact autograd**. This is a
fundamental algorithmic mismatch. Two paths:

(A) **Cheaper path**: change the competitor's TSNE invocation to
`method="exact"` so sklearn computes dense affinities like dagua.
But that's a competitor-side change, NOT an algorithm fidelity
improvement. Document if you go this route.

(B) **Harder path**: implement Barnes-Hut sparse `P` in dagua. Massive
work. Defer.

Recommended: start with the smaller levers (#2, #3, #6, #7) which are
clean alignment, then evaluate whether the Barnes-Hut residual is
acceptable.

Apply top 4 small levers:

1. **Distance-matrix disconnected fill** (`tsne_competitor.py:55-59`
   uses global `max(max_finite * 2, 1)` while dagua's
   `graph_utils.py:301-308` uses per-row `max + 1`).
2. **NumPy-compatible init RNG** (Round 18 already tested this alone;
   add it back as part of a bundle, may help at margins).
3. **Replace `argmin(row)` self-mask** with explicit diagonal masking
   (`tsnet.py:244-245, 274`).
4. **Mirror sklearn convergence controls**: progress check every 50
   iters, `min_grad_norm=1e-7`, `n_iter_without_progress=300` after
   exploration (`_t_sne.py:301-444, 1088-1095`).

## Process

1. Read `ROUND_19_DIFF_tsnet.md` fully.
2. Baseline: 3 seeds × 5 small graphs:
   ```
   python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_20/tsnet/baseline
   ```
3. Apply the 4-lever bundle.
4. Tests: `pytest tests/test_layout/ -x --tb=short -q -k "tsnet"`.
5. Re-measure.
6. COMMIT criterion: median improves >= 0.03 OR aggregate TOST shifts up.
7. Commit `feat(fidelity): round 20 tsnet -- <short>` if met.

## Scope

**Allowed**:
- `dagua/layout/ops/tsnet.py`
- `dagua/layout/ops/pipelines/tsnet.py`
- `dagua/layout/ops/graph_utils.py` -- ONLY the disconnected-fill
  function (around line 301)
- `eval_output/algo_fidelity/round_20/tsnet/**`
- `.project-context/research/sprint_algo_fidelity/ROUND_20_*tsnet*.md`
- `tests/test_layout/test_*tsnet*.py`

**Out of scope**:
- The competitor adapter `dagua/eval/competitors/tsne_competitor.py`
- Barnes-Hut sparse P implementation (Round 21+)
- Other family pipelines

## Verification
- pytest tsnet tests pass
- live_compare runs cleanly
- git diff scope clean

ONE commit only IF improvement.
</task>

<scope_constraints>tsnet files only. graph_utils.py disconnected-fill function only.</scope_constraints>
