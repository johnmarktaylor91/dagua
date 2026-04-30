<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 28 STRAGGLER FIX for **dot** (graphviz dot binary).

## Round 27 finding

Source: `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_dot.md`

Baseline (`classic_sugiyama` vs `graphviz_dot`, 5 graphs, 30 seeds):
median 0.006317, worst mixed_width_labels 0.044430. Already very close.

Two R3 follow-up gaps the diff doc identified:
1. `_dot_lattice_lp` (in `dagua/layout/ops/pipelines/dagua_native.py`) computes
   spacing from mean node dimensions instead of point-unit `nodesep`/`ranksep`
   (file:line dagua_native.py:1197, :1199)
2. Various missing graphviz dot features (network simplex, mincross, x-position
   network simplex, clusters, flat/self/multiedge, aspect/ratio scaling) — these
   are wholesale-rewrite scope and out of R28.

## Your job

Fix item 1 only. Make `_dot_lattice_lp` use point-unit nodesep/ranksep
(matching the R3 sugiyama pipeline default) instead of mean node dimensions.

Items 2+ are explicitly out of scope: document as principled_residual
(`large_rewrite_required`).

## Reference

- `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_dot.md`
- `dagua/layout/ops/pipelines/dagua_native.py` (around line 1056-1262
  for `_dot_lattice_lp` function and helpers)
- `dagua/layout/ops/pipelines/sugiyama.py` (already R3-correct for spacing)

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_28/dot/{baseline,post_fix}
```

Required: improvement on mixed_width_labels (current 0.044 max) OR no
regression on the others. If item 1 doesn't move the metric (because
lattice path may not be exercised on the bounded benchmark subset), document
that and accept as residual.

## Scope

- DO NOT TOUCH render/styles, cluster sprint files
- Stage commits explicitly. Commit format `feat(fidelity): round 28 dot -- <terse>`
- After commit: `pytest tests/test_layout/ -x --tb=short -q -k "sugiyama or dot"`

## Output

Per-round SUMMARY at `eval_output/algo_fidelity/round_28/dot/SUMMARY.md`.

</task>

<completeness_contract>
Apply item 1 if line-local; otherwise document. Wholesale rewrite explicitly
out of R28 scope.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
