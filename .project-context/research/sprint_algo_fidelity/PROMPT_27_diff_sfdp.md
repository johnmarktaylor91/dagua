<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 27 ADVERSARIAL LINE-BY-LINE diff for **classic_sfdp** vs **graphviz_sfdp**.

This pair was NEVER line-by-lined in Rounds 19/21 (those covered 21 other families).
Phase 1 sfdp converged to TOST equivalent_at_1x via the R9 graphviz seed-fix
without line-by-line code comparison. Now we go through it carefully.

## Your job

Brutally adversarial line-by-line comparison of dagua's sfdp implementation
against graphviz's sfdpgen reference C source. Produce a ranked list of every
divergence, no matter how small.

## Reference clones

- Graphviz source: `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/`
  - Key files: `spring_electrical.c`, `sparse_solve.c`, `Multilevel.c`,
    `post_process.c`, `QuadTree.c`, `stress_model.c`
- Dagua sfdp: `dagua/layout/ops/pipelines/sfdp.py` and dependencies (likely
  references ops in `dagua/layout/ops/`).
- Dagua dispatch: `dagua/layout/ops/pipelines/__init__.py` line 71 maps "sfdp"
  to `layout_sfdp_pipeline`.

## Output

Write `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_sfdp.md`
with:
- Per-section ranked list of every divergence with file:line on both sides
- Concrete categorical labels per item (algorithm-correctness / numerical /
  parameter-default / convention / scaffolding)
- Estimated fix size (lines net) and risk level
- Distinguish: items already addressed in earlier rounds, items missed entirely

Then run:
```bash
python scripts/algo_fidelity_live_compare.py classic_sfdp graphviz_sfdp \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/sfdp/baseline
```

Record baseline RMSD in your diff doc. **DO NOT apply fixes in this round** —
just produce the ranked diff and the baseline measurement. Round 28 will fix.

## Scope constraints

- DO NOT TOUCH: `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`, `scripts/build_gallery_audit.py`, `tests/test_render/**`, `.project-context/research/sprint_clusters/**`, `.project-context/research/sprint_graphviz_parity/**`.
- This round is DIFF-ONLY. No code edits, no commits — only the diff doc + baseline measurement.

</task>

<research_mode>
This is a diagnostic round. Your output is the ranked diff document, not code.
</research_mode>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Skip nothing.
</default_follow_through_policy>
