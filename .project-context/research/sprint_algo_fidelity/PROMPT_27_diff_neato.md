<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 27 ADVERSARIAL LINE-BY-LINE diff for **dagua's neato mode** vs **graphviz neato binary**.

Phase 1 neato converged at TOST equivalent_at_0.5x. The dispatch path was never
line-by-lined against graphviz lib/neatogen.

## Your job

Brutally adversarial line-by-line comparison of dagua's neato-mode dispatch vs
graphviz lib/neatogen. dagua's neato is implemented via dispatch to either
`stress_majorization` or `classical_mds` pipelines (already line-by-lined in
Round 21 vs ogdf_stress / igraph_mds, but the graphviz-neato-specific tweaks
in dagua_native.py have not been compared).

## Reference clones

- Graphviz source: `/home/jtaylor/projects/_references/graphviz/lib/neatogen/`
  - Key files: `neato.c`, `stress.c`, `bfs.c`, `kkutils.c`, `pca.c`,
    `quad_prog_solver.c`, `solve.c`, `multispline.c`
- Dagua neato dispatch: `dagua/layout/engine.py` (find how `algorithm="neato"`
  routes; likely to stress_majorization or classical_mds)
- Dagua impl: `dagua/layout/ops/pipelines/stress_majorization.py`,
  `dagua/layout/ops/pipelines/classical_mds.py`,
  `dagua/layout/ops/pipelines/dagua_native.py` (neato-specific tweaks)

## Output

Write `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_neato.md`
with:
- Per-section ranked list of every divergence with file:line on both sides
- Cover: model selection (stress vs MDS vs Kamada-Kawai-like),
  initialization (PCA?), iteration count, convergence criteria, edge length
  targeting, weight function, post-processing
- Concrete categorical labels and estimated fix sizes

Then baseline:
```bash
python scripts/algo_fidelity_live_compare.py classic_stress_maj graphviz_neato \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/neato/baseline
```

Note cache availability for graphviz_neato. **DIFF-ONLY**, no code edits, no
commits.

## Scope constraints

DO NOT TOUCH render/styles. DIFF-ONLY round.

</task>

<research_mode>
Diagnostic round.
</research_mode>

<default_follow_through_policy>
Skip nothing.
</default_follow_through_policy>
