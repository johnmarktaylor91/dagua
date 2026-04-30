<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 27 ADVERSARIAL LINE-BY-LINE diff for **dagua's fdp mode** vs **graphviz fdp binary**.

Phase 1 fdp converged at TOST equivalent_at_0.25x via R5 attempts and R9
seed-fix. Round 21 line-by-lined classic_fmmm vs ogdf_fmmm but the
graphviz-fdp-specific tweaks were not compared.

## Your job

Brutally adversarial line-by-line comparison of dagua's fdp-mode dispatch vs
graphviz lib/fdpgen. graphviz fdp implements FMMM-like multilevel
spring-electrical layout, but with graphviz-specific tweaks distinct from OGDF
FMMM.

## Reference clones

- Graphviz source: `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/`
  - Key files: `fdpinit.c`, `xlayout.c`, `dbg.c`, `tlayout.c`, `clusteredges.c`,
    `layout.c`, `comp.c`
- Dagua fdp dispatch: `dagua/layout/engine.py`
- Dagua impl: `dagua/layout/ops/pipelines/fmmm.py`,
  `dagua/layout/ops/pipelines/dagua_native.py` (fdp-specific tweaks),
  `dagua/layout/ops/fmmm.py`

## Output

Write `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_fdp.md`
with full ranked diff list. Cover: multilevel coarsening strategy
(graphviz uses different coarsening than OGDF), force law parameters
(K, edge length scaling), iteration counts, convergence, port/cluster handling,
spline edge routing if applicable.

Baseline:
```bash
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/fdp/baseline
```

DIFF-ONLY round. No edits.

## Scope constraints

DO NOT TOUCH render/styles.

</task>

<research_mode>
Diagnostic round.
</research_mode>

<default_follow_through_policy>
Skip nothing.
</default_follow_through_policy>
