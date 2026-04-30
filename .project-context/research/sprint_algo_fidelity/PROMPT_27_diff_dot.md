<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 27 ADVERSARIAL LINE-BY-LINE diff for **dagua's dot mode** vs **graphviz dot binary**.

Phase 1 dot converged at median RMSD 0.019 via the R3 sugiyama point-spacing
fix. But the underlying dot-mode dispatch in `dagua/layout/ops/pipelines/dagua_native.py`
(3213 lines) was never line-by-lined against graphviz lib/dotgen.

## Your job

Brutally adversarial line-by-line comparison of dagua's dot-mode pipeline path
vs graphviz lib/dotgen. dagua_native.py contains specific replicators
(`_dot_lattice_lp`, `_should_dot_lattice_lp`, `_lattice_uniform_centered_slots`,
`_back_edge_relayer`, etc.) and a docstring at line 1061 says "Replicate
graphviz_dot's layered DAG layout via two LPs."

Produce a ranked list of every divergence between dagua_native's dot-style
sections and graphviz's dot algorithm. No matter how small.

## Reference clones

- Graphviz source: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/`
  - Key files: `dot.c`, `rank.c`, `mincross.c`, `position.c`, `flat.c`,
    `cluster.c`, `acyclic.c`, `class1.c`, `class2.c`, `aspect.c`
- Dagua dot path:
  - `dagua/layout/ops/pipelines/dagua_native.py` (3213 lines; focus on
    functions with `dot`, `lattice`, `layer`, `rank`, `back_edge` in name)
  - `dagua/layout/ops/pipelines/sugiyama.py` (the underlying base; already
    line-by-lined in Round 21 vs igraph_sugiyama, but the
    graphviz-dot-specific tweaks live in dagua_native.py)
  - `dagua/layout/engine.py` (algorithm dispatch)

## Output

Write `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_dot.md`
with:
- Per-section ranked list of every divergence with file:line on both sides
- Cover: rank assignment, mincross, x-coordinate position, back-edge handling,
  cluster positioning, edge crossing minimization, aspect-ratio targeting,
  spacing units (nodesep/ranksep), edge routing if relevant
- Concrete categorical labels (algorithm-correctness / numerical /
  parameter-default / convention / scaffolding)
- Estimated fix size (lines net) and risk level
- Distinguish items already addressed in R3 from items missed

Then run a baseline measurement. Use `classic_sugiyama` as the dagua side
(since that's what dot dispatches to) and `graphviz_dot` as the target:
```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/dot/baseline
```
Note: this comparator will only work if the cached benchmark has graphviz_dot
positions for these graphs. If the cache is missing graphs, document which.

Record baseline RMSD in your diff doc. **DO NOT apply fixes in this round** —
diff doc + baseline only. Round 28 will fix.

## Scope constraints

Same as other Round 27 prompts. DO NOT TOUCH render/styles. DIFF-ONLY round.

</task>

<research_mode>
Diagnostic round. Output is the ranked diff document.
</research_mode>

<default_follow_through_policy>
Skip nothing. dagua_native.py is large; budget time accordingly.
</default_follow_through_policy>
