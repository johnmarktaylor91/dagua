<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 21 ADVERSARIAL DIFF for **fr** family (dagua `classic_fr` vs reference `nx_spring`).

This is part of an exhaustive sweep covering EVERY dagua-vs-reference
pairing. Even if the family is currently `strong_equivalent` in the
mega-run, the user wants every last divergence catalogued -- GPT-5.5
may find something new.

## Inputs

**Dagua side (READ ALL):**
- Locate `dagua/layout/ops/fr.py` or related ops files for this engine.
- Locate `dagua/layout/ops/pipelines/fr.py` (the pipeline wiring).
- `dagua/eval/variants.py` for variant configs.
- `dagua/eval/competitors/` for the adapter that runs nx_spring.

**Reference side (READ ALL):**
- Reference path hint: `networkx/drawing/layout.py`
- Search for the actual implementation if the hint path doesn't exist:
  For igraph_*: source at /home/jtaylor/projects/_references/igraph/src/layout/<algo>.c (or .cpp). For ogdf_*: source at /home/jtaylor/projects/_references/ogdf/src/ogdf/<category>/<algo>.cpp + headers in /home/jtaylor/projects/_references/ogdf/include/ogdf/<category>/. For nx_* (networkx): source at /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py. For fa2_*: /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2_modified/ (preferred) or fa2/. For sgd2_*: search site-packages for sgd2 or its installed name. For umap_*: /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/. For tsne_*: /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py.
- For composite/hybrid engines, locate all relevant files.

**Existing analysis to skim:**
- `eval_output/fidelity_report/report.md` for the current verdict on fr.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md` for sprint context.

## What to do

**This is a DIAGNOSIS-ONLY round.** Do NOT edit any source files. No commits.

Produce ONE document: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fr.md`

Sections (be brutally exhaustive):

1. **Files read** -- list every source file you read on both sides.
2. **Overall pipeline structure** -- compare the high-level flow of the dagua and reference implementations.
3. **Energy / loss / objective** -- per-term comparison; cite formulas with file:line refs on both sides.
4. **Force / gradient computation** -- if applicable.
5. **Initialization** -- random scheme, scale, RNG type (numpy/torch/python random).
6. **Iteration / convergence** -- step count, learning-rate schedule, convergence test.
7. **Hyperparameter alignment table** -- exhaustive Y/N match per param + dagua default vs reference default.
8. **Edge cases** -- self-loops, multi-edges, disconnected components, weighted edges, empty graph.
9. **Numerical precision** -- float32 vs float64, dtype boundaries, summation order.
10. **RNG semantics** -- specifically does dagua's torch seed produce same sequence as reference's RNG?
11. **Edge-case bugs** -- anything that looks like an off-by-one, wrong sign, wrong direction, etc.
12. **Ranked fix list** -- 5+ items ranked by expected RMSD impact, each with file:line refs and proposed fix size estimate.
13. **Recommended Round 22+ fix scope** -- bundle of top-K levers for one followup round.

Be exhaustive. Cite specific line:line refs throughout. If the family
is `strong_equivalent` already, focus on residual sub-percent
divergences -- e.g., float precision, summation order, RNG semantics --
even if no obvious algorithmic divergence exists.

## End state
ONE markdown report at the path above. NO code changes. NO commits.
</task>

<scope_constraints>DIAGNOSIS-ONLY. NO file edits. NO commits. Read-only.</scope_constraints>

<verification_loop>File ROUND_21_DIFF_fr.md exists and is exhaustive (>10KB) with line:line refs throughout.</verification_loop>
