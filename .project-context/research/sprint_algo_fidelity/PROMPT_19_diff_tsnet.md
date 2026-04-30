<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 19 ADVERSARIAL DIFF for **tsnet** family. Reference is sklearn
TSNE on shortest-path distance matrices. Round 18 noted dagua uses
torch RNG, sklearn uses numpy RNG -- different sequences on same seed.

## Inputs

**Dagua side (READ ALL):**
- `dagua/layout/ops/tsnet.py`
- `dagua/layout/ops/pipelines/tsnet.py`

**Reference (READ ALL):**
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py`
- The competitor adapter at `dagua/eval/competitors/tsne_competitor.py`
  to confirm the exact sklearn invocation params

**Existing analysis:**
- `eval_output/algo_fidelity/round_18/SUMMARY.md`

## What to do

**DIAGNOSIS-ONLY.** Produce ONE document:
`.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_tsnet.md`

Cover:

1. **Initialization** (sklearn: PCA or random*1e-4 numpy normal; dagua:
   ?). What shape, scale, RNG?
2. **Distance matrix** (the reference uses graph shortest-path
   distances passed via `metric='precomputed'`)
3. **Perplexity calibration** (sklearn binary searches for sigma per
   point such that effective neighbors = perplexity)
4. **High-D affinities P** (sklearn: exp(-d^2/(2*sigma^2)) symmetrized)
5. **Low-D affinities Q** (sklearn: t-distribution kernel, 1/(1+d^2))
6. **KL gradient** formula details
7. **Early exaggeration** phase (sklearn: P *= 12, for first 250 iters)
8. **Optimizer** (sklearn uses momentum-based gradient descent; dagua: ?)
9. **Learning rate** (sklearn: 'auto' = N/12 by default)
10. **Convergence / iteration count** (sklearn: max_iter=1000, two phases)
11. **RNG semantics** (numpy vs torch -- same seed, different sequence)
12. **Hyperparameter alignment table**
13. **Ranked fix list**
14. **Recommended Round 20 fix scope**

Be exhaustive. Cite line:line refs.

## Constraints

DIAGNOSIS ONLY. No file edits. No commits.
</task>

<scope_constraints>
DIAGNOSIS-ONLY. Read-only.
</scope_constraints>

<verification_loop>
File ROUND_19_DIFF_tsnet.md exists, exhaustive, with line:line refs.
</verification_loop>
