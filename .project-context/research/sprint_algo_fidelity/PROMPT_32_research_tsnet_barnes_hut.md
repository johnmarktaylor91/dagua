<task>
R32 RESEARCH -- tsnet Barnes-Hut alignment.

## Why this matters

R31 codex finding: the `tsne_graph` reference adapter constructs `sklearn.manifold.TSNE` without overriding `method=`, so sklearn defaults to `"barnes_hut"`. Dagua's `tsnet` pipeline does dense exact KL gradient via PyTorch autograd. Fundamental algorithm mismatch (approximation method) is the dominant residual after R31 fixes.

## Your job

PURE RESEARCH. No code edits.

1. Read sklearn `_t_sne.py` for the barnes_hut path:
   - `python -c "import sklearn.manifold._t_sne; print(sklearn.manifold._t_sne.__file__)"`
   - Key functions: `_kl_divergence_bh`, `_gradient_descent`, Barnes-Hut tree construction (`_barnes_hut_tsne.pyx` extension)
2. Read dagua's tsnet implementation: `dagua/layout/ops/tsnet.py`, `dagua/layout/ops/pipelines/tsnet.py`.
3. Decide: should we implement Barnes-Hut in dagua's tsnet, or should we change the reference adapter to use `method="exact"`?

   - Option A: Implement Barnes-Hut tree in dagua. Pure Python/PyTorch. ~500-1000 LoC. Matches sklearn default behavior.
   - Option B: Change reference adapter to pass `method="exact"`. dagua already does exact. 1-line change. But changes the "reference" semantics.

4. Write a recommendation.

## Output

`eval_output/algo_fidelity/round_32/tsnet_barnes_hut/REPORT.md` with:
- Sklearn Barnes-Hut implementation summary
- Both options' tradeoffs
- Recommendation + reasoning
- If recommend A: LoC estimate, complexity, expected RMSD delta
- If recommend B: what to update in `dagua/eval/competitors/tsne_competitor.py`
</task>

<research_mode>
Diagnostic round. Output is the REPORT.md.
</research_mode>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Read deeply.
</default_follow_through_policy>
