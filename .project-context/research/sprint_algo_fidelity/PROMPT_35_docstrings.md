<task>
R35 per-engine fidelity docstring documentation.

Bake a "fidelity status" section into every `dagua/layout/ops/pipelines/<engine>.py`
docstring documenting:
1. Which reference implementation this targets
2. What fidelity_mode flags do (if any)
3. Known divergences / architectural floors

This improves maintenance and lets future contributors understand the
fidelity contract per engine.

## Your job

For each engine in `dagua/layout/ops/pipelines/`:
- fr.py, kk.py, fa2.py, sugiyama.py, spectral.py, classical_mds.py,
  stress_majorization.py, neato.py, pivot_mds.py, reingold_tilford.py,
  linlog.py, gem.py, tsnet.py, maxent_stress.py, davidson_harel.py,
  fmmm.py, graphopt.py, drl.py, lgl.py, sfdp.py, umap_layout.py, neulay.py,
  sgd2_multi.py, stress_sgd.py, fcose.py, yifanhu.py
- (plus dagua_native.py)

Add a docstring section at the top of `build_<engine>_pipeline()` (or the
public entry point):
```python
def build_<engine>_pipeline(...):
    """Build a <engine> layout pipeline.

    Reference fidelity
    ------------------
    Targets: <reference name + version + paper ref>
    Fidelity mode: <flag if any, what it does>
    Verified at: round_<N>, RMSD median <X> on bounded subset
    Known divergences:
        - <item 1>
        - <item 2>
    """
```

Pull info from:
- `eval_output/algo_fidelity/round_27/<engine>/PLAN_*.md` (DIFF docs)
- `eval_output/algo_fidelity/round_31/<engine>/SUMMARY.md`
- `eval_output/algo_fidelity/round_32/<engine>/SUMMARY.md`
- `eval_output/algo_fidelity/round_33/<engine>/SUMMARY.md`
- `eval_output/algo_fidelity/round_34/<engine>/SUMMARY.md`
- `eval_output/fidelity_report_100seed_final/report.md` (final verdicts)

If an engine has no fidelity work, leave docstring as-is.

## Output

A single commit per ~5 engines, message format:
`docs(layout): round 35 -- fidelity docstrings for <engines>`

Plus `eval_output/algo_fidelity/round_35/docstrings/SUMMARY.md` listing
which engines got documentation.

Use commit-safe wrapper.
</task>

<completeness_contract>
Document every engine that has fidelity work. Skip ones that have none.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
