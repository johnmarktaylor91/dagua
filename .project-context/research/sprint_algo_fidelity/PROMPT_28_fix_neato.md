<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 28 STRAGGLER FIX for **neato** (graphviz neato binary).

## Round 27 finding

Source: `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_neato.md`

Baseline (`classic_stress_maj` vs `graphviz_neato`, 5 graphs, 30 seeds):
median 0.035264, worst tl_mlp_3layer 0.038680.

Cache only had 1 `graphviz_neato` seed → all TOST `not_tested`.

Top P0 finding from R27: **dagua has no in-process `algorithm="neato"`
dispatch.** Users wanting graphviz-neato-equivalent layout currently have
to use `classic_stress_maj` competitor or know the algo dispatch table.

## Your job (two phases)

### Phase A: Add algorithm="neato" dispatch (P0)

Add `"neato"` to `dagua/layout/ops/pipelines/__init__.py` PIPELINE_REGISTRY.
The neato pipeline should dispatch to stress_majorization with graphviz-
neato-faithful defaults:
- mode=major (the default; SMACOF stress majorization)
- model=shortpath (BFS-distance shortest paths, not naive)
- random init (NOT MDS init — graphviz neato uses random by default)
- 200 max iterations with early termination
- Default Epsilon = 0.0001
- packing on by default

Create `dagua/layout/ops/pipelines/neato.py` with:
```python
def layout_neato_pipeline(...): ...
```

Register it. Add a public-API test:
```python
import dagua
g = dagua.DaguaGraph(...)
pos = dagua.layout(g, dagua.LayoutConfig(algorithm="neato"))
```

### Phase B: Faithful-config defaults

Update `classic_stress_maj` (or add a new variant `classic_neato`) so when
`algorithm_params={"graphviz_neato_fidelity": True}` is passed, the pipeline:
- Random init (not MDS)
- 200 maxiter
- Epsilon convergence
- Packing
- Mode=major model=shortpath

Wire `classic_neato` competitor in `dagua/eval/competitors/classic_competitor.py`.
Add variants in `dagua/eval/variants.py` pairing `classic_neato` with
`graphviz_neato`.

## Reference

- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neato.c`
  (entrypoint), `stress.c` (SMACOF), `kkutils.c`, `pca.c`, `solve.c`
- `dagua/layout/ops/pipelines/stress_majorization.py`
- `dagua/layout/ops/pipelines/__init__.py`

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_neato graphviz_neato \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_28/neato/{baseline,post_fix}
```

If `classic_neato` doesn't yet exist, fall back to `classic_stress_maj`
which is the current proxy.

## Scope

- DO NOT TOUCH render/styles, cluster sprint files
- Stage commits explicitly. Commit format `feat(fidelity): round 28 neato -- <terse>`
- After each commit: `pytest tests/test_layout/ -x --tb=short -q -k "neato or stress"`

## Output

Per-round SUMMARY at `eval_output/algo_fidelity/round_28/neato/SUMMARY.md`.

</task>

<completeness_contract>
At minimum: add algorithm="neato" dispatch + a faithful-config path. If
post-fix RMSD doesn't improve due to single-seed reference cache, document
that as a measurement-limitation residual (not a real divergence).
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
