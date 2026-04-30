<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 18 of the algo_fidelity sprint, last queued Phase 2 family.

Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `eval_output/algo_fidelity/round_13/SUMMARY.md` (only winning Phase 2 round so far)

## Round 18 target: tsnet family

Mega-run verdict: **partial_match** (RMSD 0.15-0.27 across 4 variants:
default, perp5, perp50, steps200, steps2000).

The reference is `tsne_graph` adapter
(`dagua/eval/competitors/tsne_competitor.py`) which uses **sklearn t-SNE**
on shortest-path graph distances. Seed propagates via sklearn's
`random_state`.

### sklearn TSNE defaults

The reference uses sklearn.manifold.TSNE on graph shortest-path distances.
Key sklearn defaults (from sklearn>=1.2):
- `n_components=2`
- `perplexity=30.0`
- `early_exaggeration=12.0`
- `learning_rate='auto'` (or 200.0 in older versions)
- `n_iter=1000` (max_iter in newer versions)
- `init='pca'` (changed from 'random' in sklearn 1.2)
- `metric='euclidean'` (the reference passes `metric='precomputed'` for distance matrix)

Look at the actual sklearn version + the reference adapter to confirm
what defaults the cached graphviz_neato-style positions used.

### Dagua tsnet surface

- `dagua/layout/ops/tsnet.py`
- `dagua/layout/ops/pipelines/tsnet.py`

Variant params per the reference: `{learning_rate, max_iter, perplexity}`.

## What to do

### Step 1: Live multi-seed baseline (10 min)

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_18/baseline_small
```

If within-floor is high (sklearn t-SNE is highly stochastic), tsnet may
already be stochastic-floor faithful. Document.

### Step 2: Diagnose (15 min)

Read `dagua/layout/ops/tsnet.py` end-to-end. Compare to sklearn TSNE
algorithm:
- Initialization (PCA vs random vs uniform)
- Perplexity calibration (binary search for sigma per point)
- KL divergence + early exaggeration phase + main phase
- Learning rate schedule
- Adam-style momentum / gradient updates

Identify highest-confidence divergence. Most likely:
- Init (PCA vs random)
- Learning rate magnitude (sklearn uses 200 or 'auto'=N/12)
- Early exaggeration factor (12.0 default)
- Number of iterations split between exaggeration / main phase

Write `.project-context/research/sprint_algo_fidelity/ROUND_18_DIAGNOSIS.md`.

### Step 3: ONE focused lever (15-30 min)

Same playbook. Most likely small fix: a hyperparameter alignment.

If the divergence is fundamental (different KL formulation, different
optimizer), document as architectural floor and move on.

### Step 4: Measure on the same small subset

```
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_18/post_fix
```

COMMIT criterion: median improves by >= 0.03 OR aggregate TOST flips
toward equivalent_at_<=2x.

### Step 5: Tests + commit OR residual

```
pytest tests/test_layout/ -x --tb=short -q -k "tsnet" 2>&1 | tail -20
```

If COMMITTED:
```
feat(fidelity): round 18 -- tsnet-vs-sklearn first lever (<short>)

- Identified divergence: <one sentence>
- Fix: <one sentence>
- tsnet small-graph median: <BEFORE> -> <AFTER>
- TOST aggregate: <verdict>
- Tests: <count> passed
```

If RESIDUAL: `ROUND_18_RESIDUAL.md`.

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_18/SUMMARY.md`.

### Step 7: Update STATE.md

Append iteration log row. Set `current_round: 19` and
`current_family: phase_2_complete`. Round 18 closes Phase 2 attacks.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All other family pipelines

**Allowed:**
- `dagua/layout/ops/tsnet.py` (PRIMARY)
- `dagua/layout/ops/pipelines/tsnet.py`
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- `eval_output/algo_fidelity/round_18/**`
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*tsnet*.py` for snapshot updates
</scope_constraints>

<default_follow_through_policy>
Same playbook as Round 13. Phase 2 has been mostly diminishing returns
post-davidson_harel; expect tsnet to land either an alignment win or
classify as architectural floor. A clean residual is fine.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED** if commit criterion met
2. **RESIDUAL** if no high-confidence small fix
3. **STOCHASTIC_FLOOR_MATCH** if multi-seed shows already equivalent
4. **BLOCKED** if hard infra issue
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q -k "tsnet"
- live_compare with bounded subset runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- live_compare for tsnet times out
- dagua tsnet ops file not found

Write ROUND_18_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measurable improvement.
- Never delete eval_output files.
</action_safety>
