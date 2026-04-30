<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 17 of the algo_fidelity sprint. Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `eval_output/algo_fidelity/round_15/SUMMARY.md` and round_16/ SUMMARY.md
   (graphopt context: hyperparameters already aligned, residual at architectural floor)

## Round 17 target: neulay family

Mega-run verdict: **partial_match** (RMSD 0.16-0.20 across 6 variants:
default, steps200, steps500, lr01, radius001, use_gcn). NeuLay is
**GNN-based** (uses a Graph Convolutional Network to predict positions),
NOT a physics-based force-directed layout. This makes it structurally
different from previous Phase 2 families.

### NeuLay reference

The "neulay" competitor in `dagua/eval/competitors/neulay_competitor.py`
loads upstream `neulay` or `NeuLay` Python package via `importlib`. It
requires PyTorch Geometric.

Variant params per the competitor adapter:
`{gcn_steps, lr, radius, steps, use_gcn}`.

If the upstream package is NOT installed, the adapter returns None and
live_compare cannot run. The mega-run apparently had it installed
(produced cached positions). Round 17 needs to detect availability
first.

### Dagua neulay surface

- `dagua/layout/ops/neulay.py`
- `dagua/layout/ops/pipelines/neulay.py`

Neither has been tweaked in this sprint. The implementation is GNN-based
per the variant params (use_gcn flag, gcn_steps, lr).

## What to do

### Step 1: Probe environment + baseline (10 min)

```
cd /home/jtaylor/projects/dagua
# Check upstream availability
python -c "import torch_geometric; import neulay" 2>&1 | head -5
ls eval_output/benchmark_full/positions/ | grep neulay | head -5

# Multi-seed baseline on small subset
python scripts/algo_fidelity_live_compare.py classic_neulay neulay \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_17/baseline_small
```

If upstream `neulay` is not available, the live comparator may still
work via cached graphviz-style positions in benchmark_full. Document
exactly what runs.

If within-target floor is high (e.g., > 0.10), neulay may already be
stochastic-floor faithful and the partial_match is mostly noise.

### Step 2: Diagnose (15 min)

Read `dagua/layout/ops/neulay.py` and `dagua/layout/ops/pipelines/neulay.py`
end-to-end. Compare to:
- The upstream `neulay` package source if installed (locate via
  `python -c "import neulay; print(neulay.__file__)"`)
- The competitor adapter's variant params for hints about what
  parameters matter

Identify the highest-confidence divergence. Likely candidates:
- Network architecture (number of GCN layers, hidden dim, activation)
- Training schedule (steps, learning rate, optimizer)
- Random init scheme (Xavier/Glorot vs uniform vs ...)
- Loss function formulation (which loss terms? what weights?)

Write `.project-context/research/sprint_algo_fidelity/ROUND_17_DIAGNOSIS.md`.

### Step 3: ONE focused lever (15-30 min)

Same playbook. Keep change small. If the upstream package isn't
available for live measurement, you may need to compare against
cached positions only, which is fine if multi-seed RMSD still works
at the file level.

### Step 4: Measure on the same small subset

```
python scripts/algo_fidelity_live_compare.py classic_neulay neulay \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_17/post_fix
```

COMMIT criterion: median improves by >= 0.03, OR aggregate TOST flips
toward equivalent_at_<=2x.

### Step 5: Tests + commit OR residual

```
pytest tests/test_layout/ -x --tb=short -q -k "neulay" 2>&1 | tail -20
```

If COMMITTED:
```
feat(fidelity): round 17 -- neulay-vs-upstream first lever (<short>)

- Identified divergence: <one sentence>
- Fix: <one sentence>
- neulay small-graph median: <BEFORE> -> <AFTER>
- TOST aggregate: <verdict>
- Tests: <count> passed
```

If RESIDUAL: `ROUND_17_RESIDUAL.md`.

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_17/SUMMARY.md`.

### Step 7: Update STATE.md

Append iteration log row. Set `current_family: tsnet` for Round 18.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All other family pipelines (sugiyama, fmmm, sfdp, stress_majorization,
  classical_mds, davidson_harel, drl, graphopt, tsnet, fa2)

**Allowed:**
- `dagua/layout/ops/neulay.py` (PRIMARY)
- `dagua/layout/ops/pipelines/neulay.py`
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- `eval_output/algo_fidelity/round_17/**`
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*neulay*.py` for snapshot updates
</scope_constraints>

<default_follow_through_policy>
NeuLay is GNN-based, structurally different from prior Phase 2 families.
The fix may not be as simple as a single hyperparameter alignment.

If after diagnosis, the divergence is fundamental (e.g., different
architecture, different loss formulation), document it as a
`principled_residual: architectural_difference` and move on. The
goal is "good faith effort" per the user, not exhaustive perfection.

If multi-seed TOST shows neulay is already stochastic-floor faithful
on most graphs, classify and move on quickly.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED** if commit criterion met
2. **RESIDUAL** if no high-confidence small fix
3. **STOCHASTIC_FLOOR_MATCH** if multi-seed shows already equivalent
4. **BLOCKED** if upstream neulay unavailable AND cached positions don't work
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q -k "neulay"
- live_compare with bounded subset runs cleanly OR cached positions used
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- Upstream neulay unavailable AND cached benchmark_full positions don't
  exist for the test graphs
- live_compare for neulay times out

Write ROUND_17_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measurable improvement.
- Never delete eval_output files.
</action_safety>
