<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 14 of the algo_fidelity sprint. Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `eval_output/algo_fidelity/round_13/SUMMARY.md` (the davidson_harel win pattern)
3. `eval_output/fidelity_report/report.md` (drl_* rows)

## Round 14 target: drl family (next worst Phase 2)

Mega-run verdict: **partial_match** (RMSD 0.13-0.20 across 5 variants:
default, coarsen, coarsest, refine, final). DRL is stochastic. igraph
adapter properly threads seed.

## igraph drl source -- READ FIRST

Source: `/home/jtaylor/projects/_references/igraph/src/layout/drl/`
- `drl_layout.cpp` -- entry point and defaults (start at line 240
  `igraph_layout_drl_options_init`)
- `drl_graph.cpp` -- main simulated-annealing iteration
- `DensityGrid.cpp` -- density-based repulsion grid
- `drl_layout.h` / `drl_graph.h` -- structures

### igraph DRL defaults (DEFAULT preset, lines 246-275)

```c
options->edge_cut = 32.0 / 40.0;  // = 0.8

// Phase 1: init
options->init_iterations   = 0;       // skipped by default
options->init_temperature  = 2000;
options->init_attraction   = 10;
options->init_damping_mult = 1.0;

// Phase 2: liquid
options->liquid_iterations   = 200;
options->liquid_temperature  = 2000;
options->liquid_attraction   = 10;
options->liquid_damping_mult = 1.0;

// Phase 3: expansion
options->expansion_iterations   = 200;
options->expansion_temperature  = 2000;
options->expansion_attraction   = 2;     // KEY: drops from 10 to 2
options->expansion_damping_mult = 1.0;

// Phase 4: cooldown
options->cooldown_iterations   = 200;
options->cooldown_temperature  = 2000;
options->cooldown_attraction   = 1;
options->cooldown_damping_mult = 0.1;    // damps fast

// Phase 5: crunch
options->crunch_iterations   = 50;
options->crunch_temperature  = 250;      // big temperature drop
options->crunch_attraction   = 1;
options->crunch_damping_mult = 0.25;

// Phase 6: simmer
options->simmer_iterations   = 100;
options->simmer_temperature  = 250;
options->simmer_attraction   = 0.5;
options->simmer_damping_mult = 0;        // no damping
```

The COARSEN preset varies liquid_attraction (10 -> 2) and
expansion_attraction (2 -> 10). Other presets (COARSEST, REFINE, FINAL)
similar variations -- check
`drl_layout.cpp:igraph_layout_drl_options_init` for all 5 templates.

### Energy / force model

DRL uses a density-based repulsion grid (DensityGrid). Each node
contributes a Gaussian density to a coarse grid; repulsion is then
computed from grid density rather than O(N^2) pairwise. Edge attraction
is FR-style proportional to distance.

## Dagua drl surface

- `dagua/layout/ops/drl.py` (or similar -- locate it)
- `dagua/layout/ops/pipelines/drl.py`

Variants in `dagua/eval/variants.py`: classic_drl_default, coarsen,
coarsest, refine, final.

Investigate:
1. **Phase parameters**: do dagua's per-phase iterations/temperatures/
   attractions match igraph's defaults exactly? Especially watch
   `expansion_attraction=2` (drops from 10) and `cooldown_damping_mult=0.1`.
2. **Density grid**: does dagua use a density grid for repulsion or
   plain O(N^2)? They produce different layouts.
3. **Initial layout**: igraph DRL uses random init within a square
   box. Same as davidson_harel? Check.

## What to do

### Step 1: Live multi-seed baseline on small subset (10 min)

Use the same bounded subset as Round 13 (was effective):
```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_drl_default igraph_drl \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,binary_tree,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_14/baseline_small
```

Use --graphs to keep it fast; if drl is also slow, drop to 4 graphs.

If within-igraph_drl floor is high (e.g., ~0.20), the family may
already be stochastic-floor faithful. Document and move on if so.

### Step 2: Diagnose (15 min)

Read dagua drl ops. Compare to igraph defaults. Identify the 1-2
highest-confidence divergences. Write
`.project-context/research/sprint_algo_fidelity/ROUND_14_DIAGNOSIS.md`
with file:line references.

### Step 3: ONE focused lever (15-30 min)

Same playbook as Round 13. Most likely candidates:
- Phase-parameter alignment (iterations, temperatures, attraction,
  damping per phase)
- Initial-layout-box scale / distribution alignment
- Density-grid implementation if dagua diverges there

If dagua doesn't have a density grid at all and uses plain O(N^2)
repulsion, that's a structural difference -- document but don't try
to add a grid this round (too invasive).

### Step 4: Measure on the same small subset

```
python scripts/algo_fidelity_live_compare.py classic_drl_default igraph_drl \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,binary_tree,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_14/post_fix
```

COMMIT criterion: median improves by >= 0.03 (smaller threshold than
Round 13 because partial_match starts closer than divergent).

### Step 5: Tests + commit

```
pytest tests/test_layout/ -x --tb=short -q -k "drl" 2>&1 | tail -20
```

If COMMITTED:
```
feat(fidelity): round 14 -- drl-vs-igraph first lever (<short>)

- Identified divergence: <one sentence>
- Fix: <one sentence>
- drl_default small-graph median: <BEFORE> -> <AFTER>
- TOST aggregate: <verdict>
- Tests: <count> passed
```

If RESIDUAL: write `ROUND_14_RESIDUAL.md`.

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_14/SUMMARY.md`.

### Step 7: Update STATE.md

Append iteration log row. Set `current_family: graphopt` for Round 15.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All other family pipelines (sugiyama, fmmm, sfdp, stress_majorization,
  classical_mds, davidson_harel, graphopt, neulay, tsnet, fa2)

**Allowed:**
- `dagua/layout/ops/drl.py` or wherever the drl ops live (PRIMARY)
- `dagua/layout/ops/pipelines/drl.py`
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- `eval_output/algo_fidelity/round_14/**`
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*drl*.py` for snapshot updates
</scope_constraints>

<default_follow_through_policy>
Follow Round 13's playbook -- it landed -0.124 median RMSD on
davidson_harel via parameter alignment + move-schedule fix. drl has
even more hyperparameters per phase (6 phases × 4 params each).
Higher chance of finding a single big lever that aligns multiple
phases at once.

If drl is already stochastic-floor faithful per multi-seed TOST
(within-igraph floor >= dagua-vs-igraph floor), classify and move on.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED** if commit criterion met
2. **RESIDUAL** if no high-confidence fix lands or measurement times out
3. **STOCHASTIC_FLOOR_MATCH** if multi-seed shows already equivalent
4. **BLOCKED** if hard infra issue
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q -k "drl"
- live_compare with bounded subset runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- live_compare for drl times out even on 4 small graphs × 3 seeds
- dagua drl ops file not found at expected location

Write ROUND_14_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measurable improvement.
- Never delete eval_output files.
</action_safety>
