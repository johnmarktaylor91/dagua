<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch -- DO NOT create a new branch.

Round 12 of the algo_fidelity sprint. The graphviz portion is COMPLETE
per `algo_fidelity_SUMMARY.md`. This round attacks **Phase 2** -- the
less-important families with `divergent` or `partial_match` verdicts in
the existing fidelity report.

Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`
3. `eval_output/fidelity_report/report.md` (skim the davidson_harel rows)

## Round 12 target: davidson_harel (worst Phase 2 family)

Mega-run verdict: **divergent** (RMSD 0.34-0.36, 3 variants:
rounds50/100/200). davidson_harel is stochastic (uses RNG for moves).
The divergent verdict is REAL: igraph's competitor adapter properly
threads seed (line 46 in `dagua/eval/competitors/igraph_competitor.py`),
so this isn't a measurement artifact like graphviz_fdp was.

## igraph davidson_harel source -- READ FIRST

Source: `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c`

Key implementation details from the source:

### Defaults (from `igraph_layout_davidson_harel` entry point, around line 141)

```c
igraph_real_t width = sqrt(no_nodes) * 10, height = width;
igraph_real_t move_radius = width / 2;
igraph_real_t fine_tuning_factor = 0.01;
igraph_int_t no_tries = 30;       /* moves per node per outer round */

/* Energy function weights -- defaults documented in API comments */
igraph_real_t w_node_dist     = 1.0;     /* weight_node_dist */
igraph_real_t w_borderlines   = 0.0;     /* weight_border */
igraph_real_t w_edge_lengths  = 0.0001;  /* weight_edge_lengths */
igraph_real_t w_edge_crossings= 1.0;     /* weight_edge_crossings */
igraph_real_t w_node_edge_dist= 0.2;     /* weight_node_edge_dist */
```

### Initialization (lines 197-218)

```c
if (!use_seed) {
    for (igraph_int_t i = 0; i < no_nodes; i++) {
        x = MATRIX(*res, i, 0) = RNG_UNIF(-width / 2, width / 2);
        y = MATRIX(*res, i, 1) = RNG_UNIF(-height / 2, height / 2);
    }
}
```

So initial coordinates are uniform random in [-width/2, +width/2]^2
where width = sqrt(N) * 10.

### Energy function (search the file for `dh_energy`)

Energy is a weighted sum of:
- node-node repulsion (w_node_dist=1.0)
- node-border distance (w_borderlines=0.0 by default)
- edge length deviation from ideal (w_edge_lengths=0.0001)
- edge-edge crossings count (w_edge_crossings=1.0)
- node-edge distance (w_node_edge_dist=0.2)

### Annealing schedule

- `maxiter` outer rounds (caller supplies; the variants use 50/100/200)
- Each round: try `no_tries=30` candidate moves per node within
  `move_radius`
- `move_radius` shrinks by `cool_fact` per round (caller supplies;
  default in API doc is around 0.75-0.95)
- After maxiter rounds, optional `fineiter` rounds with smaller moves
  (`fine_tuning_factor=0.01`)

## Dagua davidson_harel surface

- `dagua/layout/ops/davidson_harel.py` -- ops: PrepareDHState,
  InitializeDHPositions, DHAnnealingRound, DHCool, FinalizeDHPositions
- `dagua/layout/ops/pipelines/davidson_harel.py` -- pipeline wiring
  (rounds=100 default per `build_davidson_harel_pipeline`)

Investigate:
1. **Initialization**: does dagua use `RNG_UNIF(-width/2, width/2)` with
   `width = sqrt(N) * 10`, or different scale / distribution?
2. **Energy function weights**: do dagua's defaults match
   {1.0, 0.0, 0.0001, 1.0, 0.2}? Order matters: node_dist,
   border, edge_lengths, edge_crossings, node_edge_dist.
3. **no_tries=30**: does dagua try 30 candidate moves per node per
   round, or fewer / more?
4. **Cooling factor**: igraph's API docstring suggests reasonable
   cool_fact ~0.75. dagua may use different.
5. **Move-acceptance criterion**: igraph picks the best move out of
   `no_tries`; dagua may do something else (random first improvement,
   gradient).

## What to do

### Step 1: Live multi-seed baseline (10 min)

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_davidson_harel igraph_davidson_harel \
    --seeds 5 --output-dir eval_output/algo_fidelity/round_12/baseline
```

Capture:
- within-igraph_davidson_harel floor (median, p95)
- dagua-vs-igraph (median, p95)
- aggregate TOST verdict

If within-floor is high (e.g., > 0.20), the family is highly stochastic
and TOST may already say equivalent. Don't proceed with algorithm
fixes if Round 9-style stochastic-floor finding rescues davidson_harel.

If within-floor is small and dagua-vs-igraph is large -> real
divergence; proceed to Step 2.

### Step 2: Compare dagua DH to igraph DH (15 min)

Read `dagua/layout/ops/davidson_harel.py` end-to-end. Map each piece
to its igraph counterpart. Identify the 1-2 highest-confidence
divergences. Document in
`.project-context/research/sprint_algo_fidelity/ROUND_12_DIAGNOSIS.md`
with file:line references on both sides.

### Step 3: ONE focused lever (15-30 min)

Apply the smallest plausible fix. Most likely candidates:
- **Hyperparameter alignment**: match igraph's energy weights /
  no_tries / cool_fact / fine_tuning_factor defaults. ~20-50 lines.
- **Init scale fix**: if dagua initializes at unit scale instead of
  sqrt(N)*10 box, this may matter for the overall energy landscape.
  ~10 lines.
- **Move-acceptance fix**: if dagua uses a different rule, change
  to "best of 30 tries" per igraph. Could be 30+ lines.

### Step 4: Measure with multi-seed (10 min)

```
python scripts/algo_fidelity_live_compare.py classic_davidson_harel igraph_davidson_harel \
    --seeds 5 --output-dir eval_output/algo_fidelity/round_12/post_fix
```

COMMIT criterion: aggregate TOST verdict moves toward equivalent_at_<=2x,
OR family median improves by >= 0.05.

### Step 5: Tests + commit

```
pytest tests/test_layout/ -x --tb=short -q -k "davidson" 2>&1 | tail -20
```

If COMMITTED:
```
feat(fidelity): round 12 -- davidson_harel-vs-igraph first lever (<short>)

- Identified divergence: <one sentence>
- Fix: <one sentence>
- davidson_harel family median: 0.34X -> <NEW> (or TOST verdict
  shift)
- Within-floor / between-floor at TOST equivalence
- Tests: <count> passed
```

If RESIDUAL:
- Write `ROUND_12_RESIDUAL.md` classifying. Move to drl in Round 13.

### Step 6: Per-round summary

Write `eval_output/algo_fidelity/round_12/SUMMARY.md`.

### Step 7: Update STATE.md

Append iteration log row. Update `state` and `current_round`.
Set `current_family: drl` for Round 13 if davidson_harel is closed
(committed or residual).
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All other family pipelines (sugiyama, fmmm, sfdp, stress_majorization,
  classical_mds, drl, graphopt, neulay, tsnet, fa2 etc.) -- Round 12 is
  davidson_harel only

**Allowed in Round 12:**
- `dagua/layout/ops/davidson_harel.py` (PRIMARY)
- `dagua/layout/ops/pipelines/davidson_harel.py` (caller / config)
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- `eval_output/algo_fidelity/round_12/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*davidson*.py` ONLY if a test snapshot needs updating
</scope_constraints>

<default_follow_through_policy>
The Round 3 win on dot was a single hyperparameter alignment.
Davidson-Harel has many hyperparameters that may diverge from igraph
defaults. Try the same playbook: align defaults to match igraph,
measure with multi-seed TOST, commit if it lands.

If davidson_harel turns out to be stochastic-floor faithful (Round 9
pattern) -- great, classify and move on without code changes.

If a single hyperparameter alignment doesn't move the needle, then
the divergence is in the algorithm structure (move selection /
energy function math) -- write residual and move to drl.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED** if commit criterion met
2. **RESIDUAL** if no high-confidence lever
3. **STOCHASTIC_FLOOR_MATCH** if multi-seed TOST shows family is
   already equivalent within within-igraph floor (write
   ROUND_12_STOCHASTIC_FLOOR_MATCH.md, no code changes, just commit
   the analysis)
4. **BLOCKED** if hard infra issue
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q -k "davidson"
- live_compare with --seeds 5 runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- igraph clone at `/home/jtaylor/projects/_references/igraph` is missing
  (verify `lib/.../davidson_harel.c` exists -- actually under
  `src/layout/davidson_harel.c`)
- live_compare for davidson_harel times out (graph too slow)
- No determinism in dagua DH despite fixed seed

Write ROUND_12_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measurable improvement OR clear
  classification.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files.
</action_safety>
