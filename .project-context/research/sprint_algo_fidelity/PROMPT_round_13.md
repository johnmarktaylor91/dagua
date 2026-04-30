<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 13 of the algo_fidelity sprint, retry of davidson_harel attack.
Round 12 was BLOCKED by slow live_compare timeout but identified
concrete divergences. Round 13 reduces measurement scope and applies
the fix.

Read these in order:
1. `.project-context/research/sprint_algo_fidelity/ROUND_12_BLOCKED.md`
2. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`

## Diagnosis from Round 12 (already done)

Two concrete divergences from
`/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c`:

### Divergence 1: Energy weights
igraph defaults at lines 161-166:
```c
w_node_dist     = 1.0     /* MISSING in dagua */
w_borderlines   = 0.0
w_edge_lengths  = 0.0001
w_edge_crossings= 1.0
w_node_edge_dist= 0.2
```
igraph uses **unnormalized** energy terms.

dagua at `dagua/layout/ops/davidson_harel.py:16-20, 180-193`:
```python
border=0.1
edge_lengths=0.2
edge_crossings=2.0
node_edge_dist=0.5
# NO node_dist term -- this is the biggest gap
```
dagua uses **normalized** energy terms.

### Divergence 2: Move schedule
igraph at lines 151-162, 233-255, 262-263, 422-423:
- Tries **30 circular directions** per node per round
- `move_radius = width/2` where `width = sqrt(N)*10`
- Accepts downhill OR via SA probability `exp(-dE/T)`

dagua at `dagua/layout/ops/davidson_harel.py:353-380`:
- One random square move per node per round
- Radius derived from energy-scaled temperature

## Round 13 plan

### Step 1: Build a SMALL graph subset for fast iteration (10 min)

The 5-seed multi-seed comparator times out on the full graph set
(~30 graphs × 5 seeds × dagua + 5 seeds × igraph = lots of work).
Build a tight subset of 4-6 small graphs that davidson_harel handles
quickly:
- `linear_3layer_mlp` (6 nodes)
- `parallel_multiedge_bundle` (3 nodes)
- `binary_tree` (small)
- `nested_shallow_enc_dec` (6 nodes)
- `tl_mlp_3layer` (7 nodes)
- `mixed_width_labels` (6 nodes)

Add a `--graphs` CLI flag to the comparator (already supported per
Round 1) and run baseline on this subset. Should finish in ~2-3 min
per side, total <10 min for 5 seeds.

```
cd /home/jtaylor/projects/dagua
python scripts/algo_fidelity_live_compare.py classic_davidson_harel igraph_davidson_harel \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,binary_tree,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_13/baseline_small
```

(Use --seeds 3 instead of 5 to keep total time bounded; 3 is enough
for a meaningful TOST.)

If even this is too slow (> 15 min), drop to 3 graphs and 3 seeds.
Document.

If davidson_harel is reasonably equivalent already on small graphs
(within-floor ~ between-floor), the divergent verdict is mostly a
large-graph phenomenon and Round 13 documents that.

### Step 2: Apply the energy-weight alignment lever

In `dagua/layout/ops/davidson_harel.py`:
- Add the missing `node_dist` term to the energy function with
  weight `1.0` matching igraph's default.
- Switch other weights to igraph's exact defaults:
  - `border = 0.0` (igraph default; dagua had 0.1 -- if dagua was
    intentionally non-zero, document why, but switching to 0.0
    matches igraph)
  - `edge_lengths = 0.0001` (dagua had 0.2)
  - `edge_crossings = 1.0` (dagua had 2.0)
  - `node_edge_dist = 0.2` (dagua had 0.5)
- Switch from normalized to unnormalized energy terms (igraph's
  formulation). This is the more invasive change. If it requires
  > 80 lines, drop the normalization change and keep only the weight
  realignment.

If the move-schedule fix (Divergence 2) is also small, include it.
Otherwise it's Round 14's lever.

### Step 3: Measure post-fix on the same small subset

```
python scripts/algo_fidelity_live_compare.py classic_davidson_harel igraph_davidson_harel \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,binary_tree,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_13/post_fix
```

COMMIT criterion (relaxed for Phase 2 family):
- aggregate TOST verdict moves toward equivalent_at_<=2x of within-floor, OR
- median Procrustes RMSD on these small graphs improves by >= 0.05

### Step 4: Tests

```
pytest tests/test_layout/ -x --tb=short -q -k "davidson" 2>&1 | tail -30
pytest tests/test_layout/ -x --tb=short -q 2>&1 | tail -10
```

If snapshot tests on davidson_harel fail because the energy
function changed, **carefully evaluate** whether they're frozen-old-state
expectations (legitimate to update) or correctness invariants
(the fix is wrong).

### Step 5: Commit OR document residual

If COMMITTED:
```
feat(fidelity): round 13 -- davidson_harel-vs-igraph energy weight alignment

- Added missing node_dist=1.0 energy term (was absent in dagua)
- Aligned other energy weights to igraph defaults
  {border=0.0, edge_lengths=0.0001, edge_crossings=1.0, node_edge_dist=0.2}
- davidson_harel small-graph median: <BEFORE> -> <AFTER>
- TOST aggregate: <verdict>
- Tests: <count> passed
```

If RESIDUAL: write `ROUND_13_RESIDUAL.md` classifying.

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_13/SUMMARY.md`.

### Step 7: STATE.md update

Append iteration log row. If COMMIT lands AND verdict is
equivalent_at_<=2x: mark davidson_harel as `partial_match -> weak_equivalent`
(meeting user's secondary stop criterion). Set `current_family: drl`.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All other family pipelines

**Allowed in Round 13:**
- `dagua/layout/ops/davidson_harel.py` (PRIMARY)
- `dagua/layout/ops/pipelines/davidson_harel.py` (caller)
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- `eval_output/algo_fidelity/round_13/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_layout/test_*davidson*.py` ONLY if a test snapshot needs updating

**Out of scope:**
- Move-schedule rewrite if too invasive (defer to Round 14)
- Other family pipelines
</scope_constraints>

<default_follow_through_policy>
Even a partial fix (just energy weights, no move-schedule change)
that closes most of the gap is valuable. Aim for the highest-confidence
small change first.

If the fix is fundamentally simple but tests need adjustment for the
new behavior, that's expected and OK -- document the test changes.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED** if commit criterion met
2. **RESIDUAL** if no high-confidence fix lands or measurement still
   times out
3. **STOCHASTIC_FLOOR_MATCH** if multi-seed shows already equivalent
4. **BLOCKED** if hard infra issue (e.g., even small subset times out)
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q -k "davidson"
- live_compare with bounded subset runs cleanly
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- Even 3 graphs × 3 seeds for davidson_harel times out > 10 min
- The dagua davidson_harel ops file structure has been refactored

Write ROUND_13_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF measurable improvement.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files.
</action_safety>
