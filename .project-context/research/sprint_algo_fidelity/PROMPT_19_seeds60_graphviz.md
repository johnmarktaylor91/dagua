<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 19 60-SEED POWER ANALYSIS for graphviz fdp/sfdp/neato.

The user wants more statistical power on the multi-seed TOST analysis
that validates dagua's drop-in graphviz replacement claim. Currently
9 seeds; bump to 60 seeds.

## Background

Round 9 fixed graphviz_competitor.py to thread `-Gseed` and `-Gstart`
to the graphviz binary. Generated 9-seed cache at
`eval_output/algo_fidelity/round_9/graphviz_seeded_cache/`. TOST
verdicts at 9 seeds: fdp/sfdp/neato all CONVERGED.

For 60 seeds: dagua side should also run 60 seeds for proper
distribution comparison.

## What to do

### Step 1: Generate 60-seed graphviz cache (~30-60 min depending on graph set)

Locate or write the regen script (Round 9 created
`scripts/regen_graphviz_seeds.py` or similar). Run it for seeds 42..101
(60 seeds) for the small bounded graph subset:

`linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,binary_tree,inception_block,petersen_10,edge_label_braid`

(9 graphs to keep total time bounded -- ~9 graphs × 60 seeds × 3
engines = 1620 graphviz invocations, ~30-60 min depending on machine.)

Save to: `eval_output/algo_fidelity/round_19/graphviz_seeded_cache_60/`

If full 60-seed runs hit time limits, scale down to 30 seeds and
document.

### Step 2: Run multi-seed comparator at 60 seeds

For fdp, sfdp, neato_stress, neato_mds:
```
python scripts/algo_fidelity_live_compare.py classic_<engine> graphviz_<target> \
    --seeds 60 \
    --graphs <same 9 graphs> \
    --graphviz-cache-dir eval_output/algo_fidelity/round_19/graphviz_seeded_cache_60 \
    --output-dir eval_output/algo_fidelity/round_19/<engine>_60seed
```

If `--seeds 60` for dagua side is also too slow, run with `--seeds 30`
on the dagua side (still pairs each dagua seed against all 60 graphviz
seeds for 30*60=1800 RMSDs per graph, which gives even better power).
Document the choice.

### Step 3: TOST verdicts at 60 seeds

For each pairing, compute:
- within-graphviz floor (median, p95) at 60 seeds
- within-dagua floor (median, p95) at 30 seeds
- dagua-vs-graphviz (median, p95) at 30*60 pairs
- TOST aggregate verdict at margin factors {0.25x, 0.5x, 1x, 1.5x, 2x}
  (added 0.25x for stricter test)
- Per-graph TOST verdict counts

Write `.project-context/research/sprint_algo_fidelity/ROUND_19_60SEED_TOST.md`
with:
- Comparison vs Round 9 9-seed verdicts (did anything change?)
- Whether stronger margins (0.25x, 0.5x) still classify as equivalent
- Confidence intervals on the within-floor estimates

### Step 4: Update SUMMARY

Append a "60-seed validation" section to
`.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`
with the 60-seed verdicts. Don't replace the 9-seed numbers; add
alongside.

### Step 5: Commit

```
feat(fidelity): round 19 -- 60-seed graphviz TOST power analysis

- Generated 60-seed graphviz cache (fdp/sfdp/neato) on bounded subset
- Re-ran multi-seed TOST with 30-60 seeds per side
- Verdicts: <list per family>
- 0.25x stricter margin: <if any pass>
- ROUND_19_60SEED_TOST.md with full details
```
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All pipeline ops files

**Allowed:**
- `scripts/regen_graphviz_seeds.py` (extend if needed)
- `scripts/algo_fidelity_live_compare.py` (extend if needed for 60 seeds)
- `eval_output/algo_fidelity/round_19/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`

NO algorithm code changes this round.
</scope_constraints>

<completeness_contract>
1. **COMMITTED**: 60-seed cache generated, TOST verdicts updated, report
   written, SUMMARY updated, commit on develop.
2. **SCALED-DOWN COMMITTED**: ran with fewer seeds (e.g. 30) due to
   time constraints; commit with documentation of the scale-down.
3. **BLOCKED** if hard infra issue.
</completeness_contract>

<verification_loop>
- multi_seed_summary.json with at least 30 seeds per side per pairing
- ROUND_19_60SEED_TOST.md written
- pytest tests/test_layout/ unaffected
</verification_loop>
