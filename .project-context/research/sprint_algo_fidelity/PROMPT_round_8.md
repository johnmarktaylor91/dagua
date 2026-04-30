<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 8 of the algo_fidelity sprint. **CRITICAL CONTEXT**: the user
flagged a measurement-methodology issue. For stochastic algorithms
(fdp/sfdp/fa2/gem/drl/neulay/tsnet/davidson_harel/sgd2/lgl/graphopt
and graphviz neato's default MODE_MAJOR which uses INIT_RANDOM):

  Single-seed Procrustes RMSD confounds "algorithmic divergence" with
  "random-init basin difference." The right test is statistical
  equivalence: do dagua and graphviz produce layouts drawn from
  comparable distributions? Multi-seed + TOST equivalence test.

Round 8 builds the multi-seed infrastructure and re-evaluates
fdp/sfdp/neato residuals from Rounds 4-7 under that lens.

## What exists

- `scripts/fidelity_analysis.py` already imports TOST:
  ```
  from statsmodels.stats.weightstats import ttost_ind
  ```
  With margin factors 0.5x/1x/1.5x/2x (lines 60-67) and existing
  per-metric Procrustes within-vs-between TOST infrastructure
  (line 1296 area).
- `eval_output/benchmark_full/positions/` has 9 seeds per stochastic
  engine per graph: `<graph>__<engine>__seed42.pt` through `seed50.pt`
  (and the no-suffix file = seed42 typically).
- `dagua.eval.variants.VARIANT_REGISTRY` has `is_stochastic` flag on
  103 of 112 variants.

## What to build

### Step 1: Multi-seed live_compare (`scripts/algo_fidelity_live_compare.py` upgrade)

Add to the existing single-seed comparator a `--seeds N` CLI flag:

- For each graph in scope:
  - **dagua side**: run the dagua engine with N seeds (e.g., 42..42+N-1).
    For deterministic engines (variant.is_stochastic == False), N=1 is
    forced. For stochastic engines, run N=N.
  - **graphviz side**: load up to N cached positions from
    `eval_output/benchmark_full/positions/<graph>__<target>__seed<S>.pt`
    for S in [42..50]. If fewer than N seeds exist, use what's available.
  - Compute pairwise Procrustes RMSD: N_dagua x N_graphviz pairs per graph.
    For each pair, record (graph, dagua_seed, graphviz_seed, rmsd).
  - Compute "within-graphviz floor": pairwise RMSDs among graphviz seeds
    (N_graphviz choose 2). This is the stochastic floor -- the minimum
    RMSD you'd see between two runs of the SAME algorithm on the SAME
    graph with different seeds.

- Output: `<output-dir>/multi_seed_rmsd.csv` with columns
  (graph, side, seed_a, seed_b, rmsd) where side is one of
  "dagua_vs_graphviz", "within_graphviz", "within_dagua"
  (last useful as sanity check).

- Output summary `<output-dir>/multi_seed_summary.json` per graph:
  - n_dagua_seeds, n_graphviz_seeds
  - mean / median / p95 of dagua_vs_graphviz RMSDs
  - mean / median / p95 of within_graphviz RMSDs (the stochastic floor)
  - **TOST equivalence verdict** at margins {0.5x, 1x, 1.5x, 2x}
    of the within_graphviz floor: are dagua_vs_graphviz means within
    margin of within_graphviz means?
  - Verdict label per family: `equivalent_at_<margin>x` or `not_equivalent`.

For deterministic engines the multi-seed mode degenerates to
single-seed (skip TOST, single RMSD value reported).

### Step 2: Re-evaluate parked families

Run multi-seed analysis on:
- `classic_fmmm` vs `graphviz_fdp` (currently parked at flail=2,
  median 0.247, "uniform >0.15 floor")
- `classic_sfdp` vs `graphviz_sfdp` (currently parked at flail=1,
  median 0.092)
- `classic_stress_maj` vs `graphviz_neato` (Round 7 OUTLIER_RESIDUAL,
  median 0.035, worst 0.382)
- `classic_classical_mds` vs `graphviz_neato` (Round 7
  OUTLIER_RESIDUAL, median 0.045, worst 0.333)

Use --seeds 5 minimum (more if fast enough, max 9 since cache has 9).

### Step 3: Write re-evaluation report

`.project-context/research/sprint_algo_fidelity/ROUND_8_RE_EVAL.md`:
- Per-pairing table:
  | Pairing | within-graphviz floor (median) | dagua-vs-graphviz (median) | TOST verdict |
- For each parked residual, state whether the multi-seed evidence
  changes the classification:
  - "equivalent_at_1x" or better -> reclassify as faithful, mark
    family CONVERGED with classification update
  - "equivalent_at_2x" -> partial reclassification (within stochastic
    range, but at the upper bound of margin)
  - "not_equivalent" -> classification stands; the uniform floor is
    real algorithmic divergence
- For graphs that fail the worst-graph criterion: report whether
  graphviz-vs-graphviz floor on that specific graph is also >= 0.15
  (which would mean "the algorithm itself is unstable on this graph
  -- not a dagua bug").

### Step 4: Update STATE.md

For any family whose verdict changes from RESIDUAL to CONVERGED or
vice versa, update the iteration log. Specifically:
- If fdp now reads as "equivalent_at_<=2x of within-graphviz" and
  within-graphviz floor on the affected graphs is comparable to
  the dagua-vs-graphviz mean: un-park fdp and mark as CONVERGED with
  new classification "stochastic_floor_match".
- Same logic for sfdp.
- Same logic for neato outliers.

### Step 5: Per-round summary

Write `eval_output/algo_fidelity/round_8/SUMMARY.md` with the
re-evaluation outcome per family. Keep concise (1 paragraph per
family).

### Step 6: Tests + commit

```
pytest tests/test_layout/ -x --tb=short -q 2>&1 | tail -20
```

If the multi-seed comparator code change is non-trivial:
```
feat(fidelity): round 8 -- multi-seed comparator + TOST re-evaluation

- scripts/algo_fidelity_live_compare.py: --seeds N flag, dagua-vs-graphviz +
  within-graphviz pairwise RMSD distributions, TOST equivalence test at
  margins 0.5x/1x/1.5x/2x.
- Re-evaluation of fdp/sfdp/neato residuals under stochastic-floor lens.
- Verdict changes: <list any reclassifications>
- Tests: <count> passed
```

If the only change is comparator infra (no verdict shifts), still
commit -- the infra is reusable.

## Strategic note

This is the round that determines whether dagua's "drop-in graphviz
replacement" claim holds for stochastic families. If the multi-seed
analysis shows dagua-vs-graphviz RMSD distributions are within the
within-graphviz stochastic floor, dagua is faithful even where
single-seed Procrustes looked bad. If not, the architectural
mismatches (random init, sequential updates, FR force law) really do
matter and need bigger fixes.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- Any pipeline file from prior rounds (sugiyama, fmmm, sfdp, stress) --
  Round 8 is INFRA + RE-EVALUATION only, not algorithm fixes

**Allowed in Round 8:**
- `scripts/algo_fidelity_live_compare.py` (PRIMARY upgrade)
- `scripts/algo_fidelity_cross.py` (if a related upgrade helps)
- `eval_output/algo_fidelity/round_8/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`

**Out of scope this round:**
- Pipeline algorithm changes
- New variant registry entries
- run_benchmark execution
</scope_constraints>

<default_follow_through_policy>
This is a measurement-methodology round. The output is BETTER MEASUREMENT,
not algorithm fixes. The test for whether Round 8 succeeded is whether
we can correctly classify each parked residual as either:
- Stochastic-floor faithful (dagua matches graphviz within seed-noise)
- Real algorithmic divergence (dagua differs beyond seed-noise)

Either outcome is valuable. Don't try to "improve" anything -- just
measure cleanly.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED**: comparator upgraded, re-evaluation done, RE_EVAL.md
   written, SUMMARY.md written, STATE.md updated with any reclassifications,
   commit on develop.
2. **BLOCKED**: ROUND_8_BLOCKED.md if multi-seed cache turns out
   inaccessible / inconsistent.
</completeness_contract>

<verification_loop>
- live_compare with --seeds 5 must run cleanly on at least one
  pairing without errors
- multi_seed_summary.json must contain TOST verdicts for each
  pairing and graph
- pytest unaffected since this is infra-only
</verification_loop>

<missing_context_gating>
ABORT if:
- benchmark_full positions don't have multi-seed cache for graphviz
  (verify via `ls eval_output/benchmark_full/positions/ | grep seed4`)
- statsmodels.stats.weightstats.ttost_ind isn't usable (it IS used
  in fidelity_analysis.py already so should work)

Write ROUND_8_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF infra is built + re-evaluation produces
  meaningful output.
- No pipeline edits.
- Never delete eval_output files.
</action_safety>
