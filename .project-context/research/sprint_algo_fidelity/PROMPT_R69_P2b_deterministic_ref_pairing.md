<task>
Fix a seed-pairing bug in `scripts/fast_fidelity_report.py` that makes ~50 fidelity
variants get NO verdict, then re-run the report (positions already exist -- do NOT
re-run the benchmark).

## The bug
The report pairs each reimplementation (`classic_*`) layout against its reference adapter
per-seed for Procrustes RMSD. The pairing loop (~lines 114-140):

    reimp_seeds = seeds_per_pair.get((reimp, graph), set())   # e.g. {42,43,44,45,46}
    ref_seeds   = seeds_per_pair.get((ref, graph), set())     # e.g. {None}
    common = sorted(reimp_seeds & ref_seeds)[: args.max_seeds]
    if not common:
        skipped_no_pair += 1
        continue
    for seed in common:
        reimp_key = f"{graph}::{reimp}::seed{seed}"
        ref_key   = f"{graph}::{ref}::seed{seed}"

DETERMINISTIC reference adapters (igraph_sugiyama, ogdf_gem, ogdf_pivot_mds, graphviz_sfdp,
graphviz_neato, igraph_mds, ogdf_stress, ogdf_fmmm, ...) are run ONCE with `seed=None` (they
ignore the seed). The reimpl variants ran at int seeds 42-46. So `{42..46} & {None}` is EMPTY
-> counted as `skipped_no_pair`, no verdict. Last run logged "no-pair skips: 4278", leaving
~50 variants (sugiyama, sfdp, neato, gem, pivot_mds, classical_mds, stress_maj, fmmm,
maxent_stress, sgd2_multi families, etc.) UNCLASSIFIED -- mostly the graphviz/igraph/ogdf
bit-exact ports that should be near-perfect.

## The fix (pairing logic ONLY -- do not touch RMSD math or thresholds)
When a reference has only a deterministic result (`None in ref_seeds and not (reimp_seeds &
ref_seeds)`), pair EACH reimpl seed against that single deterministic ref result:
- iterate reimpl seeds (int, capped at max_seeds; if reimpl is also {None}, use None once);
  for each: reimp_key=f"{graph}::{reimp}::seed{seed}" (or f"{graph}::{reimp}" if seed is None),
  ref_key=f"{graph}::{ref}::seedNone" with fallback to the seedless key f"{graph}::{ref}"
  (the existing position fallback near line 130 already does this -- ensure this path uses it).
  Comparing each reimpl seed against the single deterministic ref output is correct.
- Symmetric: if the reimpl is deterministic ({None}) but ref ran int seeds, pair the reimpl's
  single result against each ref int seed.
- Keep the existing int-seed-overlap path UNCHANGED for stochastic-vs-stochastic pairs.
- Do NOT call sorted() on a set containing None (guard the mixed None/int case).
- Count these deterministic pairings in total_pairs; in failures use the reimpl seed (or 'None').

## Re-run after fixing (do NOT re-run the benchmark)
    export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:$LD_LIBRARY_PATH
    python3 scripts/fast_fidelity_report.py \
      --results eval_output/benchmark_5seed_fidelity/results.json \
      --positions eval_output/benchmark_5seed_fidelity/positions.h5 \
      --output eval_output/fidelity_report_r69/stage1 \
      --max-seeds 5 --bit-exact-threshold 1e-3

## Verify + report
- Print the new verdict totals (MACHINE_EPSILON / BIT_EXACT / STRONG_EQUIV / PARTIAL counts)
  and the new "no-pair skips" number (should drop dramatically from 4278).
- Confirm previously-missing variants now appear (e.g. classic_sugiyama_default,
  classic_sfdp_default, classic_neato, classic_gem_iters100, classic_pivot_mds_10,
  classic_classical_mds_default, classic_stress_maj_default).
- Write a one-paragraph note to eval_output/fidelity_report_r69/p2b_pairing_fix.md describing
  the fix and the before/after verdict counts.

## Constraints
- Pure measurement/report fix. Do not change the benchmark, the algorithms, variants.py, or
  any fidelity thresholds. Do not delete or re-run benchmark data.
