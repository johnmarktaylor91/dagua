# Round 23 RT Summary

## Changes

- Commit `3c89768` enabled `classic_rt`'s igraph fidelity mode by default for
  the competitor adapter.
- Commit `3c89768` exposed RT `roots`, `rootlevel`, `center_output`, and
  `output_scale` pipeline/config controls for controlled igraph-parity fixtures.
- Commit `3c89768` made igraph fidelity output uncentered and scaled by `50.0`
  by default, matching the igraph adapter's raw coordinate policy.
- Commit `3c89768` set `IgraphRT.layout_kwargs = {"mode": "out"}` and allowed
  `mode`, `root`, and `rootlevel` as igraph variant parameters.
- Commit `3c89768` added regression tests for scaled/uncentered igraph units and
  explicit root/rootlevel traversal.

## Ranked Items

1. `#4 Implement igraph's offset/contour algorithm exactly`: skipped. Estimated
   size is 250-350 net LOC plus golden tests, above the requested `< ~200`
   threshold. Existing Walker/Buchheim contour remains.
2. `#5 Expose/pass root and rootlevel`: addressed in `3c89768`. Dagua RT now
   accepts `roots` and `rootlevel`; igraph RT adapter exposes `root` and
   `rootlevel` variant params. Net size: about 90 LOC.
3. `#6 Clarify and test Python igraph's default mode`: addressed in `3c89768`.
   Runtime inspection showed python-igraph 1.0.0 exposes `mode`; default parity
   path is now explicit `mode="out"`. Net size: 2 LOC.
4. `#7 Handle duplicate-edge root scoring consistently`: already addressed by
   Round 22 commit `811ab39`; Round 23 kept the regression coverage passing.
   No additional source change was needed.
5. `#8 Move final centering/scaling behind a fidelity switch`: addressed in
   `3c89768`. Igraph fidelity mode now defaults to uncentered, `50.0`-scaled
   output, with overrides available. Net size: about 20 LOC.

Lower-priority verified value:

- Enabling `classic_rt`'s igraph fidelity mode by default improved the bounded
  subset post-fix RMSD without regression.

## Baseline

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_rt igraph_rt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/rt/baseline
```

Output:

```text
graphs: 5
median: 0.000000
p25: 0.000000
p75: 0.074164
p95: 0.194303
worst: mixed_width_labels 0.224338
```

## Post-Fix

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_rt igraph_rt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/rt/post_fix
```

Output:

```text
graphs: 5
median: 0.000000
p25: 0.000000
p75: 0.000000
p95: 0.000000
worst: mixed_width_labels 0.000000
```

## Verification

- `ruff check dagua/layout/ops/coordinate.py dagua/layout/ops/pipelines/reingold_tilford.py dagua/eval/competitors/classic_competitor.py dagua/eval/competitors/igraph_competitor.py tests/test_layout/test_rt_fidelity.py --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/test_rt_fidelity.py -q`: `6 passed`.
- `pytest tests/test_layout/ -x --tb=short -q -k "rt"` after commit
  `3c89768`: `27 passed, 307 deselected`.

## Concerns

- The exact igraph contour port remains the only skipped ranked item. It is
  larger than the Round 23 size limit and should be a dedicated follow-up if
  future non-bounded fixtures still show residuals.
- The worktree contained concurrent non-RT edits from other fidelity families;
  only RT hunks were staged for `3c89768`.
