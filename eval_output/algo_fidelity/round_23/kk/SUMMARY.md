# Round 23 KK Summary

## Scope

Primary diff: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_kk.md`.
Round 22 already committed ranked items 1-3 in `fbaee2a`. Round 23 applied the
remaining small, technically feasible KK items and re-hardened duplicate-edge
adapter semantics after later shared NetworkX adapter work changed the default
duplicate policy.

## Measurements

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_kk nx_kamada_kawai \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_23/kk/{baseline,post_fix}
```

| Run | Rows | Median | P25 | P75 | P95 | Worst |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 30 | 0.0000000387 | 0.0000000102 | 0.0000000418 | 0.0000000567 | `tl_mlp_3layer` 0.0000000567 |
| Post-fix | 30 | 0.0000000387 | 0.0000000102 | 0.0000000418 | 0.0000000567 | `tl_mlp_3layer` 0.0000000567 |

Result: unchanged, already-zero bounded subset. The committed value is
fidelity infrastructure and adversarial regression coverage.

## Ranked Items

| Item | Estimate | Status |
|---|---:|---|
| 1. Align `steps` semantics with NetworkX defaults | S | Already addressed in Round 22 commit `fbaee2a`; verified by existing/default tests. |
| 2. Fix weighted duplicate-edge semantics | M | Round 22 addressed Dagua KK distances. Round 23 commit `e96574a` also set `NetworkXKamadaKawai.duplicate_policy = "last"` after later shared NetworkX adapter work introduced sum-collapse for other families. |
| 3. Disable adapter-level orientation postprocess | S | Already addressed in Round 22 commit `fbaee2a`; verified by adapter-default regression. |
| 4. Normalize adapter scale policy | S | Addressed in `e96574a`: NetworkX KK now uses unit `output_scale = 1.0`, while other NetworkX adapters keep display-scale defaults. |
| 5. Preserve float64 through audit comparisons | S/M | Addressed in `e96574a`: KK finalization has opt-in float64 output via `preserve_float64`; NetworkX adapter supports `output_dtype`. |
| 6. Negative-weight validation | S | Addressed in `e96574a`: pipeline and raw KK distance op reject negative weighted shortest-path inputs. |
| 7. Add duplicate/capped-iteration regressions | M | Addressed in `e96574a`: kept duplicate last-write coverage, added NetworkX adapter duplicate/scale tests, dtype tests, negative-weight tests, and capped `maxiter` forwarding coverage. |

## Skipped / Deferred

- 1D/3D KK parity with NetworkX: deferred. This is a larger API expansion and
  outside the current 2D `classic_kk` vs `nx_kamada_kawai` fidelity pairing.
- Node-keyed `pos` dict interface parity: deferred. Dagua's tensor-based API is
  intentional in this engine path; adding dict input would be adapter/API work
  beyond the small remaining ranked fixes.
- Exact NumPy final rescale backend: deferred. Float64 audit mode reduces dtype
  loss, but replacing torch final rescale with NumPy would be a broader behavior
  change for direct pipeline callers and has no measurable residual here.
- Non-default NetworkX `center` parity: deferred. Dagua KK does not expose a
  center parameter in this path.

## Verification

- `ruff check dagua/eval/competitors/networkx_competitor.py dagua/layout/ops/pipelines/kk.py dagua/layout/ops/distance.py dagua/layout/ops/postprocess.py tests/test_layout/test_kk_fidelity.py --fix`: passed.
- `pytest tests/test_layout/test_kk_fidelity.py tests/test_pipeline_kk.py -x --tb=short -q`: passed, `20 passed in 0.29s`.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- Post-commit `pytest tests/test_layout/ -x --tb=short -q -k "kk"`: passed, `9 passed, 326 deselected in 0.30s`.
- Final `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: blocked during collection by unrelated `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'` in `tests/test_classic_drl.py:10`.
- Final `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: attempted once; interrupted after more than 10 minutes without new progress output while multiple parallel Codex runs were active in the same checkout.

## Commit

- `e96574a` — `feat(fidelity): round 23 kk -- finish parity hooks`

## Concerns

- The repository had unrelated staged and unstaged changes from parallel work.
  The kk commit used `git commit --only` with kk-scoped files and did not stage
  cosmetic files.
- `eval_output/` is gitignored, so this summary and CSV/JSON measurement files
  must be force-added if the orchestrator wants them committed later.
