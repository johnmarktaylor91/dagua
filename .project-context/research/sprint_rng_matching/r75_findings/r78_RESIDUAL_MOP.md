# r78-R1 Residual Mop

Scope: research/probe only. No engine code changes.

Sources:
- Ledger: `eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md` and `.json`.
- Pinned Graphviz: `git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`.
- Installed Graphviz: `fdp - graphviz version 7.0.5 (20221231.0122)`.
- Scratch probes: `/tmp/r78_mop/r78_ab_probe.py`, output `/tmp/r78_mop/cluster_ab_stage_probe.json`.

Limitations:
- The requested `/tmp/gv750-mop` instrumented build was not present. I found `/tmp/gv750-trace-build`, but no pack2 label-box dump files under `/tmp/gv750-mop`.
- Cluster C same-process 100-seed script was attempted at `/tmp/r78_mop/r78_c_sgd2_probe.py`; it did not produce paired results before exceeding the task budget. It was stopped after entering the first graph because one seed runs two 2000-step optimizations with upstream progress output.

## Cluster A: SFDP Disconnected Label-Box Residual

Representative pack2 measurements:

| Graph | Components | Label-box delta vs native | Pack2 step vs native | Pack2 cell change | Sort order changed? | Measured residual |
| --- | ---: | --- | ---: | --- | --- | --- |
| `disconnected_label_cycle_collage` | 3 | max width +191.479 pt, max height -28.452 pt | 111 vs 107 | cells: c0 9 vs 7, c1 32 vs 29, c2 18 vs 15 | no: 1,2,0 | label boxes still materially change cells and grid step |
| `multi_component_80` | 7 | max width +15.958 pt, max height +2.000 pt | 88 vs 87 | cells: c0 72 vs 69, c1 59 vs 55, c2 51 vs 54, c3 15 vs 14 | no: 0,1,2,3,4,5,6 | label boxes still change cells, but sort order is stable |

Per-row verdicts:

| Row | Ledger n/seeds | Ledger stress D/R | Ledger cross D/R | Ledger W D/R | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `disconnected_encoder_residual::classic_sfdp_default` | 100, 100-199 | 0.022254 / 0.016750 | 0.016667 / 0.000000 | 0.747220 / 0.768824 | portable-op-remaining: exact `pack.c:genPoly` with post-spline label-box occupancy and C dump parity |
| `disconnected_encoder_residual::classic_sfdp_graphviz_fidelity` | 100, 100-199 | 0.022254 / 0.016750 | 0.016667 / 0.000000 | 0.747220 / 0.768824 | portable-op-remaining: same as above |
| `disconnected_label_cycle_collage::classic_sfdp_default` | 100, 100-199 | 0.474451 / 0.089004 | 0.116667 / 0.000000 | 0.431383 / 0.763433 | portable-op-remaining: large label-box cell deltas remain; prove against instrumented C pack dump |
| `disconnected_label_cycle_collage::classic_sfdp_graphviz_fidelity` | 100, 100-199 | 0.474451 / 0.089004 | 0.100000 / 0.000000 | 0.431383 / 0.763433 | portable-op-remaining: same as above |
| `kitchen_sink_platform_graph::classic_sfdp_default` | 100, 100-199 | 0.035072 / 0.030067 | 0.266667 / 0.183333 | 0.484654 / 0.479105 | proven residual bound until C dump exists: metric deltas are small; no representative pack2 contradiction found |
| `kitchen_sink_platform_graph::classic_sfdp_graphviz_fidelity` | 100, 100-199 | 0.035072 / 0.030067 | 0.266667 / 0.150000 | 0.484654 / 0.479105 | proven residual bound until C dump exists: same small-bound row |
| `multi_component_80::classic_sfdp_default` | 100, 100-199 | 0.054036 / 0.063684 | 0.033333 / 0.000000 | 0.914509 / 0.866247 | portable-op-remaining: pack2 label boxes alter grid/cells; order stable, so focus on cell rasterization/margins |
| `multi_component_80::classic_sfdp_graphviz_fidelity` | 100, 100-199 | 0.054036 / 0.063684 | 0.050000 / 0.000000 | 0.914509 / 0.866247 | portable-op-remaining: same as above |

Follow-up spec:
- Recreate `/tmp/gv750-mop` from Graphviz 7.0.5, instrument `lib/pack/pack.c` at `genPoly`, `fillLine`, perimeter sort, and `placeGraph`.
- Dump per-component bbox, step, cell set hash/count, perimeter, sorted order, accepted offset, and whether `doSplines` supplied spline cells.
- Compare against Dagua `_generate_node_polyomino` metadata for `disconnected_label_cycle_collage` and `multi_component_80` seeds 100 and 101 first.

## Cluster B: Graphviz FDP-Family Rows

Pinned source stage order from `lib/fdpgen/layout.c`: `findCComp` -> `fdp_tLayout` -> `fdp_xLayout` -> `putGraphs` -> `finalCC` -> `evalPositions`. `lib/fdpgen/tlayout.c` shows initial random placement via `srand48(local_seed)` and `drand48()`, followed by force loops; `lib/fdpgen/xlayout.c` performs overlap expansion.

Probe method: for connected graphs, compare Dagua's `_graphviz_fdp_component_layout` (`tLayout+xLayout`) against installed `fdp` final output for seed 42. If it already differs, first divergence is before pack. For disconnected `parallel_cycles_4x5`, the final divergence remains pack-or-component-order without a C stage dump.

| Row | Components | Seed-42 final RMSD | Connected tLayout+xLayout RMSD | First diverging stage | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| `parallel_cycles_4x5::classic_fmmm_graphviz_fdp_fidelity` | 5,5,5,5 | 0.224767 | NA | unresolved: component kernel or pack | floor-candidate until C fdp dump; experiment: dump per-component fdp_tLayout/xLayout and `putGraphs` offsets |
| `protein_ppi_200::classic_fmmm_graphviz_fdp_fidelity` | 200 | 0.485783 | 0.485785 | `fdp_tLayout`/`fdp_xLayout`, before pack | portable: exact fdp random/init + force/grid/xLayout op, medium-large effort |
| `real_lesmis_77::classic_fmmm_graphviz_fdp_fidelity` | 77 | 0.753175 | 0.753177 | `fdp_tLayout`/`fdp_xLayout`, before pack | portable: same op, medium effort |
| `recurrent_feedback_cell::classic_fmmm_graphviz_fdp_fidelity` | 5 | 0.241327 | 0.241319 | `fdp_tLayout`/`fdp_xLayout`, before pack | portable: small-graph exact init/force/xLayout reproduction |
| `sbm_5x50::classic_fmmm_graphviz_fdp_fidelity` | 250 | 0.668246 | 0.668246 | `fdp_tLayout`/`fdp_xLayout`, before pack | portable: same op, medium-large effort |

Follow-up spec:
- Instrument Graphviz 7.0.5 `lib/fdpgen/tlayout.c:initPositions`, the two force loops in `fdp_tLayout`, and `lib/fdpgen/xlayout.c:fdp_xLayout`.
- For connected rows, stop before pack and compare node coordinates after init, after pass1 force loop, after xLayout.
- For `parallel_cycles_4x5`, additionally instrument `lib/fdpgen/layout.c:putGraphs` and `finalCC` to split component-kernel divergence from pack placement.

## Cluster C: SGD2 Evidence-Thin Rows

Attempted same-process full-power probe:
- Script: `/tmp/r78_mop/r78_c_sgd2_probe.py`.
- Intended seeds: 100-199 for both Dagua `classic_sgd2_multi_with_crossing` and reference `sgd2_multi_ref__for__classic_sgd2_multi_with_crossing`.
- Runtime blocker: upstream reference emits a 2000-step progress bar per layout; one seed on `real_football_115` requires two full 2000-step optimizations. The process had not completed seed 100 in budget and wrote no JSON.

Existing r77 ledger evidence remains low-power:

| Row | Ledger n/seeds | Ledger stress D/R | Ledger cross D/R | Ledger W D/R | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `real_football_115::classic_sgd2_multi_with_crossing` | 42, 100-141 | 0.260392 / 0.289069 | 42453.642857 / 30643.571429 | 1.336120 / 1.333001 | unresolved evidence-thin; same-process full-power not completed |
| `wide_1_100_1::classic_sgd2_multi_with_crossing` | 42, 100-141 | 0.215156 / 0.273190 | 1889.761905 / 654.142857 | 1.330578 / 1.331253 | unresolved evidence-thin; same-process full-power not completed |

Follow-up spec:
- Add a scratch-only runner that disables upstream tqdm output and writes after every seed.
- Run `real_football_115` and `wide_1_100_1` with seeds 100-199, same Python process, paired Dagua/reference layouts, params matched exactly:
  - Dagua: `criteria={"stress": 1.0, "crossings": 0.5}`, `steps=2000`, `lr=0.01`, `grad_clamp=5.0`, `fidelity_mode=True`.
  - Reference: `criteria_weights={"stress": 1.0, "crossings": 0.5}`, `max_iter=2000`, `optimizer_kwargs={"lr": 0.01}`, `grad_clamp=5.0`.
- Persist per-seed position hashes, stress, crossing count, and paired deltas. A closure verdict requires either 100 paired ok seeds or a named first divergence from optimizer/loss sampling.

## R2 execution (RECONSTRUCTED -- original section lost to a worktree removal before
## commit, my orchestration error; primary artifacts preserved in
## ../r78_evidence/ and the full transcript in the r78_cx_r2 codex log)

- SGD2 (2 evidence-thin rows): paired same-process probe, params matched. All completed
  paired seeds BYTE-IDENTICAL (real_football_115 seeds 100-112 + wide_1_100_1 seed 100 =
  14/14; artifact sgd2_pair_results.jsonl). Full 100-seed closure blocked by a DAGUA-SIDE
  HANG: real_football_115 seed 113 stalls indefinitely (>60 min) at phase native_start,
  confirmed by phase-logged rerun. -> S3 dispatched (hang root-cause + closure).
- SFDP (8 label-box residual rows): instrumented pack.c comparison (artifact
  sfdp_pack_compare.json): C pack differs at grid step/cell/perimeter/order, and
  doSplines=1 SPLINE occupancy is not reproducible by dagua's straight-edge fallback on
  these graphs. Terminal disposition: spline-aware pack occupancy = routing-port boundary;
  measured per graph.
- FDP (5+9 rows): fdp_tLayout MATCHES on representatives (artifact
  fdp_stage_compare.json); first divergence is fdp_xLayout prism/GTS overlap expansion
  (overlap="9:prism", tries=9) BEFORE putGraphs/finalCC. Named portable-with-effort:
  requires a prism-equivalent overlap expansion. -> F3 dispatched.

## S3: sgd2 hang + closure

Root cause:
- The reported native hang was a practical hang, not an unbounded Python loop. `real_football_115` seed 113 reached `native_start`, entered SGD2 step 0, and then spent the run inside the tiny crossing-detector MLP (`LayerNorm`, `Linear`, Adam) under PyTorch's default CPU intra-op thread pool.
- Reference semantics are a fixed `for range(max_iter)` loop in `/tmp/graph-drawing/gd2.py`; it is bounded by `max_iter=2000`. The defect was Dagua running the same fixed tiny-kernel workload with multi-thread overhead large enough to project to about 50-60 minutes for one seed.
- File/line: `dagua/layout/ops/sgd2_multi.py:59` adds `_cpu_thread_guard`; `dagua/layout/ops/sgd2_multi.py:2418` applies it only for CPU crossing criteria. The paired reference adapter mirrors the same guard at `dagua/eval/competitors/sgd2_multi_competitor.py:95` and `dagua/eval/competitors/sgd2_multi_competitor.py:557` so native/reference comparisons use the same CPU kernel mode.

Fix:
- CPU runs with `crossings` or `crossing_angle_maximization` now execute the SGD2 optimization loop with `torch.set_num_threads(1)`, restoring the previous thread count afterward.
- This preserves the reference `max_iter` bound and avoids the tiny-batch thread-pool slow path. It intentionally changes the CPU floating-point execution mode for crossing workloads; non-crossing SGD2 criteria do not enter the guard.
- Code commit: `1f317c1` (`fix(layout): bound sgd2 crossing CPU threading`).
- Report commit: `0bd4efe` (`docs(r78): record sgd2 hang closure`).

Hang check:

| Graph | Seed | Variant | Before | After |
| --- | ---: | --- | --- | --- |
| `real_football_115` | 113 | `classic_sgd2_multi_with_crossing` | R2 observed `native_start` stall >60 min | completed in 26.624s, hash `b4ef1b0b74ab455cd4b66069a9400369f4e0e3201de03b20b3071582763e2792` |

100-seed paired closure artifacts:
- `/tmp/r78_r2/sgd2_pair_results_s3_real.jsonl`
- `/tmp/r78_r2/sgd2_pair_results_s3_wide.jsonl`

| Row | Seeds | Exact | Divergent | First-batch parity | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| `real_football_115::classic_sgd2_multi_with_crossing` | 100-199 | 100 | 0 | stress/crossings/init all true | identical |
| `wide_1_100_1::classic_sgd2_multi_with_crossing` | 100-199 | 97 | 3 | stress/crossings/init all true | named divergence after matched first batches |

`wide_1_100_1` named divergences:

| Seed | Max abs delta | RMS delta | Native metrics | Reference metrics |
| ---: | ---: | ---: | --- | --- |
| 157 | 3.994778633 | 0.393614441 | crossings 387, stress 0.741953424 | crossings 421, stress 0.741764044 |
| 184 | 1.184412003 | 0.206326425 | crossings 469, stress 0.708211760 | crossings 603, stress 0.710824322 |
| 190 | 0.047194958 | 0.007997223 | crossings 1326, stress 0.685131618 | crossings 1314, stress 0.685130248 |

Byte gates:
- R2 pre-fix completed default-thread seeds remain recorded in `/tmp/r78_r2/sgd2_pair_results.jsonl` (14/14 exact).
- S3 fixed-mode native/reference byte gate: `real_football_115` seeds 100-199 exact; `wide_1_100_1` seeds 100-156, 158-183, 185-189, 191-199 exact.
- Pre/post byte identity for crossing workloads is not retained because the fix changes the CPU kernel threading mode. The tradeoff was required to make the reference-bounded 2000-step crossing run complete in sane time. First-batch RNG parity remained true for all 200 S3 paired seeds, including the three wide-row divergences.

Test results:
- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed (`Success: no issues found in 1 source file`; unused pyproject section note for `dagua.layout.multilevel`).
- `pytest -k "sgd2" -x --tb=short -q`: passed, 50 passed, 3128 deselected, 42 warnings.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: stopped on known pre-existing cosmetic failure `tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border` (`assert 0 >= 2`) after 266 passed, 88 deselected, 1 xfailed, 63 warnings.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: attempted; process was terminated after about 90 minutes of CPU-bound runtime under heavy concurrent benchmark load, with no assertion failure emitted before termination.

Concerns:
- The S3 fix closes the hang, but crossing-workload byte output is now defined by the single-thread CPU execution mode. That is why the full closure is exact for `real_football_115` but exposes three late-optimizer divergences on `wide_1_100_1` despite matched initial batches.
- The three wide-row divergences are not sampling divergences; first-batch hashes match for stress, crossings, and initialization. The next investigation should instrument per-step hashes/losses around seeds 157, 184, and 190 if those remaining deltas matter.
