# Native Cost Model Calibration (C2, fCoSE refit) - 2026-07-21

## Scope

C2 refits ONLY the fCoSE generation terms of
`dagua.layout.ops.pipelines.native_cost_model.FROZEN_COST_TABLE`, replacing the
C1 single linear prior with a piecewise exact/Barnes-Hut regime split. All
other families (stress, arm_s, apsp, ruler, w5, directed_*, opaque) retain
their C1-2026-07-20 constants unchanged; see
`native_cost_model_calibration_2026-07-20.md` for those.

## Why the C1 fCoSE prior was wrong

C1 froze one line, `alpha * (N+E) * steps + beta` with
`("fcose","cuda") alpha=0.0012`, anchored so `r8_nested_scale_1k_budget`
priced fCoSE out. The M2 megasprint forensics (M2_ANALYSIS_FABLE.md) proved
this line over-priced small/medium rows by 2-4 orders of magnitude
(`r79_undirected_sbm_low_mix_4x25`: predicted 2229.7 DWU vs 0.32s real), so
the ledger starved the fCoSE contest arms that were the actual winners on four
rows (`r8_nested_parent_child_backedges`, `kitchen_sink_hybrid_net`,
`r79_undirected_sbm_low_mix_4x25`, `random_bipartite_60`), each recoverable
exactly by admission alone.

The physical cause is a regime cliff in the fCoSE pipeline itself
(`FCoSEConfig.max_exact_repulsion_nodes = 512`, the marketplace seam uses the
default): at or below 512 nodes the spring embedder runs exact vectorized
pairwise repulsion and a full 2500-step arm is overhead-dominated
(~0.2-0.6s on this box); above 512 the per-step Barnes-Hut tree pass dominates
and cost jumps 3-4 orders of magnitude (~4.3 s/step at n=800, ~6.5 s/step at
n=1000, i.e. ~16,000s for a full 2500-step arm on cuda). No single (alpha,
beta) line can represent both ends.

## Model change

- New frozen constant `FCOSE_EXACT_REPULSION_NODE_CAP = 512` mirroring the
  pipeline default.
- fCoSE generation prices through `"force"` (exact regime, N <= 512) or
  `"force_bh"` (Barnes-Hut regime, N > 512); both keep the linear
  `alpha * (N+E) * steps + beta` term form. A custom table without
  `"force_bh"` falls back to `"force"` (never an accidental zero cost).
- `estimate_native_work_cost` metadata records `fcose_regime` for decision
  logs and replay.

## Box

- Date: 2026-07-21
- Host: local development workstation for `/home/jtaylor/projects/dagua`
  (same box as C1), 20 cores, CUDA device visible; ~15 cores idle during
  harvest.
- Python: `/home/jtaylor/anaconda3/envs/py311/bin/python`
- Branch: `codex/r0-determinism` at base `9b5d4c26`
- `torch.set_num_threads(1)`, CUDA warmup run before sampling.

## Telemetry

Harvested by `~/.claude/research/dagua/megasprint/c2_measure_fcose.py`, which
mirrors the marketplace seam call exactly
(`layout_fcose_pipeline(quality="default", randomize=True)`,
`FCOSE_REFERENCE_STEPS = 2500`):

- `~/.claude/research/dagua/megasprint/c2_fcose_telemetry_exact.jsonl`:
  44 samples, N in {10..500} x {cuda, cpu}, full 2500 reference steps
  (plus one er_500 cpu 500-step sample), 2 seeds each.
- `~/.claude/research/dagua/megasprint/c2_fcose_telemetry_bh.jsonl`:
  8 samples, sbm_8x100 (N=800) and r8_nested_scale_1k_budget (N=1000) at
  25/50 steps. Short-step samples extrapolate linearly in planned steps
  because the Barnes-Hut per-step cost is stationary; full 2500-step runs
  are ~3-16 kilo-seconds and were not run.

Key measured anchors (cuda): sbm_low_mix (N=100, E=642) 0.25s/arm;
rgg_500 (N=500, E=3491) 0.54s/arm; scale_1k 6.5 s/step.

## Fit

`scripts/calibrate_native_cost_model.py` with per-term fit modes
(`TERM_FIT_MODES`), envelope 2.0 (fcose family default):

- `force`: **intercept_slope** -- raw P10 seconds as overhead beta, P90 of
  overhead-subtracted rates as alpha, both then enveloped. The C1 `p90_rate`
  fit anchored alpha on tiny-row raw rates (overhead divided by tiny volume),
  which would over-price the largest exact-regime rows ~200x and leave
  admission on a knife edge.
- `force_bh`: **rate** -- zero-intercept P90 rate, enveloped. The 25/50-step
  sample seconds are pure per-step cost; a P10-seconds beta would be a
  measurement artifact.
- All other terms keep the C1 `p90_rate` methodology (`p90_rate` remains the
  default mode).

## Frozen constants (verbatim calibrate output)

| Key | force (alpha, beta) | force_bh (alpha, beta) |
| --- | --- | --- |
| `("fcose","cpu")` | (5.49172932331e-06, 0.2094) | (0.00164199078341, 0.0) |
| `("fcose","cuda")` | (3.12987012987e-07, 0.3726) | (0.00427751152074, 0.0) |

`score` terms unchanged from C1.

## Resulting prices at the reference 2500 steps (cuda)

| Row | N+E | Old predicted | New predicted | Real |
| --- | ---: | ---: | ---: | ---: |
| kitchen_sink_hybrid_net | 44 | ~134 | 0.41 | 0.21s |
| r8_nested_parent_child_backedges | 65 | ~197 | 0.42 | 0.19s |
| random_bipartite_60 | 150 | ~452 | 0.49 | 0.20s |
| r79_undirected_sbm_low_mix_4x25 | 742 | 2229.7 (gen+score) | 0.95 | 0.25s |
| rgg_500 | 3991 | ~11975 | 3.50 | 0.54s |
| r8_nested_scale_1k_budget | 3038 | 9115.5 | 32488 | ~16000s (extrapolated) |

The scale anchor is preserved with wide margin: the Barnes-Hut price keeps
n=1000 fCoSE above 100x the 300s row budget (real cost is itself >50x budget),
so `r8_nested_scale_1k_budget` continues to skip fCoSE and admit Arm-S, and
the `--expect-no-skip arm_s` parity gate still guards a bad refit loudly.

## Verification (2026-07-21, this branch)

- The four M2 regression rows recover to main@M2's corrected-scorer scores
  with all three fCoSE seeds admitted (winners: fcose_seed0_raw,
  fcose_seed1_raw, fcose_seed0, fcose_seed2); per-row numbers in the C2
  commit message and M2_ANALYSIS_FABLE.md addendum.
- `scripts/native_determinism_gate.py r8_nested_scale_1k_budget rgg_500
  residual_block --runs 3 --expect-no-skip arm_s --min-score 80` PASS.
- `scripts/native_determinism_gate.py <4 rows> --runs 3 --expect-no-skip
  fcose` PASS (x3 idle + x3 load determinism, no budget skips).

## Gaps

- Exact-regime cpu alpha is dominated by the er_500 500-step sample; cpu
  mid-size rows are over-priced ~30x relative to real (still admitted with
  headroom). The benchmark box runs cuda; a cpu-box pass can tighten this.
- tsnet/stress small-n over-pricing (~24x, admitted today) is NOT touched by
  C2 -- changing currently-admitting families would flip other rows' admission
  without re-verification. Deferred to the post-merge re-baseline.
- No BH-regime samples between N=513 and N=800; the fitted rate applies the
  N=1000 P90 down to N=513 (protective, and the corpus has no rows in that
  band).
