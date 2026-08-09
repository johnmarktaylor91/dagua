# W2-3 completion record

## Changes

- Added `dagua/layout/ops/sprawl_repair.py`, a registered `RadialWinsorize`
  postprocess op that mirrors FIELD's deterministic two-pass RMS-radius recipe.
- Added input-only admission from either runtime `c5_whitespace_ratio > 16` or
  full-radius / robust-90%-radius `> 1.25`.
- Added at most two `sprawl_repaired` contest candidates, chosen from the best
  proxy-ranked already-admitted outlier geometries, with an explicit W1-C
  family-quota seat and every raw source retained.
- Added terminal `sprawl_repaired` honest-referee comparison and repaired W5
  warm start. New degeneracy flags, severe-G6 regressions, declared-layered
  regressions, ties, and score failures all retain the exact incumbent.
- Added focused op, gate, byte-inertness, state-seam, and tie-to-strict contest
  tests. No `ScaleBandCalibrate` implementation was added and `field.py` was
  not modified.

## Assumptions

- "Robust-vs-full extent" means maximum radius divided by the radius containing
  90% of nodes, centered on the coordinate median. The `1.25` gate denotes a
  visible 25% tail beyond robust occupancy; it is structural and was not fit to
  a dev graph.
- The two-RMS cap is corpus-independent: the second-moment bound limits mass
  beyond that radius to at most one quarter before FIELD's second pass.
- The acceptance tie-to-strict smoke is the focused honest-referee candidate
  test (`10.0 -> 11.0`); the two named dev rows were confirmation targets, not
  threshold-fitting inputs.

## Test results

- Import guard: PASS; `dagua.__file__` resolves under this worktree.
- `ruff check . --fix`: PASS (`All checks passed!`).
- `mypy --follow-imports=silent dagua/cli.py`: PASS (`Success: no issues found
  in 1 source file`).
- Focused tests: PASS, `130 passed, 3 warnings in 252.21s`:
  `test_ops_sprawl_repair`, `test_native_finisher`,
  `test_native_contest_cascade`, `test_pipeline_dagua_native`, and
  `test_native_guardrails`.
- Targeted candidate smoke: PASS; the closed gate makes zero score calls and
  returns the identical tensor object, while an admitted repair converts the
  synthetic tied incumbent into the strict honest-referee winner.
- Dev confirmation: `ash85` remains tied at `+0.198`; `arc130` remains tied at
  `+0.332`. No conversion is claimed for these two rows.
- Protected sparse wins: PASS. `1138_bus +2.514`, `bcspwr07 +1.778`,
  `bcspwr08 +1.603`, and `bcspwr09 +1.876` all remain strict.

## Controversial choices

- The repair is available both in the bounded undirected contest and at the
  terminal W5 seam. The former gives it a family-quota referee seat; the latter
  consumes the already-plumbed C5 signal and covers future sprawl rows that
  bypass the bounded contest.
- Candidate sources are proxy-ranked only after passing the structural extent
  gate. Ranking by tail severity alone selected visibly poor basins in the
  targeted smoke; proxy ranking keeps the repair attached to viable geometry
  without consulting graph identity or field scores.

## Concerns

- The funded honest-half mechanism did not convert `ash85` or `arc130` under
  the current W1 cascade. Their exact tensors remain byte-identical. Any future
  claim that these rows flipped needs separate evidence; scale calibration was
  intentionally not used to manufacture that result.
- Full training-121 and Tier-2 suites were not run because this packet called
  for fast checks only.

## Knowledge

- FIELD's private helper is only a two-pass radial clamp; mirroring it avoids a
  dependency from composable ops into scale-strategy internals.
- W1-C family quotas are the reliable seam for ensuring a new geometry family
  reaches the honest referee even when its proxy rank is weak.

PACKET: OK
