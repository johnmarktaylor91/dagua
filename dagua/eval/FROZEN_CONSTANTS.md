# Dagua V3 Frozen Constants

Phase A updates for the converged GG-3 compensation fix:

- C2' angle-weighted crossings: per-crossing cost is `[0.5, 1.5]`, with 90-degree crossings cheapest.
- C2 size multiplier remains the frozen log-size band `[0.5, 1.0]`.
- C6 crossing angle is diagnostic-only with effective weight `0.0`.
- C8 path continuity is diagnostic-only with effective weight `0.0`.
- C4 uses a smooth clearance band below `0.5` mean node diagonals and includes labels when label geometry is supplied.
- G2 compactness gives no credit below `2.0` mean node-diagonal cluster-member spacing.
- G6 severe-breach ramp is `1.0x` at score `0.60`, linearly increasing to `2.0x` at score `0.55`; it remains tiered pricing only and is removed from the `tier1_only` measurement instrument.
- Severe-G6 declared-weight breach uses the shared facets `G6_weighted_ksm` and `G6_local_weight_monotonicity`, severe floor `0.55`, and GG-3 pair-form floor drop `0.05`.
- Score-neutral row flag vocabulary includes `severe_g6_breach`.
- GG-3 BLOCK requires `tier1_only` drop `>= 5.0` and two-layout buyback `>= 1.0`, or an independent severe-G6-floor breach from `dagua.eval.ruler_v3.severe_g6_floor_breach`.
- Tiered headline is now the uncapped `tiered_linear` composite; `tiered == tiered_linear` for corpus and single-row reporting.
- The clustered softmin `softmin(tiered_linear, tier1_only + 9.3004; tau=1.0)` is retained only as the GG-3/E1 adjudication hold instrument and as the audit columns `tiered_capped` / `tiered_hold_instrument`.
- Condition-5 aggregate deltas and material-hold ineligibility callers must compute the aggregate leg on the capped hold-instrument view, not the linear headline.
- Re-frozen `A_f` envelope table remains documented for audit provenance, but the frozen pre-freeze cap consumed by hold adjudication is the resolved-family clustered entry `9.3004`; no bars moved.
