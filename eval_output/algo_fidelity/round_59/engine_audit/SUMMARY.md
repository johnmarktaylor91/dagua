# R59 Engine Audit

## Method

Re-ran all smoke harnesses found under `eval_output/algo_fidelity/round_41`
through `round_44`, plus the canonical non-`smoke_harness.py` scripts for
LinLog and tsNET. No smoke harnesses were present in rounds 42-44.

Environment:

- `PATH=/tmp/graphviz_instr/bin:$PATH`
- Current Dagua checkout
- Fidelity-mode pipeline calls using the R44 float64 default or explicit
  `fidelity_dtype=torch.float64` where the pipeline exposes it
- Procrustes comparisons normalized both tensors to `torch.float64`

Several legacy smoke scripts now fail when executed directly because they pass
float64 Dagua positions into older float32-only comparison helpers:
`classical_mds`, `davidson_harel`, `drl`, `gem`, and `lgl`. The
`stress_sgd` smoke script also installs namespace stubs that hide
`resolve_fidelity_dtype`. For the audit table below, the same graph cases and
reference calls were run through an import-based runner with a dtype-normalized
Procrustes helper.

## Pre-Port RMSD

| Engine | Rows | Mean RMSD | Median RMSD | Max RMSD | Status |
|---|---:|---:|---:|---:|---|
| classical_mds | 12 | 5.25782182583347e-09 | 6.31264626792690e-09 | 8.40599476748008e-09 | pass |
| dagua_native | 12 | 3.45768897738058e-17 | 1.60798517218623e-17 | 1.06147855651499e-16 | pass |
| davidson_harel | 12 | 7.16316529180147e-09 | 7.33830518088618e-09 | 8.29634273049974e-09 | pass |
| drl | 12 | 4.35840931539688e-08 | 4.16828129326979e-08 | 9.03625408162094e-08 | pass |
| fa2 | 24 | 7.51178924556433e-05 | 1.08245328224636e-16 | 1.70925140163862e-03 | above threshold, off-limits |
| gem | 12 | 1.62979716642014e-08 | 1.58006787680925e-08 | 2.23336821819218e-08 | pass, off-limits file |
| graphopt | 12 | 9.77659212136129e-17 | 9.33390769655577e-17 | 1.57820752540158e-16 | pass |
| lgl | 12 | 1.71220750561086e-01 | 1.50449694144629e-01 | 3.62374615853917e-01 | above threshold, off-limits |
| linlog | 12 | 2.78741456549612e-17 | 7.58892011468595e-18 | 9.73920170039985e-17 | pass |
| maxent_stress | 12 | 7.95113996008039e-09 | 7.52946183039887e-09 | 1.01545432488034e-08 | pass |
| pivot_mds | 12 | 4.86085123169849e-09 | 5.65916105200222e-09 | 8.12508282278953e-09 | pass |
| reingold_tilford | 12 | 5.84341160570251e-17 | 4.95842141341075e-17 | 1.34568035959885e-16 | pass |
| stress_sgd | 12 | 7.42057426543712e-09 | 7.44628683318495e-09 | 1.06772695000936e-08 | pass |
| tsnet | 12 | 2.78116667679183e-17 | 2.84256380672226e-17 | 7.32959188605953e-17 | pass |

## Chase List

Engines with `max RMSD > 1e-6`:

| Engine | Max RMSD | Highest case | Action |
|---|---:|---|---|
| fa2 | 1.70925140163862e-03 | `star`, seed `0`, Barnes-Hut `True` | Not changed; `dagua/layout/ops/pipelines/fa2.py` is explicitly off-limits for this task. |
| lgl | 3.62374615853917e-01 | `star`, seed `43` | Not changed; `dagua/layout/ops/pipelines/lgl.py` is explicitly off-limits for this task. |

No in-scope engine exceeded `1e-6`, so no divergent step needed to be ported in
this pass.

## Ports Applied

None.

## Post-Port RMSD

No in-scope code changes were made. The final table is therefore identical to
the pre-port table.

| Engine | Rows | Mean RMSD | Median RMSD | Max RMSD | Status |
|---|---:|---:|---:|---:|---|
| classical_mds | 12 | 5.25782182583347e-09 | 6.31264626792690e-09 | 8.40599476748008e-09 | pass |
| dagua_native | 12 | 3.45768897738058e-17 | 1.60798517218623e-17 | 1.06147855651499e-16 | pass |
| davidson_harel | 12 | 7.16316529180147e-09 | 7.33830518088618e-09 | 8.29634273049974e-09 | pass |
| drl | 12 | 4.35840931539688e-08 | 4.16828129326979e-08 | 9.03625408162094e-08 | pass |
| fa2 | 24 | 7.51178924556433e-05 | 1.08245328224636e-16 | 1.70925140163862e-03 | above threshold, off-limits |
| gem | 12 | 1.62979716642014e-08 | 1.58006787680925e-08 | 2.23336821819218e-08 | pass, off-limits file |
| graphopt | 12 | 9.77659212136129e-17 | 9.33390769655577e-17 | 1.57820752540158e-16 | pass |
| lgl | 12 | 1.71220750561086e-01 | 1.50449694144629e-01 | 3.62374615853917e-01 | above threshold, off-limits |
| linlog | 12 | 2.78741456549612e-17 | 7.58892011468595e-18 | 9.73920170039985e-17 | pass |
| maxent_stress | 12 | 7.95113996008039e-09 | 7.52946183039887e-09 | 1.01545432488034e-08 | pass |
| pivot_mds | 12 | 4.86085123169849e-09 | 5.65916105200222e-09 | 8.12508282278953e-09 | pass |
| reingold_tilford | 12 | 5.84341160570251e-17 | 4.95842141341075e-17 | 1.34568035959885e-16 | pass |
| stress_sgd | 12 | 7.42057426543712e-09 | 7.44628683318495e-09 | 1.06772695000936e-08 | pass |
| tsnet | 12 | 2.78116667679183e-17 | 2.84256380672226e-17 | 7.32959188605953e-17 | pass |

## Notes

- Direct canonical harness execution succeeded for `fa2`, `graphopt`,
  `linlog`, `maxent_stress`, `pivot_mds`, `reingold_tilford`, and `tsnet`.
- Direct canonical harness execution failed for several older scripts only
  because their comparison helpers did not cast references to float64.
- `dagua_native` direct harness execution timed out at 240 seconds due to its
  extra component diagnosis passes. The audit runner used the same smoke cases
  and compared the direct pipeline to the adapter reference without the
  additional diagnosis toggles.
- The only remaining `>1e-6` engines are explicitly assigned to parallel work
  in the task scope.
