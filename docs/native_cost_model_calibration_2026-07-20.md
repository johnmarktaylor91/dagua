# Native Cost Model Calibration - 2026-07-20

## Scope

C1 freezes `dagua.layout.ops.pipelines.native_cost_model.FROZEN_COST_TABLE` for
the modeled wall-second ledger. The table is device-class keyed and uses the
joint design form `alpha * volume + beta`.

## Box

- Date: 2026-07-20
- Host: local development workstation for `/home/jtaylor/projects/dagua`
- Python: `/home/jtaylor/anaconda3/envs/py311/bin/python`
- Branch: `codex/r0-determinism` at base `d801ca98`
- Device policy: CPU constants are authoritative for the harvested directed
  rows; CUDA entries are conservative modeled mirrors because no representative
  CUDA directed telemetry was available in the bounded C1 harvest window.

## Sources

- `~/.claude/research/dagua/megasprint/m1_out/idle_run1..3`: M1 scale-row
  telemetry. These runs preserve the scale_1k fCoSE skip / Arm-S admit anchor.
- `eval_output/fidelity_definitive/per_combo_r79_*.jsonl`
- `eval_output/fidelity_definitive/per_combo_r78_merged.jsonl`
- `eval_output/r81_base/results.rows.jsonl`
- `eval_output/r81_regate/results.rows.jsonl`
- Directed row sizes were remapped through `dagua.eval.graphs.get_test_graphs`
  before summarizing telemetry.

No full benchmark or 121-corpus generous-budget run was performed for C1.

## Envelope

Measured values use P90 idle runtime and family envelopes:

- CPU-parallel / Python-heavy directed arms: `2.0x`
- Native stress-family arms: `1.8x`
- GPU mirrors: `1.0x` to `1.2x` where no direct CUDA samples existed
- W5 tiny-row constants: retained from the prior recalibration,
  `step=0.0437`, `referee=0.019`
- fCoSE scale-row anchor: retained as a conservative modeled prior so
  `r8_nested_scale_1k_budget` continues to price fCoSE out and admit Arm-S.

## Sample Counts

Directed telemetry after graph-size remap:

| Family | Samples | Idle P90 | Frozen CPU package |
| --- | ---: | ---: | ---: |
| `directed_sugiyama` | 1029 | 1.073s | 2.2 DWU |
| `fcose` directed rows | 16 | 0.793s | conservative scale prior retained |
| `stress` directed rows | 1563 | 9.418s | existing stress table retained; `directed_stress_blend` modeled at 12 DWU |
| `tsnet` directed rows | 65 | 4.717s | still priced through existing stress call path |

Broad fidelity telemetry used for non-directed families:

| Family | Samples | Idle P90 | Freeze note |
| --- | ---: | ---: | --- |
| `directed_pivot_mds` | 60 | 3.976s | modeled at 6 DWU package |
| `directed_recombinant` | 0 direct samples | n/a | conservative 2.5 DWU package, bounded by six candidates |
| `directed_ordering` | 0 direct samples | n/a | conservative 5 DWU package, above 2.5s wall cap |
| `directed_yifanhu` | 0 direct samples | n/a | conservative 4 DWU package |
| `directed_mrtree` | 0 direct samples | n/a | conservative 4 DWU package |

## Freeze Decisions

- Directed Sugiyama no longer uses the legacy 60-DWU raw prior. It routes
  through `directed_sugiyama`, fixing the budget-starvation residual where a
  300s row could admit only four Sugiyama arms.
- Recombinant no longer routes through generic `opaque`. It uses
  `directed_recombinant` and keeps the old prior only as metadata.
- PivotMDS, stress-blend, ordering, YifanHu, and MrTree candidate groups now
  debit the ledger once per package so directed admission is deterministic and
  visible.
- Generic `opaque` is frozen at 30 DWU for unknown arms. Known directed arms
  have explicit families and should not depend on the opaque fallback.

## Gaps

The C1 harvest did not produce direct CUDA samples or direct recombinant,
ordering, YifanHu, or MrTree wall samples. Those entries are conservative
modeled priors, documented here for replay review.
