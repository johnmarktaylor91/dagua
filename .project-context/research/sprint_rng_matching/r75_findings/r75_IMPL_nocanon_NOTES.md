# r75 No-Canonical Reference Implementation Notes

## What Changed

- Added `AlgorithmVariant.reference_expressible`, defaulting to `True`.
- Marked `classic_sfdp_theta04`, `classic_sfdp_theta08`, and `classic_sfdp_steps200` as `reference_expressible=False`.
- Persisted `no_canonical_reference` from `scripts/definitive_fidelity_analysis.py`.
- Routed scored no-canonical rows to a dedicated `NO_CANONICAL_REFERENCE` report tier, outside divergent/identical/headline accounting.
- Left pre-existing `INSUFFICIENT_DATA` rows as insufficient while still persisting the no-canonical flag.
- Added report sections for:
  - no canonical reference rows
  - quality-superior but distinct rows
- Added regression tests for no-canonical routing and quality-superior distinct reporting.

## Commit SHA

Final commit SHA is self-referential for a note committed with the implementation; record the SHA from `git log -1 --oneline` after commit. The final handoff includes the actual SHA.

## Evidence

Graphviz 7.0.5 source check:

```text
lib/sfdpgen/spring_electrical.c:
44: ctrl->bh = 0.6;
46: ctrl->maxiter = 500;
```

This matches the r75 finding that Graphviz SFDP ignores the benchmark-passed `theta` and `maxiter` graph attributes; the probe observed bit-identical reference positions across those settings (RMS approximately 4e-16).

## Focused r75 Dry-Run

Input:

```text
/home/jtaylor/projects/dagua/eval_output/fidelity_definitive/r75_truebaseline.jsonl
```

Counts:

```text
raw_rows 409
flagged_no_canonical 55
new_no_canonical_tier 47
new_no_canonical_by_variant:
  classic_sfdp_steps200: 19
  classic_sfdp_theta04: 14
  classic_sfdp_theta08: 14
old_final_rungs_for_moved:
  4: 32
  3: 10
  3Q: 5
old_divergent_rung4 237
new_divergent_rung4 205
non_nocanon_identity_tier_changes 0
quality_superior_distinct 0
```

The 55 flagged rows include 8 pre-existing `INSUFFICIENT_DATA` rows. Those keep their no-data status; the 47 scored rows move to `NO_CANONICAL_REFERENCE`.

## Rendered Report Excerpts

No-canonical section:

```markdown
NO CANONICAL REFERENCE (dagua extension parameters -- reference cannot express these settings; excluded from fidelity accounting)

Graphviz 7.0.5 SFDP initializes the Barnes-Hut value as the `spring_electrical.c` constant `bh = 0.6` and keeps `maxiter` internal; the r75 probe found the reference positions bit-identical across the ignored `theta` and `maxiter` settings (RMS approximately 4e-16). These Dagua variants are legitimate extension knobs, but fidelity to a non-expressible reference is not coherent.

| Variant | Count |
|---|---:|
| `classic_sfdp_steps200` | 19 |
| `classic_sfdp_theta04` | 14 |
| `classic_sfdp_theta08` | 14 |
| **Total** | 47 |
```

Quality-superior distinct section on the current r75 input:

```markdown
No rows.
```

The section renderer was also covered by a fixture row that remains rung 4 and renders:

```text
QUALITY-SUPERIOR BUT DISTINCT (dagua measurably better on every failing quality leg -- these layouts are DIFFERENT from the reference, not equivalent)
```

## Controls Result

Command:

```bash
python scripts/definitive_fidelity_report.py --controls \
  --controls-dir /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/controls \
  --output-dir /tmp/r75_nocanon_controls
```

Result:

```text
gate_5_quality_identical_laundering:
  passed: true
  scored: 40
  three_q_count: 0
  three_q_percent: 0.0
  leaked_combo_ids: []
```

The command exits nonzero because gate 3 and gate 6 remain pre-existing failures:

```text
gate_3_negative: passed=false, non_primary_percent=90.0
gate_6_reference_self_split_positive: passed=false, scored=0
```

## Test Results

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_quality_battery_correctness.py -q
14 passed, 3 warnings
```

Requested broad command:

```text
pytest tests/ -k "quality or battery or report" -x -q
```

Result: failed on unrelated pre-existing verbose-output smoke test:

```text
FAILED tests/test_smoke.py::TestVerboseOutput::test_direct_layout_verbose_reports_node_count
AssertionError: assert '3' in ''
```

The single test reproduces deterministically and is outside the approved fidelity/report scope.

## Assumptions And Choices

- Rows with no canonical reference but already `INSUFFICIENT_DATA` retain their no-data tier; this reconciles the persisted flag count (55) with the approved 47-row no-canonical accounting move.
- `quality_superior_distinct` is rendered as an informational overlay only. It does not change final rung or accounting.
- The full report path was too slow because report assembly entered expensive artifact processing. The dry-run evidence above uses report finalization and section-rendering code directly against the r75 JSONL input.

## Concerns

- The broad requested pytest selector currently includes an unrelated smoke test that fails because verbose layout emits no captured node-count output.
- Tracked `.pyc` files in this repository can carry stale registry objects; the report/analysis scripts include an explicit three-variant fallback so the no-canonical decision does not depend solely on runtime dataclass shape.
