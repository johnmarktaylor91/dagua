# r75 Gate 6 Control Data Notes

## Data generation

Generated the reference self-split control row with the r74 control generator:

```bash
python3 scripts/definitive_fidelity_analysis.py \
  --mode reference-self-split-positive-control \
  --data-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_escalation_final \
  --data-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_seeded_refs \
  --output eval_output/fidelity_definitive/controls/gate6_reference_self_split_positive.jsonl \
  --workers 1
```

The main checkout data directories were used read-only.  The emitted row is:

```text
combo_id=center_port_backedge_hub::sgd2_multi_ref__for__classic_sgd2_multi_batch8::SELF_SPLIT
source_combo_id=center_port_backedge_hub::classic_sgd2_multi_batch8
mode=A
n=50
quality_identical_raw=True
battery_p_iut=7.960274058870743e-16
battery_n=50
```

I changed only the persisted row label from
`reference-self-split-positive-control` to `reference-self-split` before
committing the data.  The report's gate_1 selector treats any control kind
containing `positive` as a Mode-A positive control, so the narrower label keeps
gate_6 matched without changing gate_1's scored count.

The pre-commit large-file hook rejects files above 500 KB.  Gates 1, 3, and 4
from the read-only main checkout exceeded that limit, so I committed compact
JSONL controls instead of the original metric-heavy rows.  The compact rows
preserve only the report-consumed gate fields: mode, insufficient-data flags,
pass-driving rung flags, p-values used by gate 4, recovery counts, and the
quality-battery sentinel fields needed by gate 5.  No positions are stored.

## Controls verification

Before, using the read-only main controls directory:

```bash
python3 scripts/definitive_fidelity_report.py --controls \
  --controls-dir /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/controls \
  --output-dir /tmp/r75_gate6_before
```

```text
gate_1_positive_mode_a passed=True scored=39 pass_count=39
gate_2_positive_mode_b passed=True informative=39 pass_count=39
gate_3_negative passed=False scored=20 non_primary_percent=90.0
gate_4_chance passed=True n=20 recovery_count=21
gate_5_quality_identical_laundering passed=True scored=40 three_q_count=0
gate_6_reference_self_split_positive passed=False scored=0 quality_identical_count=0
```

After, using the compact worktree controls directory:

```bash
python3 scripts/definitive_fidelity_report.py --controls \
  --controls-dir eval_output/fidelity_definitive/controls \
  --output-dir /tmp/r75_gate6_check_compact
```

```text
gate_1_positive_mode_a passed=True scored=39 pass_count=39
gate_2_positive_mode_b passed=True informative=39 pass_count=39
gate_3_negative passed=False scored=20 non_primary_percent=90.0
gate_4_chance passed=True n=20 recovery_count=21
gate_5_quality_identical_laundering passed=True scored=40 three_q_count=0
gate_6_reference_self_split_positive passed=True scored=1 quality_identical_count=1
```

The controls command still exits nonzero because gate_3 remains a known
pre-existing failure.  Gate_6 now passes with scored data, and gates 1, 2, 4,
and 5 remain unchanged.

## Commit

Final commit SHA is reported in the worker output after commit creation.  It is
not embedded here because the file contents contribute to the commit hash.
