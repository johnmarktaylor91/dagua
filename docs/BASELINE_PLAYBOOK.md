# Baseline Playbook

This is the short playbook for freezing reference baselines before a serious iteration sprint.

The goal is simple:

- protect a known-good placement baseline
- protect a known-good visual baseline
- make comparisons explicit instead of fuzzy

## 1. Freeze A Benchmark Baseline

Once the current standard benchmark run is complete:

```bash
dagua benchmark-freeze placement-baseline \
  --output-dir /home/jtaylor/projects/dagua/eval_output \
  --suite standard
```

Use a more descriptive label if needed, for example:

- `placement-baseline-pre-visual-reset`
- `placement-baseline-round-0`

## 2. Freeze A Visual-Audit Baseline

After building the visual-audit suite you want to compare against:

```bash
dagua visual-audit-freeze reference \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_audit
```

## 3. Regenerate The Iteration Artifacts

Before changing anything major:

```bash
dagua benchmark-report --output-dir /home/jtaylor/projects/dagua/eval_output --no-pdf
dagua benchmark-deltas --output-dir /home/jtaylor/projects/dagua/eval_output
```

Then consult:

- `eval_output/report/placement_dashboard.md`
- `eval_output/report/placement_summary.md`
- `eval_output/report/layout_similarity.md`
- `eval_output/report/benchmark_deltas.md`

## 4. During Visual Iteration

Rebuild only the panels and graphs you actually need:

```bash
dagua visual-audit-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_audit \
  --graphs residual_block long_range_residual_ladder \
  --panels ladder competitor_stepwise run_to_run_diff \
  --compare-to-baseline reference
```

## 5. During Placement Iteration

Prefer reading:

- `placement_dashboard.md`
- `placement_summary.md`

before looking at full visual outputs.

That keeps the judgment focused on node placement rather than on the currently weak visual layer.

## 6. Keep A Short Written Note

When you freeze a baseline, also update:

- `docs/OPEN_ISSUES.md`

with:

- what is strong
- what is weak
- what must not regress

This makes the baseline operational, not just archival.
