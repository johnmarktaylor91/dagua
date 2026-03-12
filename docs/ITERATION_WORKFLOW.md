# Iteration Workflow

This is the default Dagua iteration workflow going forward.

The point is to make the loop disciplined, repeatable, and emotionally sane.

## Principle

Separate:

- placement quality
- visual language
- scale robustness

Do not let a weak visual layer trick you into misreading a strong placement engine.

## Default Loop

### 1. Start From The Placement Artifacts

Open:

- `eval_output/report/placement_dashboard.md`
- `eval_output/report/placement_summary.md`
- `eval_output/report/benchmark_deltas.md`

Ask:

- where is Dagua losing?
- is it broad or local?
- is it placement, or just presentation?

### 2. Check Similarity Before Making Claims

Open:

- `eval_output/report/layout_similarity.md`

Ask:

- is Dagua actually finding a different geometry?
- or is it close to a competitor and only visually worse?

### 3. Choose One Target

Choose:

- one graph
- one metric
- one failure mode

Do not “improve everything at once.”

Use:

- `docs/MONEY_GRAPHS.md`
- `docs/OPEN_ISSUES.md`

### 4. Make The Change

Keep the change narrow enough that you can explain:

- what you changed
- what you expected to improve
- what could regress

### 5. Rebuild Only What You Need

For placement iteration:

```bash
dagua placement-sprint --output-dir /home/jtaylor/projects/dagua/eval_output
```

For visual iteration:

```bash
dagua visual-audit-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_audit \
  --graphs residual_block \
  --panels ladder competitor_stepwise run_to_run_diff
```

### 6. Compare Against Baselines

Use:

- frozen benchmark baseline
- frozen visual-audit baseline

If no baseline exists yet, create one before making broad claims.

### 7. Log The Result

Update:

- `docs/OPEN_ISSUES.md`

and, when helpful:

- benchmark freeze labels
- visual-audit frozen baselines

## Placement-Specific Workflow

When the question is “is the node placement good?”:

1. ignore theme quality
2. ignore typography unless it affects sizing
3. read placement artifacts first
4. inspect competitor stepwise outputs only after metrics

The placement engine is the thing to protect.

## Visual-Reset Workflow

When the question is “does this look good?”:

1. start from simple graphs
2. use decomposition and kill-switch views
3. compare against frozen baseline
4. expand to harder graphs only after simple ones feel intentional

Read:

- `docs/VISUAL_RESET_BRIEF.md`

## Scale Workflow

When the question is “does it survive at giant scale?”:

1. watch the run
2. record where it dies
3. patch the narrowest real blocker
4. relaunch

Use:

- `scripts/bench_large.py`
- `/tmp/dagua-bench-1b.log`
- the rare benchmark suite

## The Rule

Every serious iteration should leave behind:

- a clearer artifact
- a clearer baseline
- or a clearer diagnosis

If it leaves none of those, the loop is too fuzzy.
