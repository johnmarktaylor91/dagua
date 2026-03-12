# Iteration Workflow

This is the default Dagua iteration workflow going forward.

The point is to make the loop disciplined, repeatable, and emotionally sane.

## Principle

Separate:

- placement quality
- visual language
- scale robustness

And be explicit about which loop you are in before changing code.

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

For collaborative visual iteration with numbered graphs:

```bash
dagua visual-session-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_review_session
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

Current known placement conclusion:

- Dagua is already competitive on DAG consistency and overlap avoidance.
- The clearest placement gap is still edge crossings against the strongest hierarchical competitors.
- Crossing work should preserve the current wins on DAG order and overlap rather than trading them away casually.

## Visual-Reset Workflow

When the question is “does this look good?”:

1. start from simple graphs
2. use decomposition and kill-switch views
3. compare against frozen baseline
4. expand to harder graphs only after simple ones feel intentional

Read:

- `docs/VISUAL_RESET_BRIEF.md`

## Collaborative Visual Session Workflow

When we are iterating together on visuals, use the numbered session folder instead of browsing a large artifact tree.

1. Build the session:

```bash
dagua visual-session-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_review_session
```

2. Open the images in order:

- `01_...png`
- `02_...png`
- `03_...png`
- and so on from simple to complex

3. For each graph:

- compare Dagua with the strongest competitor on that exact graph
- name the first thing that is clearly worse
- decide whether the problem is:
  - text hierarchy
  - edge language
  - cluster geometry
  - information density
  - something else

4. Write the decision in:

- `eval_output/visual_review_session/SESSION_NOTES.md`

5. Rebuild only the graph currently under discussion:

```bash
dagua visual-session-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_review_session \
  --graphs residual_block
```

6. Stay on that graph until the direction is clearly better.

7. Move to the next numbered graph only after the current one is resolved.

## Scale Workflow

When the question is “does it survive at giant scale?”:

1. watch the run
2. record where it dies
3. patch the narrowest real blocker
4. relaunch

Do not mix scale debugging with visual judgment. Giant-run failures are almost
always infrastructure, memory, or tensor-shape issues first.

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
