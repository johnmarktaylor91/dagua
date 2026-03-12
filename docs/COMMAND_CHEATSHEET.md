# Dagua Command Cheat Sheet

The fast commands worth remembering while iterating.

## Benchmarks

Check current benchmark progress:

```bash
dagua benchmark-status --output-dir /home/jtaylor/projects/dagua/eval_output --suite standard
```

Watch progress live:

```bash
dagua benchmark-watch --output-dir /home/jtaylor/projects/dagua/eval_output --suite standard --follow --interval 15
```

List stored runs:

```bash
dagua benchmark-list --output-dir /home/jtaylor/projects/dagua/eval_output --suite standard
```

Show one graph's stored result:

```bash
dagua benchmark-show residual_block --output-dir /home/jtaylor/projects/dagua/eval_output --suite standard --competitor dagua
```

Freeze a benchmark baseline:

```bash
dagua benchmark-freeze placement-baseline --output-dir /home/jtaylor/projects/dagua/eval_output --suite standard
```

Compare two runs:

```bash
dagua benchmark-compare-runs RUN_A RUN_B --output-dir /home/jtaylor/projects/dagua/eval_output --suite standard --competitor dagua
```

Regenerate report artifacts:

```bash
dagua benchmark-report --output-dir /home/jtaylor/projects/dagua/eval_output --no-pdf
dagua benchmark-deltas --output-dir /home/jtaylor/projects/dagua/eval_output
```

Refresh the placement-facing artifacts in one shot:

```bash
dagua placement-sprint --output-dir /home/jtaylor/projects/dagua/eval_output
```

Refresh and freeze the current standard run as a named placement baseline:

```bash
dagua placement-sprint \
  --output-dir /home/jtaylor/projects/dagua/eval_output \
  --freeze-label placement-baseline
```

## Visual Audit

Build the full suite:

```bash
dagua visual-audit-build --output-dir /home/jtaylor/projects/dagua/eval_output/visual_audit
```

Fast partial rebuild for one graph:

```bash
dagua visual-audit-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_audit \
  --graphs residual_block \
  --panels ladder competitor_stepwise run_to_run_diff
```

Freeze a named visual baseline:

```bash
dagua visual-audit-freeze reference --output-dir /home/jtaylor/projects/dagua/eval_output/visual_audit
```

Rebuild against a frozen baseline:

```bash
dagua visual-audit-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_audit \
  --graphs residual_block \
  --panels ladder run_to_run_diff \
  --compare-to-baseline reference
```

Build the numbered collaboration folder:

```bash
dagua visual-session-build --output-dir /home/jtaylor/projects/dagua/eval_output/visual_review_session
```

Fast rebuild for the one graph currently under discussion:

```bash
dagua visual-session-build \
  --output-dir /home/jtaylor/projects/dagua/eval_output/visual_review_session \
  --graphs residual_block
```

## Docs

Rebuild glossary:

```bash
python /home/jtaylor/projects/dagua/scripts/build_glossary.py --output-dir /home/jtaylor/projects/dagua/docs/glossary
```

Rebuild showcase gallery:

```bash
python /home/jtaylor/projects/dagua/scripts/build_gallery.py --output-dir /home/jtaylor/projects/dagua/docs/gallery
```

Rebuild algorithm explainer visuals:

```bash
python /home/jtaylor/projects/dagua/scripts/build_how_dagua_works.py
```

## Showcase Exports

Render a poster from a benchmark graph:

```bash
dagua poster graph.yaml /tmp/residual.png \
  --benchmark-graph residual_block \
  --benchmark-suite standard \
  --output-dir /home/jtaylor/projects/dagua/eval_output
```

Render a tour from a benchmark graph:

```bash
dagua tour graph.yaml /tmp/residual-tour.mp4 \
  --benchmark-graph residual_block \
  --benchmark-suite standard \
  --output-dir /home/jtaylor/projects/dagua/eval_output
```

## Billion Benchmark

Launch:

```bash
cd /home/jtaylor/projects/dagua
python -u scripts/bench_large.py 1b --device cuda
```

Tail the log used in tmux/nohup workflows:

```bash
tail -f /tmp/dagua-bench-1b.log
```
