# Iterating Dagua's Native Algorithm Against the Originals Quality Baseline

**Purpose.** This is a clean-pickup handoff for improving **dagua's own default/native layout
algorithm** by comparing its layout *quality* against the quality of every **original external
reference algorithm** (Graphviz, igraph, OGDF, dagre.js, elkjs, d3, etc.) on the shared corpus.
This is a QUALITY goal (make dagua's native layouts as good as / better than the best references),
distinct from the fidelity campaign (making dagua's *reimplementations* match their references
bit-for-bit).

## The baseline (what you iterate against)

A one-seed quality pass over ALL original algorithms was run to give you references fast (rather
than waiting weeks for the full closeout benchmark).

- **Positions:** `eval_output/originals_1seed_quality/` (results.json + positions/*.pt), one seed
  (seed 42) per original engine across the full corpus.
- **Quality metrics + leaderboard:** `eval_output/originals_1seed_quality_report/report.md`
  (+ the per-graph/per-engine data the report is built from in that dir).
- **Metric:** the r83 honest GD-2025 ruler (frozen, Goodhart-resistant) computed by
  `scripts/quality_runtime_analysis.py`. See memory `project_r83_honest_ruler.md`.
- **DONE sentinel:** `eval_output/originals_1seed_quality/.ORIGINALS_QUALITY_DONE` exists when the
  run + metrics finished.

Regenerate / extend the baseline:
```bash
# 1-seed quality run over originals only (resumable):
python scripts/run_benchmark.py --engines originals --seeds 1 --seed-start 42 \
    --workers 3 --timeout 600 --watchdog-timeout 3600 --resume \
    --output-dir eval_output/originals_1seed_quality
# quality metrics + report:
WORKERS=4 bash scripts/run_quality_runtime_pipeline.sh \
    eval_output/originals_1seed_quality eval_output/originals_1seed_quality_report
```
(The exact chained runner used: `~/.claude/research/dagua/megasprint_completeness/run_originals_quality.sh`.)

`--engines originals` selects only the external reference engines (tagged `original_for`), NOT dagua
reimplementations (`classic_*`, `*_reimpl`) and NOT the native algo. `--engines reimpl` / `all` are
the other filters.

## Dagua's native algorithm (what you improve)

- Engine name: **`dagua`** (the native default). Dispatched via
  `dagua/layout/engine.py` with `LayoutConfig(algorithm=...)`; the native pipeline is
  `dagua/layout/ops/pipelines/dagua_native.py`. Topology dispatch / graph classification:
  `dagua/layout/graph_classify.py`.
- History: memory `project_r83_honest_ruler.md` (native 38->104/108 best-or-tied on the honest
  ruler), `project_r81_native.md` (73->94/108 best-or-tied, merged), `project_r79_native_algo.md`
  (layered/quality/scale wins; residual: undirected-class route to stress core).

## How to iterate

1. Run dagua's native algo on the corpus and compute its quality with the SAME ruler:
   ```bash
   python scripts/run_benchmark.py --engines dagua --seeds 1 --seed-start 42 \
       --workers 3 --timeout 600 --resume --output-dir eval_output/native_iter
   WORKERS=4 bash scripts/run_quality_runtime_pipeline.sh \
       eval_output/native_iter eval_output/native_iter_report
   ```
2. Compare `native_iter_report/report.md` vs `originals_1seed_quality_report/report.md`
   **per graph**: where is dagua's native quality below the best original? Those graphs are the
   improvement targets. "best-or-tied on N/108" is the headline metric.
3. Change the native algo (dagua_native.py / graph_classify.py / native op params), re-run steps 1-2,
   keep changes that raise best-or-tied count without regressing others. One dimension per iteration;
   lock wins with a test.

## Sacred rules (do not violate)

- **GLaDOS is a SEALED final holdout.** NEVER run or tune the native algo against GLaDOS. Tune ONLY
  on the 108-graph corpus. Run GLaDOS ONCE on a frozen algo, never iteratively. See memory
  `feedback_glados_sacred_holdout.md`.
- **Positions must always be saved** (never `--no-positions`) so metrics are recomputable later.
- **Honest ruler only.** Don't Goodhart the metric; the r83 ruler is frozen precisely to resist that.
- Useful files go to durable locations (repo, `~/.claude/research/`, `~/tools/`), never `/tmp`
  (48h purge). Reference sources: `~/tools/dagua-refs/` (+ `REFERENCE_SOURCES.md` manifest).

## Concurrent state at handoff (2026-07-14)

- A full closeout benchmark (all algos x full corpus x quality + fidelity, seed-optimized battery) is
  running as a weeks-long background crawl: `eval_output/benchmark_closeout_fullseed/` (cron
  keepalive `~/.claude/scripts/dagua_benchmark_keepalive.sh`, mirrored to
  `/mnt/locker/jt3295/dagua_closeout_benchmark`). Do not disturb it; it runs at low workers.
- Both-sides benchmark pairing merged (every reimpl paired with its external original).
- Reference sources being consolidated to `~/tools/dagua-refs/` with a `REFERENCE_SOURCES.md` manifest.
- Sprint log / durable notes: `~/.claude/research/dagua/megasprint_completeness/STATE.md`.
