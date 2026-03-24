NEW SESSION: Read this file first. Then read CLAUDE.md, AGENTS.md,
and .project-context/knowledge/. The knowledge files contain durable
project understanding. This baton file contains live session state.
After reading everything, your first action should be: check benchmark
progress with `tail -5 .project-context/tasks/full-variant-bench.log`
and `cat .project-context/tasks/full-variant-bench.status`.

PRIORITY: You own the algorithm layout benchmark workstream. Aesthetics/
themes/cosmetics are handled by another Claude instance -- do NOT touch
dagua/styles.py, dagua/render/, or themes.

## Goal

Complete the variant benchmark run, analyze results, and continue
expanding layout algorithm coverage.

## Completed This Session (2026-03-22)

### Pitstop fixes (applied to working tree, not yet committed)
- FA2 reference adapter: runtime introspection filters unsupported kwargs
  (dissuadeHubs removed from variant_param_names, _accepted_init_params()
  filters engine_kwargs at call time)
- SGD2 multi reference adapter: patched criteria.stress + ideal_edge_length
  (0-d tensor fix via torch.stack/advanced indexing, NetworkX EdgeView fix
  via list() wrapping), disabled vis_interval (broken V.plot call), added
  evaluate_interval so get_result_dict works, fixed .X -> .pos attribute
- Benchmark runner (scripts/run_benchmark.py):
  - Skip-after-3 counts ALL failures (errors + timeouts), not just timeouts
  - Graph-size-aware timeout (30s floor, scales linearly to 180s at 500+ nodes)
  - Graphs sorted small-to-large (fast results first)
  - Rolling submission window (replaced batch-drain pattern)
  - SIGINT handler with graceful shutdown + guaranteed save-on-exit
- Results cleanup: removed 3,995 stale "running" + 3,300 fixed-engine errors

### New competitors added
- Cytoscape fcose (Node.js subprocess, dagua/eval/competitors/cytoscape_fcose_competitor.py)
  - npm deps: cytoscape, cytoscape-fcose (installed in project root)
  - 2 variants: default, quality=proof
- Gephi YifanHu (Java subprocess, dagua/eval/competitors/gephi_competitor.py)
  - JAR: lib/gephi-toolkit-0.10.1-all.jar (80MB)
  - Java helper: dagua/eval/competitors/gephi_layout.java
  - Compiled cache: dagua/eval/competitors/_gephi_build/
  - 1 variant: default
- FR->KK warm-start chain (classic_competitor.py ClassicFrKk)
  - 2 variants: default (50+300), long (100+300)
- KK->FR warm-start chain (classic_competitor.py ClassicKkFr)
  - 2 variants: default (300+50), long (300+100)

### Edge weights added to all remaining algorithms
- davidson_harel, drl (renamed weights->edge_weights), fmmm, gem,
  sugiyama, umap, sfdp (exposed internal), neulay
- All 20/20 classic algorithms now accept edge_weights

### Variant registry
- 97 -> 104 variants (7 new for the 4 new competitors)
- 56 total available competitors

## In Progress

- **Full variant benchmark v2**: ~11.6% complete (50K/433K), launched ~21:47 EDT Mar 22
  - Status: `cat .project-context/tasks/full-variant-bench.status`
  - Progress: `tail -5 .project-context/tasks/full-variant-bench.log`
  - Output: `eval_output/variant_bench_full/`
  - Workers: 6 parallel (forkserver), heavy engines serial after
  - Skip-after-3-failures active per (engine, graph) combo
  - ETA: ~Monday evening Mar 24

## Immediate Next Steps

1. **Wait for benchmark to finish** -- Pushover notification will fire.
2. **Analyze results** when done:
   - Run `python scripts/compare_reimpl_vs_original.py --input eval_output/variant_bench_full`
   - Key metrics: Procrustes disparity per variant, timeout rates, error rates
   - Check new competitor performance (fcose, YifanHu, warm-start chains)
3. **Commit all changes** -- everything is in working tree uncommitted

## Key Files

| File | Purpose |
|------|---------|
| scripts/run_benchmark.py | Benchmark runner (with all pitstop fixes) |
| dagua/eval/variants.py | Variant registry (104 entries) |
| dagua/eval/competitors/classic_competitor.py | Classic adapter + ChainCompetitor classes |
| dagua/eval/competitors/fa2_competitor.py | Fixed FA2 reference adapter |
| dagua/eval/competitors/sgd2_multi_competitor.py | Fixed SGD2 multi reference adapter |
| dagua/eval/competitors/cytoscape_fcose_competitor.py | NEW: fcose adapter |
| dagua/eval/competitors/gephi_competitor.py | NEW: YifanHu adapter |
| dagua/eval/competitors/gephi_layout.java | NEW: Java helper for Gephi |
| dagua/layout/classic/*.py | All 20 with edge_weights now |
| eval_output/variant_bench_full/ | Benchmark output directory |

## TorchLens Interaction
- TorchLens decorates torch functions when get_test_graphs() traces neural nets
- unwrap_torch() doesn't fully restore torch.tensor behavior (bug reported, user says fixed)
- SGD2 adapter has defense-in-depth: patched criteria.stress and ideal_edge_length
  use torch.stack / advanced indexing instead of torch.tensor on 0-d tensor lists

## Git State
- Branch: feat/bench-and-aesthetics
- Uncommitted: many files (all pitstop + Codex changes)
- NOTE: aesthetics Claude also has uncommitted changes on this branch
