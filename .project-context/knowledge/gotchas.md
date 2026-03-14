# Dagua Gotchas & Edge Cases

- [LAYOUT] Crossing loss is O(E²) — needs interval amortization for large graphs. Performance-sensitive.
- [LAYOUT] Seed doesn't affect layout — init is fully deterministic from topology. Random perturbation needed for exploration.
- [LAYOUT] LR/RL direction: node_sizes not swapped before layout computation.
- [LAYOUT] Back-edge routing creates wide arcs that can overlap with other nodes.
- [LAYOUT] Multilevel/coarsening path is the most fragile layout code. Recent hardening with checkpoint validation, but needs smoke test coverage for each failure mode fixed.
- [RENDER] Multi-line node labels: secondary line font scaling is hardcoded (0.8x).
- [RENDER] Edge arrowheads: mutation_scale=1 makes heads very small at some zoom levels.
- [RENDER] Cluster label position is hardcoded (top-left) — should respect ClusterStyle.label_position.
- [EVAL] Non-Dagua competitor results are cached between benchmark rounds. If you change competitor adapter code, delete cached results.
- [EVAL] Long benchmark runs write `progress.json` alongside `results.partial.json` — check status with `dagua benchmark-status`.
- [EVAL] `scripts/bench_large.py` guards against duplicate concurrent runs unless explicitly forced.
- [MYPY] Only `dagua/cli.py` is under strict mypy. Other modules use relaxed settings. Don't assume the full codebase passes `--strict`.
- [DEPS] matplotlib, pyyaml, igraph, scipy, pydot are all optional. Always use lazy imports with helpful error messages.
- [GRAPH] DaguaGraph._id_to_idx must stay in sync with the nodes list. Adding/removing nodes requires updating the mapping.
- [STYLE] Thread-local storage for defaults — tests that use `dagua.configure()` should clean up or use `dagua.defaults()` context manager.
- [LAYOUT] `_offload_level_to_disk()` only offloads `edge_index` and `node_sizes`. `fine_to_coarse`, `fine_layer_assignments`, and `coarse_layer_assignments` stay in RAM for all hierarchy levels (~22 GB at 1B scale). Not currently a problem (72 GB peak vs 120 GB usable) but would need disk offload for 2B+ graphs.
- [LAYOUT] Original graph offload in `multilevel_layout()` (line ~910) does `del cpu_ei, cpu_ns` but the DaguaGraph object still holds `_edge_index_tensor` and `node_sizes`, wasting 16 GB at 1B scale. Fix: null out graph references during offload.
- [BENCH] 1B benchmark peak RAM is ~72 GB on CPU. Fits 125 GB machine with 48 GB margin. See `.project-context/tasks/1b-oom-audit.report.md` for full analysis.
