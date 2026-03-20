# Dagua Gotchas & Edge Cases

## Competitor Benchmark Pipeline (2026-03-20 retro)
- [BENCH] Adapter seed: verify 2 seeds produce DIFFERENT outputs before any multi-seed run. Hardcoded seeds silently produce identical data.
- [BENCH] Adapter config: audit ALL settings (device, seed, timeout, iterations, graph model) — silent misconfigs waste hours.
- [BENCH] RNG source: torch.rand(seed=42) != random.random() after random.seed(42). Match the EXACT RNG the reference uses.
- [BENCH] C extensions (s_gd2, igraph C, OGDF): internal RNG can't be reproduced from Python. Compare objective values, not positions.
- [BENCH] External tools: subprocess always works. Don't assume Python bindings are needed — we use subprocess for Graphviz/ELK/dagre/OGDF.
- [BENCH] Test graphs: check if weighted/directed/connected before comparing. nx.karate_club_graph() has edge weights 1-7.
- [BENCH] Reimplementation: match the INSTALLED code (`pip show`), not the paper. Papers are ambiguous.
- [BENCH] Results: never claim fidelity without adversarial review. Show per-graph distributions, not just means.
- [BENCH] Stop short: if you can name a viable improvement (line-by-line translation, coarsening match, RNG fix), DO IT. Don't frame it as future work. Three rounds of "we could" → "then do it!" wasted hours.

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
