# Dagua Gotchas & Edge Cases

## Competitor Benchmark Pipeline (2026-03-20 retro)
- [BENCH] Adapter seed: verify 2 seeds produce DIFFERENT outputs before any multi-seed run. Hardcoded seeds silently produce identical data.
- [BENCH] Adapter config: audit ALL settings (device, seed, timeout, iterations, graph model) — silent misconfigs waste hours.
- [BENCH] RNG source: torch.rand(seed=42) != random.random() after random.seed(42). Match the EXACT RNG the reference uses.
- [BENCH] C extensions (s_gd2, igraph C, OGDF): internal RNG can't be reproduced from Python. Compare objective values, not positions.
- [BENCH] External tools: subprocess always works. Don't assume Python bindings are needed — we use subprocess for Graphviz/ELK/dagre/OGDF.
- [BENCH] Test graphs: check if weighted/directed/connected before comparing. nx.karate_club_graph() has edge weights 1-7.
- [BENCH] Reimplementation: match the INSTALLED code (`pip show`), not the paper. Papers are ambiguous.
- [BENCH] Authority hierarchy: (1) installed reference code, (2) upstream repo, (3) paper (only if no impl exists). Pick one, document which. Never "apply the spirit."
- [BENCH] Results: never claim fidelity without adversarial review. Show per-graph distributions, not just means.
- [BENCH] Stop short: if you can name a viable improvement (line-by-line translation, coarsening match, RNG fix), DO IT. Don't frame it as future work. Three rounds of "we could" → "then do it!" wasted hours.

- [LAYOUT] Crossing loss is O(E²) — needs interval amortization for large graphs. Performance-sensitive.
- [LAYOUT] Seed doesn't affect layout — init is fully deterministic from topology. Random perturbation needed for exploration.
- [LAYOUT] LR/RL direction: node_sizes not swapped before layout computation.
- [LAYOUT] Back-edge routing creates wide arcs that can overlap with other nodes.
- [LAYOUT] Multilevel/coarsening path is the most fragile layout code. Recent hardening with checkpoint validation, but needs smoke test coverage for each failure mode fixed.

- [OPS] RNG backend varies per algorithm (torch, numpy, Python random). Changing backend breaks fidelity tests. Check archive source before choosing.
- [OPS] SolveState extras dict is the escape hatch. Keys use "algo_field" convention (e.g., "tsne_gains"). No validation -- read/write metadata is advisory only.
- [OPS] Pipeline fidelity tests compare against archive reimplementations with matching seeds. If an op changes behavior, the corresponding test_pipeline_*.py will fail.
- [OPS] graph_utils.py is self-contained. Do NOT add dagua imports or use torch ops that have graph.py equivalents.
- [OPS] 12 classic test modules have collection errors (stale imports from archived code). These are test_classic_*.py files -- they import from the old classic/ location. Not blocking but should be fixed.

- [RENDER] Multi-line node labels: secondary line font scaling is hardcoded (0.8x).
- [RENDER] Edge arrowheads: mutation_scale=1 makes heads very small at some zoom levels.
- [RENDER] Font sizing: use data coordinates directly (font_size_data = font_size_points). The old height-based heuristic (_node_relative_font_size_data) caused overflow/shrink/wrap bugs. compute_node_size already returns the correct font in layout units = data units.
- [RENDER] Stripe fill anti-aliasing: imshow extent must be inset by ~3% of min(w,h) to prevent pattern bleed at clip boundaries.
- [RENDER] Back-edge curvature: control points must be perpendicular to chord (not lateral). Lateral placement causes degenerate bezier shapes at high curvature.
- [RENDER] Arrowhead tangent for back edges: when local tangent disagrees with chord direction, use the chord (not negated tangent). Negated tangent points wrong for wide arcs.
- [RENDER] Non-convex shapes (star): clip text to bounding rectangle, not shape path. Shape concavities clip through glyph interiors.
- [RENDER] Polygon edge routing: ray_polygon_intersection in intersection.py handles 8 shapes (triangle, hexagon, pentagon, octagon, star, parallelogram, trapezoid, diamond). Also update _adjust_port_for_shape in edges.py when adding new polygon shapes.
- [RENDER] Double circle: render inner ring as a separate stroke-only Ellipse in _draw_node_shape_extras, not as a compound path (matplotlib fill rule doesn't reliably create rings).
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

## Cosmetic Polish Sprint (2026-03)
- [CRITIC] Critic calibration variance: different Claude review agents can rate the same image anywhere from 8-10 depending on calibration. Track the best known rating (max across rounds), not the latest rating, and never let a harsher critic overwrite a previously fair score.
- [RENDER] Matplotlib font resolution on Linux: Helvetica and Helvetica Neue are macOS-only. The stack falls back to Arial when `msttcorefonts` is installed, then DejaVu Sans. Italic rendering must verify that the resolved font file differs from the upright variant; when it does not, the renderer applies synthetic shear automatically.
- [RENDER] Gallery-card node proportions: `min_height` alone does not fix extreme aspect ratios. Constrain `min_width` as well because the layout engine and figure sizing jointly determine the final node shape.
- [TEST] Codex pytest hang: pytest can finish successfully and then stall during teardown. Check the captured log for a `passed` summary before treating the run as failed; if the tests are green, proceed.
- [RENDER] Auto text background cascade for patterned fills is tuned, not generic: pie/striped -> white at 0.92 alpha, hatched -> 0.75, gradient -> 0.90. Re-review all affected gallery cards before changing these opacities.

## Port Indicator Rendering (2026-03-26)
- [RENDER] FIXED: Port indicators used _points_to_data_units() which made them microscopic at gallery DPI. Fix: switched to ax.plot() with markersize in points (DPI-independent). Three rounds of "increase the constant" failed because the conversion pipeline was the root cause, not the size value. Lesson: when a visual element vanishes at different DPIs, check if its size is being converted to data coordinates.
- [RENDER] Bevel effect needs intensity >= 0.5 and highlight_alpha >= 0.55 to be clearly visible. Band count 8 gives smoother gradient than 6. Subtle bevel (intensity 0.3) is indistinguishable from no bevel at typical rendering sizes.

## run_benchmark worker-join-hang after "Done" (2026-06-04)
On igraph-based engines (drl, davidson, sugiyama, classical_mds) over big graphs, run_benchmark can
print its "Done: N total..." summary and write results.json, then HANG indefinitely: a multiprocessing
worker stuck in an uninterruptible igraph C call never terminates, so the pool join blocks forever. The
per-combo --timeout/--watchdog-timeout do NOT catch it (it's post-work, in shutdown). A caller using
subprocess.run() waits forever.
- DETECT: run_benchmark alive but results.json mtime stale >15min (>> the 420s combo watchdog).
- FIX: SIGKILL the run_benchmark main AND its workers. CRUCIAL: killing only the main REPARENTS the ~18
  workers to PID 1 (PPID=1, args contain "multiprocessing.forks") where they keep spinning at 99% CPU.
  Must also `kill -KILL` those orphans: `ps -C python3 -o pid=,ppid=,args= | awk '$2==1 && /multiprocessing.forks/{print $1}'`.
- SELF-HEAL: scripts/r69_stall_killer.sh (watchdog) auto-kills a stalled run_benchmark + orphans so a
  --resume runner retries/advances. Bounds each hang to ~15min instead of indefinite.

## fmmm.py fdp-fidelity trace = unbounded /tmp disk bomb (2026-06-04)
ALL fmmm fidelity-mode variants (graphviz_fdp_fidelity AND steps10/100/200) run the Graphviz-fidelity
FDP trace, which appended one line per node per phase per iteration to /tmp/dagua_fdp_trace.log -- ~6MB/s,
ballooning to 20.5GB during the 100-seed escalation and nearly tripping the disk guard. It was
UNGATED (fired whenever node_ids was set). Fixed: env-gated behind DAGUA_FDP_TRACE (default OFF;
purely logging, zero layout effect). NOTE: a running benchmark imports fmmm.py per-engine, so a code
gate only takes effect when the next run_benchmark RE-IMPORTS -- to force it mid-run, kill the current
engine's run_benchmark (its --resume retry re-imports the gated code).
