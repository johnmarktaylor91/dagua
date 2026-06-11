<task>
Implement Task B (the runner) of the r70 definitive fidelity analysis for the dagua repo
(you are in /home/jtaylor/projects/dagua).

AUTHORITATIVE SPEC -- read it FIRST, in full:
  .project-context/research/sprint_rng_matching/SPEC_definitive_fidelity_analysis.md
(version 6, APPROVED). Implement it EXACTLY. Where this prompt and the spec disagree, the
SPEC wins. Task A (stats core) is ALREADY COMMITTED at dagua/eval/distributional_fidelity.py
with tests at tests/test_distributional_fidelity.py -- read its public API and CALL it;
do NOT reimplement statistics, do NOT modify it.

CREATE EXACTLY ONE FILE (touch nothing else):
  scripts/definitive_fidelity_analysis.py

It implements spec sec. 10 Task B -- a CLI runner that:

1. INPUTS (read-only; spec sec. 1 table has schemas, all verified):
   - --data-dir (default eval_output/benchmark_100seed_escalation_final): results.json +
     positions/*.pt. Keys `graph::engine::seedN` or `graph::engine::deterministic`.
     Resolve references via the fallback chain of scripts/fast_fidelity_report.py
     `_resolve_pos` (read that function). Load positions ONLY via each row's
     positions_file field; trust statuses, not file existence.
   - .project-context/research/sprint_rng_matching/failing_map_final.json
     ({engine: {ref, graphs}}; 64 engines, 3,955 combos).
   - For --mode deterministic and rung0-reverify: eval_output/benchmark_5seed_deterministic_refresh/
     (same schema, 5 seeds) and dagua/eval/equivalence_metrics.py (toolkit distance per
     spec sec. 7: min over aut_procrustes_rmsd, component_aligned_rmsd, + anisotropic only
     for FREE_ASPECT engines; spectrum/quality branches EXCLUDED).
   - Graph structures for stress: dagua/eval/graphs.py get_test_graphs() (edges only).

2. PER-COMBO PIPELINE (spec secs. 2-6): mode classification (Mode A >=30 matched ok seeds
   both sides; Mode B single deterministic ref with >=30 ok reimpl seeds; INSUFFICIENT_DATA
   reasons matched_seeds_lt_30 / reimpl_seeds_lt_30 / ref_seeds_lt_30 / no_reference_rows);
   call analyze_mode_a / analyze_mode_b with free_aspect=True for classic_sugiyama* engines;
   stress per spec sec. 6 (prepare_graph_distances + sample_pairs + stress_per_layout from
   Task A; BFS distances cached per graph across the run; paired/one-sample TOST with
   margin floor); record EVERYTHING in one JSON row per combo (all stats, flags, n,
   n_dropped reasons, mode, runtime ratio from results.json runtime_seconds medians,
   plain-W guard values). Do NOT apply FDR and do NOT assign final rungs (the report stage
   does that globally) -- but include every raw p-value and calibration verdict the ladder
   needs.

3. EXECUTION: parent pre-indexes results.json ONCE into small per-combo payloads (key ->
   positions_file path, status, seed); ProcessPoolExecutor --workers default 12 with
   OMP_NUM_THREADS=1 etc. set; incremental append to per_combo.jsonl (one JSON line per
   combo, written by the parent as futures complete -- atomic enough); every row carries
   spec_version="r70-v6" + git sha (subprocess git rev-parse, once); --resume skips combos
   already present in the jsonl UNLESS their spec_version/sha mismatch (recompute those);
   --combos-file (text file "graph::engine" lines) restricts the combo set; progress.json
   heartbeat (done/total/ts) every 25 combos; psutil RSS guard (warn > 70% system RAM,
   abort worker > 85%).

4. MODES (--mode, spec secs. 5/7/8/10):
   - full: all combos from failing_map x graphs (or --combos-file subset).
   - negative-control: draw 20 mispairs per spec sec. 8.3 EXACTLY (different algorithm
     tokens from the pre-registered token set in spec sec. 10; different reference BASES
     (string before "__for__"); drawn-pair stored reference layouts differ by Procrustes
     d > 0.1, using the deterministic layout where single-draw else the seed-42 layout;
     reimpl cloud pre-screen mean plain W_D <= 1.0; seeded sha256("r70::negctl") over
     sorted candidate keys). Analyze each mispair as if it were a real combo; emit the
     token map to controls/token_map.json FIRST and print it (CC signs off).
   - chance-control: 20 real Mode A combos (seeded sha256("r70::chance") over sorted
     Mode-A-eligible combo keys), reference seed labels permuted ONCE (seeded per combo)
     before analysis.
   - modeb-positive-control: combos from --combos-file (the tier1-control combos), with
     the reference truncated to its seed-42 layout treated as deterministic -> Mode B path.
   - deterministic: the 8 DETERMINISTIC_DIFFERENT engines (classic_kk_steps100/300/1000,
     classic_rt_horizontal, classic_spectral_default/nx_fidelity/random_walk/unnormalized)
     vs their refs on all refresh-dir graphs: toolkit-distance verdict + stress delta per
     spec sec. 7.
   - rung0-reverify: sugiyama variants' NON-failing-map graphs in the refresh dir:
     recompute 5-seed max per-seed Procrustes RMSD vs ref; emit per-combo
     {still_bit_exact: bool, max_rmsd} (threshold 1e-3).

5. OUTPUT: --output (default eval_output/fidelity_definitive/per_combo.jsonl for full mode;
   controls modes default under eval_output/fidelity_definitive/controls/<mode>.jsonl).

SEEDING: per-combo rng per spec sec. 10 (sha256 of "{graph}::{engine}::r70"); global
selections by purpose strings. NEVER Python hash().
</task>

<completeness_contract>
Done means: the file exists; `python3 scripts/definitive_fidelity_analysis.py --help` works;
a SMOKE RUN succeeds: `python3 scripts/definitive_fidelity_analysis.py --mode full
--combos-file /tmp/r70_smoke_combos.txt --workers 4 --output /tmp/r70_smoke.jsonl` where
you create the combos file yourself with ~10 lines spanning BOTH modes (pick from
failing_map: e.g. 3x classic_fr_steps100 graphs, 3x classic_classical_mds_default graphs,
2x classic_gem_iters100, 2x classic_sugiyama_default) -- verify the output rows contain
mode, stats, flags consistent with the spec (fr: mode A, small diag; classical_mds: mode B
near_deterministic; gem: TYPICALITY_UNINFORMATIVE likely; sugiyama: free_aspect distances)
and PRINT a summary of those 10 rows at the end of your run. ruff check clean on the new
file. No other file modified. Do NOT git commit.
</completeness_contract>

<verification_loop>
Iterate the smoke run until rows look spec-conformant. If a Task A API gap blocks you,
write a thin adapter INSIDE the runner (do not modify Task A); note it as
"SPEC-INTERPRETATION:" comment.
</verification_loop>

<action_safety>
Never invoke layout engines or references -- read stored positions only. Read-only except
the one new file and /tmp scratch. numpy/scipy/torch/h5py/psutil are installed.
</action_safety>

<default_follow_through_policy>
Most reasonable low-risk interpretation; stop only for genuine correctness walls. Do not
expand into Task C (report/FDR/aggregation).
</default_follow_through_policy>
