<task>
You are Codex working on the dagua project. The repo root is
`/home/jtaylor/projects/dagua`. The git branch is `develop` (DO NOT create a new branch -- the user has explicitly said one working branch this sprint).

This is Round 1 of the algo_fidelity sprint. The sprint goal is to perfect
dagua's faithful replication of the algorithms it claims to reimplement,
with **graphviz tools first** (dot, neato, fdp, sfdp). Drop-in graphviz
replacement is central to the dagua pitch.

Round 1 scope = INFRASTRUCTURE + BASELINE. No algorithm edits yet.

## What exists now

Existing artifacts:
- `eval_output/benchmark_full/results.json` -- per-graph quality metrics
  for ~269 engines including `graphviz_dot`, `graphviz_neato`,
  `graphviz_fdp`, `graphviz_sfdp` and all `classic_*` dagua reimpls,
  on 25 graphs each.
- `eval_output/benchmark_full/positions/` -- per-run `.pt` tensor files
  with the actual node coordinates (used for Procrustes comparison).
- `eval_output/fidelity_report/` -- existing fidelity analysis, but its
  comparison pairings come from `dagua/eval/variants.py` VARIANT_REGISTRY.
  The registry maps:
  - `sfdp_*` variants -> `graphviz_sfdp` (only graphviz pairing, marked
    is_true_original=False)
  - `sugiyama_*` variants -> `igraph_sugiyama` (NOT graphviz_dot)
  - `stress_maj_*` variants -> `ogdf_stress` (NOT graphviz_neato)
  - `fmmm_*` variants -> `ogdf_fmmm` (NOT graphviz_fdp)
- `scripts/fidelity_analysis.py` -- the existing fidelity analyzer (uses
  Procrustes via dagua.eval.pipeline_io.load_position_tensor).
- `scripts/compare_reimpl_vs_original.py` -- a different comparator with
  metric-level reporting.

The existing pipeline infrastructure is rock-solid for the registered
pairings. We need to ADD graphviz cross-comparisons without disturbing it.

## Round 1 deliverables

1. **Add a cross-comparator script: `scripts/algo_fidelity_cross.py`**
   - Reads `eval_output/benchmark_full/positions/*.pt` and
     `eval_output/benchmark_full/results.json` from disk (no
     re-benchmarking).
   - For each (dagua_engine, target_engine) pair from the table below,
     for each shared graph, computes:
     - Procrustes RMSD (use `scipy.spatial.procrustes` or replicate the
       routine used in `scripts/fidelity_analysis.py` so values are
       comparable to the existing report)
     - Per-graph quality metric deltas (aspect_ratio, dag_consistency,
       edge_length_cv, edge_straightness_mean_deg, depth_spearman_rho,
       overlap_count) from results.json
   - Writes:
     - `eval_output/algo_fidelity/round_1/data/pairwise_rmsd.csv`
       (columns: graph, dagua_engine, target_engine, n_nodes, rmsd,
       n_aligned, error)
     - `eval_output/algo_fidelity/round_1/data/quality_deltas.csv`
       (columns: graph, dagua_engine, target_engine, metric,
       dagua_value, target_value, abs_delta, rel_delta)
     - `eval_output/algo_fidelity/round_1/data/per_family_summary.json`
       (median/p25/p75/p95/worst RMSD per family pairing; count of
       graphs with rmsd > 0.15)

   **Pairings table** (priority order):
   | dagua_engine               | target_engine  | family_label    | priority |
   |----------------------------|----------------|-----------------|----------|
   | classic_sugiyama           | graphviz_dot   | dot             | P0       |
   | classic_stress_maj         | graphviz_neato | neato_stress    | P0       |
   | classic_classical_mds      | graphviz_neato | neato_mds       | P0       |
   | classic_fmmm               | graphviz_fdp   | fdp             | P0       |
   | classic_sfdp               | graphviz_sfdp  | sfdp            | P0       |
   | classic_fr                 | graphviz_neato | neato_fr_proxy  | P1       |
   | classic_kk                 | graphviz_neato | neato_kk_proxy  | P1       |

   The script must also be reusable for future rounds via a CLI flag:
   `--input-dir <path>` and `--output-dir <path>`. Round N can rerun
   it after dagua pipeline edits to compare new positions vs same
   graphviz baseline.

2. **Add a side-by-side panel script: `scripts/algo_fidelity_panel.py`**
   - For one (dagua_engine, target_engine) pair on one graph, render
     two scatter plots side-by-side (matplotlib): nodes as dots, edges
     as lines, both Procrustes-aligned to the target frame for visual
     comparison. Output: PNG.
   - CLI: `python scripts/algo_fidelity_panel.py <graph> <dagua_engine> <target_engine> [--output PATH]`
   - This is a layout-only visualization. **DO NOT** import from
     `dagua/render/` (cosmetic territory). Use raw matplotlib.

3. **Generate Round 1 baseline report:
   `.project-context/research/sprint_algo_fidelity/ROUND_1_BASELINE.md`**
   - For each P0 pairing:
     - Median, p25, p75, p95, worst RMSD
     - Count of graphs at RMSD > 0.05, > 0.15
     - Top-3 worst graphs by RMSD with the graph names
     - Top-3 worst quality-metric deltas (which metric, which graph)
   - For each P1 pairing: median + worst only
   - One-paragraph executive summary: which family is most divergent,
     which is closest, and a recommendation for which to attack first
     in Round 2.

4. **Generate one panel PNG per P0 family showing the worst-case graph**:
   - 4 PNGs in `eval_output/algo_fidelity/round_1/panels/`
   - Filename: `<family>__<graph>__worst.png`

5. **Per-round summary**: write
   `eval_output/algo_fidelity/round_1/SUMMARY.md`
   with the same content as ROUND_1_BASELINE.md but in a stable
   summary file the wake-up protocol can read.

6. **Append to iteration log** in
   `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
   one row for round 1 with: round=1, family=baseline, start=<ts>,
   end=<ts>, commit=<hash>, median RMSD before/after = "N/A baseline",
   worst graph=<top of list>, notes="Round 1 = infrastructure".
   Also update `state: ROUND_1_DONE` and `current_round: 1` at top.

## Verification loop

Before committing, run:

```
cd /home/jtaylor/projects/dagua
# Tier 1: tests for any new code
pytest tests/ -x --tb=short -q -k "fidelity or algo_fidelity" 2>&1 | tail -30
# Smoke: comparator runs end-to-end
python scripts/algo_fidelity_cross.py --input-dir eval_output/benchmark_full --output-dir eval_output/algo_fidelity/round_1
test -s eval_output/algo_fidelity/round_1/data/pairwise_rmsd.csv
test -s eval_output/algo_fidelity/round_1/data/per_family_summary.json
# Smoke: panel runs end-to-end on one P0 family worst case
python scripts/algo_fidelity_panel.py <pick worst graph> classic_sugiyama graphviz_dot \
    --output eval_output/algo_fidelity/round_1/panels/dot__test.png
test -s eval_output/algo_fidelity/round_1/panels/dot__test.png
```

If any verification fails, FIX it before committing. Don't commit a
broken Round 1 baseline -- subsequent rounds depend on it.

## Commit

ONE commit on `develop` with the message:

```
feat(fidelity): round 1 -- algo fidelity cross-comparator + graphviz baseline

- Add scripts/algo_fidelity_cross.py: dagua-vs-graphviz Procrustes RMSD + quality deltas
- Add scripts/algo_fidelity_panel.py: side-by-side comparison panels (raw matplotlib)
- Generate Round 1 baseline at eval_output/algo_fidelity/round_1/ for graphviz_{dot,neato,fdp,sfdp} pairings
- Worst-family-first recommendation in ROUND_1_BASELINE.md
```

Stage ONLY the files you create/edit. Do NOT stage any of:
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`

If git status shows those as modified, leave them untouched (they belong
to the parallel cluster sprint).
</task>

<scope_constraints>
**HARD scope (do not touch under any circumstances):**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`

**Allowed:**
- `scripts/algo_fidelity_*.py` (new)
- `eval_output/algo_fidelity/**` (new)
- `.project-context/research/sprint_algo_fidelity/**` (new)
- `tests/test_algo_fidelity_*.py` (new, if you add unit tests for the comparator)

**Not in this round:**
- Editing any `dagua/layout/ops/**` pipelines (Round 2+ only)
- Editing `dagua/eval/variants.py` (this round measures, doesn't change registry)
- Editing `dagua/eval/competitors/**` (graphviz adapters are working)
</scope_constraints>

<default_follow_through_policy>
Default to the most reasonable low-risk interpretation and keep going.
Only stop for missing details that change correctness, safety, or
irreversible actions. The Procrustes routine, file paths, and CLI shape
are specified; the implementation details (column ordering, exact median
formula, etc.) are yours to choose -- pick the standard interpretation.
</default_follow_through_policy>

<completeness_contract>
The round is COMPLETE only when:
1. `scripts/algo_fidelity_cross.py` exists, runs end-to-end on
   `eval_output/benchmark_full`, and produces all 3 output files in
   `eval_output/algo_fidelity/round_1/data/`.
2. `scripts/algo_fidelity_panel.py` exists and produces a non-empty PNG
   for at least one (dagua_engine, target_engine, graph) triple.
3. `eval_output/algo_fidelity/round_1/panels/` has 4 PNGs (one per P0
   family, showing the worst-case graph).
4. `ROUND_1_BASELINE.md` and `eval_output/algo_fidelity/round_1/SUMMARY.md`
   are written with the content described above.
5. STATE.md iteration log has the Round 1 row appended and `state:
   ROUND_1_DONE` at top.
6. ONE commit on `develop` with the prefix `feat(fidelity): round 1 --`.
7. `git status` shows NO untracked/modified files outside the
   sprint_algo_fidelity scope (cluster sprint files left alone).

If any step fails and you can't fix it in <10 min, write
`.project-context/research/sprint_algo_fidelity/ROUND_1_BLOCKED.md`
explaining what blocked you and stop. Do NOT commit a partial baseline.
</completeness_contract>

<verification_loop>
After implementing:
1. `pytest -x --tb=short -q tests/` for any failing tests touching files
   you changed (likely none for new files).
2. `python scripts/algo_fidelity_cross.py --input-dir eval_output/benchmark_full --output-dir eval_output/algo_fidelity/round_1`
   must exit 0.
3. Inspect `eval_output/algo_fidelity/round_1/data/per_family_summary.json`:
   each P0 pairing must have a non-null `median_rmsd` field.
4. Confirm 4 panel PNGs exist and are >5KB each.
5. `git diff --stat HEAD~0` should show only files in the allowed scope.
</verification_loop>

<missing_context_gating>
If you find that:
- `eval_output/benchmark_full/positions/` is missing or unreadable
- The graph name format in results.json keys (`graphname::engine::seed`)
  doesn't match what's on disk
- Procrustes computation requires a routine you can't locate

Then ABORT before any commits, write
`.project-context/research/sprint_algo_fidelity/ROUND_1_BLOCKED.md`
with what you observed, and stop. Do not guess at the data shape.
</missing_context_gating>

<action_safety>
- One commit on `develop` only. No force-push, no branch creation, no
  rebases, no tags.
- Do not modify `dagua/eval/variants.py` (registry stays untouched in
  Round 1; we measure existing data with new pairings via the
  cross-comparator).
- Do not delete files in `eval_output/`. Add only.
- Do not run the full benchmark (`run_benchmark.py`) -- this round uses
  existing data only.
</action_safety>
