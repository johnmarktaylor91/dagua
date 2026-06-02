<task>
Build the matched-seed bit-exact verification harness + small-graph fixtures + the single
STATUS source-of-truth, for dagua's RNG-matching sprint.

## Context
dagua (/home/jtaylor/projects/dagua) reimplements ~24 graph layout algorithms and wants its
`fidelity_mode` ports to be BIT-IDENTICAL to their reference engines on SMALL graphs at
MATCHED random seeds. Reference adapters live in `dagua/eval/competitors/` (graphviz, igraph,
ogdf, fa2, sklearn/tsne, networkx, umap). The graphviz adapter already passes `-Gseed=N
-Gstart=N` (see graphviz_competitor.py ~line 379), and adapters take a `seed=N` arg.

## Build `scripts/rng_match/bitexact_harness.py`
A reusable harness that, for a given list of engines (default: all), for each SMALL fixture,
for each seed in {1,2,3}:
1. Runs the REFERENCE adapter for that engine at seed=N -> reference positions [Nnodes,2].
2. Runs dagua's reimplementation in fidelity_mode at the SAME seed=N -> dagua positions.
   (Use the same dispatch the benchmark uses -- dagua/eval/competitors/classic_competitor.py
   _quick_classic / the pipeline's layout_*_pipeline with fidelity_mode set per
   dagua/eval/variants.py. Reuse existing pairing metadata: variants.py original_variant_name
   gives each reimpl's reference engine.)
3. Computes BOTH: exact `torch.allclose(a, b, atol=1e-9, rtol=0)` AND Procrustes RMSD
   (reuse the procrustes_rmsd in scripts/fast_fidelity_report.py -- 2D Procrustes:
   center, scale-normalize, optimal rotation, RMSD).
4. Records a row: engine | reference | graph | seed | rmsd | exact_match(bool) | n_nodes.

CRITICAL correctness (avoid prior bugs):
- MATCHED seeds: reference at seed=N AND dagua at seed=N (the SAME N). Never seed=None vs
  seed=42. For deterministic engines (no seed effect), run once and compare (note it).
- Per-(engine,graph,seed) RMSD -- never aggregate-only. Report per-row AND per-engine max.
- A reimpl whose reference is deterministic still compares fine (ref same every seed).
- Skip+log engines with no paired reference (don't silently drop -- record "no_reference").

## Small-graph fixtures `scripts/rng_match/small_fixtures.py`
~12-15 tiny graphs, each a dagua Graph (or edge list), 6-20 nodes, spanning structure:
path8, cycle6, star8, grid3x3, grid4x4, complete5, complete_bipartite_3x3, balanced_tree_2x3,
two_triangles_bridge, small_dag_10, small_random_12 (fixed construction, deterministic),
petersen_10, wheel7, ladder5. Keep them SMALL so chaotic FP cascade is negligible -- a clean
RNG+arithmetic match should reach <1e-7 here.

## STATUS output (the single source of truth)
After a run, write/update:
- `.project-context/research/sprint_rng_matching/STATUS.md` -- a per-ENGINE markdown table:
  engine | reference | best(max) RMSD over fixtures&seeds | worst fixture | verdict
  (BIT_EXACT if max<1e-7 / CLOSE if <1e-3 / DIVERGENT) | exact_match_count/total | timestamp.
  Sort worst-first. This is THE record JMT reads.
- `.project-context/research/sprint_rng_matching/status.json` -- machine-readable full per-row data.
CLI: `python scripts/rng_match/bitexact_harness.py [--engines a,b,c] [--seeds 1,2,3]`.
Running with a subset of engines should UPDATE (not clobber) those engines' rows in STATUS,
so per-engine port codexes can refresh just their engine.

## Also: a tiny per-engine helper
`scripts/rng_match/check_engine.py <engine>` -- runs the harness for one engine, prints the
per-fixture RMSD table + the max, so a port codex can quickly see if it hit <1e-7.
</task>

<constraints>
- Pure measurement. Do NOT modify any layout pipeline or variants.py in this task.
- Reuse existing code (procrustes_rmsd, the competitor adapters, variants.py pairings) -- don't
  reinvent. Read scripts/fast_fidelity_report.py and dagua/eval/competitors/classic_competitor.py first.
- Matched-seed is the whole point -- if you find the reference can't be seeded for some engine,
  record it explicitly (don't fake a comparison).
</constraints>

<verification>
Run it on TWO engines as a self-test and put results in your report:
- `classic_fa2_default` (known genuinely bit-exact) -> expect max RMSD < 1e-7 on the small fixtures.
- `classic_sfdp_default` (known divergent vs the benchmark) -> expect higher (this baselines
  the "before RNG-matching" state for graphviz).
Print both so we can confirm the harness discriminates correctly.
</verification>

<output>
- scripts/rng_match/bitexact_harness.py, small_fixtures.py, check_engine.py
- .project-context/research/sprint_rng_matching/STATUS.md + status.json (initial full run over
  all engines, so we have the BEFORE baseline for every engine)
- Report the fa2 vs sfdp self-test numbers.
Commit the scripts to branch develop (STATUS.md/status.json are records, commit them too).
</output>

<default_follow_through_policy>
Proceed autonomously with the most reasonable interpretation. Only stop for: a reference that
genuinely cannot be run at a matched seed (record it), or inability to reuse the existing
adapter dispatch (explain why).
</default_follow_through_policy>
