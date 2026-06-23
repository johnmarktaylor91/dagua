<task>
You are implementing 3 small, INDEPENDENT, source-faithful fidelity fixes in the dagua graph-layout
engine (r74 sprint), on the current branch (develop), in the main working tree. Each fix is in a
DIFFERENT file; implement and COMMIT each separately. dagua reimplements classic layout algorithms as
PyTorch op-pipelines under dagua/layout/ops/pipelines/ and helpers under dagua/layout/ops/.

Context you MUST read first for exact file:line and rationale:
- /tmp/r74_O5_findings.md and /tmp/r74_CX5_findings.md (Bucket-C / crashes)
- /tmp/r74_O4_findings.md and /tmp/r74_CX4_findings.md (maxent disconnected)
- /tmp/r74_O_adversarial_findings.md and /tmp/r74_CX_adversarial_findings.md (regression guards)

FIX 1 -- umap n_neighbors clamp (recovers ~9-10 INSUFFICIENT crash combos).
  Symptom: classic_umap_nn30 dies with BrokenProcessPool on small graphs (weighted_chain_20,
  regular_3_30) because n_neighbors(30) >= N. Fix: clamp effective n_neighbors = min(requested, N-1)
  in the dagua umap pipeline (dagua/layout/ops/pipelines/umap.py -- grep for n_neighbors). Verify the
  reference adapter (grep the eval/competitor umap adapter) ALSO clamps so both sides match; if the
  reference path needs the same clamp to not crash, apply it there too (this is harness/adapter code,
  NOT runtime delegation). Sanity-check: the pipeline runs without crashing on a 20-node graph with
  nn=30. Commit: "fix(umap): clamp n_neighbors to N-1 to fix nn30 crash on small graphs".

FIX 2 -- sugiyama iterative cycle-break (fixes 1 recursion-depth crash on big graphs).
  Symptom: classic_sugiyama_graphviz_fidelity on small_world_2000 hits Python max recursion depth in
  the cycle-removal / feedback-arc-set step. File: dagua/layout/ops/sugiyama.py (around the DAG-prep /
  cycle-removal, ~line 2150; grep for the recursive DFS used in cycle breaking). Convert the recursive
  DFS to an explicit-stack iterative version, preserving identical edge-reversal/feedback-arc behavior
  (must produce the SAME result on small graphs -- do not change the algorithm, only recursion->stack).
  Sanity-check: it runs on a 2000-node graph without RecursionError and is unchanged on a tiny graph.
  Commit: "fix(sugiyama): iterative cycle-break to avoid recursion overflow on large graphs".

FIX 3 -- maxent_stress disconnected-component handling (3 divergent disconnected combos).
  Symptom: classic_maxent_stress diverges on disconnected random_dag_50 because dagua runs ONE global
  stress majorization with a single global inf-fill (dagua/layout/ops/.../stress_majorization.py
  ~589-612), while OGDF StressMinimization routes disconnected graphs through ComponentSplitterLayout
  -> per-component layout -> TileToRowsCCPacker. Implement: detect disconnected input, lay out each
  connected component independently, then pack with the EXISTING TileToRows packer helper (grep
  dagua/layout/ops for the OGDF tile-to-rows offsets helper, e.g. _ogdf_tile_to_rows_offsets in the
  gem module -- reuse it, do not reimplement). Source-faithful to OGDF ComponentSplitterLayout +
  TileToRowsCCPacker. NO runtime delegation (do not import/call OGDF at runtime).
  REGRESSION GUARD (from adversarial review): 2 of the 3 target combos currently have dagua stress
  BETTER than reference (D<R). Do NOT regress connected graphs (the connected path must be byte-identical
  to before). Apply the component path ONLY to disconnected graphs. Commit:
  "fix(maxent_stress): per-component layout + TileToRows packing for disconnected graphs".
</task>

<default_follow_through_policy>
Take the most reasonable low-risk interpretation and proceed without stopping, EXCEPT for correctness/
safety/irreversible concerns. These are 3 independent fixes -- if one is blocked, still do the others.
</default_follow_through_policy>

<constraints>
- GUARDRAILS: NO RUNTIME DELEGATION (reimpl must never import/call igraph/ogdf/graphviz at runtime --
  reference adapters in eval/ are the only exception and only for the reference side). Source-faithful
  ports only. Do NOT weaken any existing test or fidelity gate.
- Note: in eval_output/.../per_combo.json, final_rung is a STRING ("4"), not an int -- never compare ==4.
- ASCII only in code/comments/commits. Conventional-commit messages. NO AI attribution anywhere (no
  Co-Authored-By, no "Generated with", nothing) -- humans only. This is a hard project rule.
- Run a quick targeted sanity check per fix (import + run the pipeline on the named small graph). Do NOT
  run the full benchmark -- that is done separately. Do NOT edit any .project-context/*.md, CLAUDE.md,
  or AGENTS.md. Do NOT touch unrelated files.
- Keep edits minimal and localized to the 3 named files (+ the umap reference adapter for FIX 1, + the
  shared packer is READ-ONLY/reused for FIX 3).
</constraints>

<verification_loop>
After each fix: python -c import sanity + run the specific pipeline on the named graph; ensure no crash
and (for FIX 2) identical output on a tiny graph. Then git add the specific files + git commit. Leave
the tree clean between fixes. Report each commit SHA + a one-line summary of what changed at the end.
</verification_loop>
