<task>
Implement 1 source-faithful fidelity fix in dagua (r74 sprint) on branch develop, main tree:
classical_mds disconnected-component handling. dagua reimplements igraph's classical MDS as a PyTorch
pipeline: dagua/layout/ops/pipelines/classical_mds.py + helpers in dagua/layout/ops/graph_utils.py.

READ FIRST for exact lines + the guard:
- /tmp/r74_O4_findings.md, /tmp/r74_CX4_findings.md (classical_mds root cause)
- /tmp/r74_O_adversarial_findings.md, /tmp/r74_CX_adversarial_findings.md (B2 verdict: KEEP, rung-3
  target, 20/20 combos are D>R so direction is clean / low regression risk)

FIX B2 -- classical_mds disconnected-component layout + packing (~14-20 divergent disconnected combos).
  Root cause (verify, cite lines): igraph's classical MDS DECOMPOSES the graph into connected
  components, runs classical MDS per component, then merges via igraph_layout_merge_dla (a stochastic
  DLA pack). dagua instead fills cross-component infinite distances with ONE global scalar
  (dagua/layout/ops/graph_utils.py ~319-352), producing a single collapsed blob -> large positional
  divergence. FIX: detect disconnected input -> run classical MDS PER connected component -> pack the
  per-component layouts with the EXISTING deterministic TileToRows packer helper (grep dagua/layout/ops
  for the OGDF tile-to-rows offsets helper, e.g. _ogdf_tile_to_rows_offsets in the gem module -- REUSE,
  do not reimplement). This targets rung-3 (statistically/quality equivalent): a deterministic pack will
  not bit-match igraph's STOCHASTIC merge_dla, and porting DLA + seeded-ref is OUT OF SCOPE -- do not
  attempt it. Source-faithful; NO runtime delegation (never import/call igraph at runtime).
  REGRESSION GUARD (MANDATORY): the CONNECTED-graph path must stay byte-identical to before (it is
  already rung-1 bit-exact -- do not disturb it). Apply the component path ONLY to disconnected graphs.
  Commit: "fix(classical_mds): per-component MDS + TileToRows packing for disconnected graphs".
</task>

<default_follow_through_policy>
Most reasonable low-risk interpretation; proceed without stopping EXCEPT correctness/irreversible
concerns or if the connected path cannot be kept byte-identical (then scope behind a disconnected branch).
</default_follow_through_policy>

<constraints>
- NO RUNTIME DELEGATION. Source-faithful. per_combo.json final_rung is a STRING ("4").
- ASCII only. Conventional commit. NO AI attribution anywhere (humans only -- hard rule).
- Sanity check: run classical_mds on a small disconnected graph (finite packed coords, no blob) AND
  confirm a CONNECTED graph's output is byte-identical to before your edit.
- Run ONLY targeted tests for this module (e.g. pytest -q -k "mds or classical" tests/test_layout/ or
  the specific mds test file) -- do NOT run the full layout suite (slow; a comprehensive re-benchmark is
  done separately). Do NOT edit .project-context, CLAUDE.md, AGENTS.md, or unrelated files.
</constraints>

<verification_loop>
Import sanity + disconnected-case run + connected-case byte-identical check + targeted tests. git add
classical_mds.py + graph_utils.py (only what changed) + commit. Report commit SHA, the connected-path
guard result, and what changed.
</verification_loop>
