<task>
Implement 1 performance/numerical-equivalence fix in dagua (r74 sprint) on branch develop, main tree:
torch-vectorize the FMMM graphviz-fdp grid-cell repulsion so the ~9 fmmm_graphviz_fdp graphs that
currently TIME OUT become scorable. File: dagua/layout/ops/pipelines/fmmm.py.

READ FIRST for exact lines:
- /tmp/r74_O3_findings.md, /tmp/r74_CX3_findings.md (fmmm fdp perf)
- /tmp/r74_O_adversarial_findings.md, /tmp/r74_CX_adversarial_findings.md (B3 = KEEP, pure perf)

FIX B3 -- vectorize graphviz-fdp grid repulsion.
  dagua already PORTS graphviz's spatial grid faithfully (cell = 3*K, repulsion between a cell and its
  same+neighbor cells; matches graphviz tlayout.c useGrid) but executes it in PURE-PYTHON per-pair loops
  (dagua/layout/ops/pipelines/fmmm.py ~5189-5251, in the graphviz_fdp tlayout repulsion helper, e.g.
  _graphviz_fdp_apply_tlayout_repulsion_lists, ~600 iterations). Timeout = pure-Python overhead, NOT
  algorithmic. FIX: torch-vectorize the exact same grid-cell repulsion (gather cell + neighbor-cell node
  pairs, compute repulsion as batched tensor ops). DO NOT switch to Barnes-Hut/quadtree -- that is LESS
  faithful than graphviz's own grid; keep graphviz's grid semantics exactly.
  CRITICAL GUARD -- NUMERICAL EQUIVALENCE: this is a perf refactor, not an algorithm change. The
  vectorized result must match the current Python-loop result on a small graph (allow only tiny
  floating-point summation-order differences; verify Procrustes RMSD ~0 / coordinates match to ~1e-5 on
  a small connected graph). If you cannot keep it numerically equivalent, STOP and report. Report the
  speedup and the max coordinate delta vs the loop on a test graph.
  Goal: the ~9 timeout graphs (e.g. larger fdp graphs) complete within the benchmark wall-clock. They
  may still land at the FP floor tier -- that is fine; the goal is to make them SCORABLE, not bit-exact.
  Commit: "perf(fmmm): vectorize graphviz-fdp grid repulsion to fix large-graph timeouts".
</task>

<default_follow_through_policy>
Most reasonable low-risk interpretation; proceed without stopping EXCEPT if numerical equivalence vs the
existing loop cannot be preserved, or any correctness/irreversible concern.
</default_follow_through_policy>

<constraints>
- NO RUNTIME DELEGATION. Keep graphviz grid semantics exact. per_combo.json final_rung is a STRING.
- ASCII only. Conventional commit. NO AI attribution anywhere (humans only -- hard rule).
- Sanity: vectorized output ~= loop output on a small graph (report max delta); time a medium graph to
  confirm speedup. Run ONLY targeted fmmm tests (pytest -q -k fmmm tests/test_layout/ or the fmmm test
  file) -- not the full suite. Do NOT edit .project-context, CLAUDE.md, AGENTS.md, unrelated files.
</constraints>

<verification_loop>
Import sanity + numerical-equivalence check vs the pre-edit loop (keep a reference copy of the loop
output before refactor) + timing + targeted fmmm tests. git add fmmm.py + commit. Report commit SHA,
the speedup, the max coord delta, and confirmation graphviz grid semantics are unchanged.
</verification_loop>
