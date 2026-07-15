<task>
Implement the highest-leverage but HIGHEST-REGRESSION-RISK sugiyama fidelity fix in dagua (r74 sprint)
on branch develop, main tree: make the igraph-targeting sugiyama layer-assignment LP use igraph's actual
objective + gating. File: dagua/layout/ops/sugiyama.py (and helpers it calls). Reference C source:
/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c.

READ FIRST -- exact lines + the MANDATORY guards (this fix is gated; honor every guard):
- /tmp/r74_O1_findings.md, /tmp/r74_CX1_findings.md (sugiyama root cause; note CX1 corrected omega to 1/2/4)
- /tmp/r74_O_adversarial_findings.md, /tmp/r74_CX_adversarial_findings.md (C1 verdict: REVISE/GATE --
  igraph variants ONLY, NOT graphviz_fidelity; ~150-183 currently-matching combos at risk)

FIX C1 -- igraph GLPK layer-assignment objective + gating (the ~231 mode-B divergent sugiyama family).
  Root cause (verify in source, cite lines): igraph's layer assignment minimizes
  sum_i (outdeg_i - indeg_i) * x_i via GLPK (igraph sugiyama.c ~611-615), and runs that LP ONLY for
  DIRECTED graphs with n <= 1000 (sugiyama.c ~564); for undirected or n>1000 it uses a different method
  (Eades/longest-path, ~661-665). dagua's `_igraph_glpk_layer_assignments` (dagua/layout/ops/sugiyama.py
  ~379) currently uses a ZERO objective ([0.0]*N) AND runs HiGHS UNCONDITIONALLY. FIX (igraph-faithful):
    (1) set the LP objective to sum_i (outdeg_i - indeg_i) * x_i (compute per-node out/in degree on the
        directed graph; mirror igraph's sign/scaling exactly -- read the C).
    (2) add igraph's gating: use the LP only for directed graphs with n <= 1000; otherwise use the same
        fallback igraph uses (longest-path/Eades layering -- match what sugiyama.c does, do not invent).
  SCOPE GUARD (MANDATORY): apply this ONLY to the IGRAPH-targeting sugiyama variants. The
  graphviz-fidelity sugiyama path (which targets graphviz dot / network-simplex x-coords) MUST remain
  BYTE-IDENTICAL -- do not touch its objective or gating. Determine how the pipeline distinguishes the
  igraph vs graphviz target (grep variants / config) and branch on it.
  REGRESSION RISK: ~150-183 sugiyama combos currently match (rung 1-3); changing the objective can move
  them. You cannot fully validate this without the benchmark (done separately by the orchestrator). Your
  job: implement igraph-faithfully, keep the graphviz path byte-identical, and sanity-check it runs and
  is deterministic on small directed + undirected graphs. NO runtime delegation (never call igraph).
  Commit: "fix(sugiyama): igraph-faithful GLPK layer objective + directed<=1000 gating (igraph variants)".
</task>

<default_follow_through_policy>
Most reasonable igraph-faithful interpretation; proceed without stopping EXCEPT: if you cannot keep the
graphviz-fidelity path byte-identical, STOP and report; also stop on any correctness/irreversible concern.
Read the igraph C to resolve the exact objective sign and fallback -- do not guess.
</default_follow_through_policy>

<constraints>
- NO RUNTIME DELEGATION. Source-faithful to igraph sugiyama.c. per_combo.json final_rung is a STRING.
- graphviz-fidelity sugiyama path BYTE-IDENTICAL (hard requirement).
- ASCII only. Conventional commit. NO AI attribution anywhere (humans only -- hard rule).
- Sanity: run sugiyama on a small DIRECTED graph (LP path) and a small UNDIRECTED graph (fallback path)
  and a graphviz-fidelity case (must be unchanged). Run ONLY targeted sugiyama tests -- not the full
  suite. Do NOT edit .project-context, CLAUDE.md, AGENTS.md, unrelated files.
</constraints>

<verification_loop>
Import sanity + small directed/undirected runs + confirm graphviz-fidelity output byte-identical to
before your edit + targeted sugiyama tests. git add sugiyama.py (+ helpers) + commit. Report commit SHA,
the exact igraph objective/gating you matched (with C line cites), and the graphviz-path guard result.
</verification_loop>
