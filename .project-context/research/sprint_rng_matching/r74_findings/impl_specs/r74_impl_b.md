<task>
Implement 2 source-faithful sfdp fidelity fixes in dagua (r74 sprint) on branch develop, main tree.
Both touch the sfdp pipeline files -- do them SEQUENTIALLY, commit each. dagua reimplements graphviz
sfdp as a PyTorch pipeline: dagua/layout/ops/pipelines/sfdp.py + dagua/layout/ops/sfdp.py + variant
config (grep variants for sfdp). Reference C source: /home/jtaylor/projects/_references/graphviz/lib/sfdpgen/.

READ FIRST for exact line numbers + the regression guards (these are MANDATORY):
- /tmp/r74_O2_findings.md, /tmp/r74_CX2_findings.md (sfdp root causes)
- /tmp/r74_O_adversarial_findings.md, /tmp/r74_CX_adversarial_findings.md (guards + honest counts)

FIX A1 -- p_neg2 force-law clamp (the highest-ROI fix; ~52 combos).
  Root cause (verify in source, cite lines): graphviz INTERNALLY discards repulsiveforce=-2:
  sfdpinit.c clamps it via late_double(...,minimum=0.0) to 0, then spring_electrical.c does
  `if (p >= 0) p = -1`, so graphviz actually runs pow(dist, 2). dagua's sfdp.py (~line 539) runs
  pow(dist, 3) for the p_neg2 variant. FIX: replicate graphviz's parse clamp so the classic_sfdp_p_neg2
  variant runs with p = -1 (pow^2), i.e. it becomes identical to sfdp_default's force law.
  PRE-VERIFY (do this and report the result): confirm in the benchmark data that classic_sfdp_p_neg2
  REFERENCE rows are byte-identical to classic_sfdp_default reference rows (they should be, since
  graphviz clamps). If they are NOT identical, STOP and report -- the premise is wrong.
  REGRESSION GUARD: ~27 of the candidate combos currently have dagua stress < reference (D<R). Changing
  pow^3 -> pow^2 changes the force law; do not assume all improve. Keep the connected/other-variant paths
  byte-identical. Commit: "fix(sfdp): honor graphviz repulsiveforce clamp (p_neg2 runs inverse-square)".

FIX B1 -- disconnected-component layout + packing (~48 disconnected sfdp combos; guarded).
  Root cause (verify, cite lines): graphviz sfdpinit.c lays out each connected component INDEPENDENTLY
  (own srand reset) and then packs them (packSubgraphs / pack.c). dagua runs ONE shared force field with
  zero component handling. This is the SAME bug class already fixed for fmmm/neato/fdp -- reuse the
  existing graphviz polyomino packer (grep dagua/layout/ops for the neato polyomino / weak-components
  helpers, e.g. _weak_components and _compute_polyomino_step in the neato module -- REUSE, do not
  reimplement). Implement: detect disconnected input -> per-component sfdp layout with per-component RNG
  reset (match graphviz's srand-per-component) -> polyomino pack. Source-faithful to graphviz. NO runtime
  delegation. REGRESSION GUARD (MANDATORY, from adversarial review): the connected path must stay
  byte-identical; ~23 of ~57 disconnected candidates currently have dagua BETTER than reference (D<R) on
  stress -- the component packing must not make those WORSE. Apply ONLY to disconnected graphs. If you
  cannot guarantee the connected path is unchanged, scope the change behind a disconnected-only branch.
  Commit: "fix(sfdp): per-component layout + polyomino packing for disconnected graphs".
</task>

<default_follow_through_policy>
Most reasonable low-risk interpretation; proceed without stopping EXCEPT: stop on FIX-A1 if the p_neg2
reference rows are NOT identical to default (premise broken), or on any correctness/irreversible concern.
</default_follow_through_policy>

<constraints>
- NO RUNTIME DELEGATION (never import/call graphviz at runtime). Source-faithful ports only.
- per_combo.json final_rung is a STRING ("4") -- never compare ==4 (int).
- ASCII only. Conventional commits. NO AI attribution anywhere (no Co-Authored-By / "Generated with" /
  Happy footer) -- hard project rule, humans only.
- Quick targeted sanity check per fix (run sfdp pipeline on a small disconnected graph + a p_neg2 case);
  do NOT run the full benchmark (done separately). Do NOT edit .project-context, CLAUDE.md, AGENTS.md.
  Keep connected-graph and non-p_neg2 behavior byte-identical.
</constraints>

<verification_loop>
Per fix: import sanity + run the specific case; for B1 confirm a CONNECTED graph's output is unchanged
vs before the edit. git add the sfdp files + commit. Report each commit SHA, the p_neg2 ref-equality
result, and what changed, at the end.
</verification_loop>
