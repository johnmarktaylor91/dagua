<task>
r76-C4b: CONNECTED SFDP first-divergence bisection, then fix-or-floor. JMT RULING (binding):
NO "FP-chaos floor" label until first-divergence bisection stops finding op differences;
floor claims need 1-ULP perturbation evidence. 23 connected sfdp rows remain divergent
(graphs: asymmetric_hourglass_hub, hexagonal_lattice_42, real_karate_34, weighted_karate_34,
weighted_chain_20, planar_60, real_lesmis_77, sparse_pair_50, long_range_residual_ladder,
clustered_longlabel_handoffs; RMSD 0.01-0.14).

READ FIRST: .project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_sfdp_triage.md
(Cluster B + its "do not re-litigate" list: coarsening RNG raw-modulo is CORRECT for 7.0.5,
overlap=false REJECTED, p_neg2 clamp CORRECT). Also r75_findings/r75_sfdp_codex.md +
r75_sfdp_sonnet.md for prior art. KNOWN GOOD: initial random coordinates match libc/graphviz
exactly for seed 100 (probe-verified: 0.315598, 0.284943, ... pairs). The divergence is
BETWEEN matched random start and final convergence: multilevel matching/coarsening,
prolongation, or spring-electrical iteration internals.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sfdp-conn (branch r76/sfdp-conn, fresh off
develop). Work ONLY here. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

PHASE 1 -- BISECTION VIA INSTRUMENTED LOCAL GRAPHVIZ BUILD (mandatory):
Build an instrumented graphviz 7.0.5 IN /tmp for TRACE PURPOSES ONLY (this is sanctioned; it
is NOT a reference build -- references stay the installed binaries). Source: `git -C
/home/jtaylor/projects/_references/graphviz worktree add /tmp/gv750-trace 7.0.5` (or git
archive) -- NEVER dirty the reference clone itself. Add fprintf(stderr,...) instrumentation to
lib/sfdpgen/Multilevel.c + spring_electrical.c dumping, per level: matrix size, matching/
cluster map, coarsest initial coordinates, first 3 iteration force norms + step sizes, and
prolongation output coords. Build only what's needed (sfdp + deps; ./configure with minimal
flags or cmake; a partial static build driving sfdp_layout via a tiny C harness is also
acceptable). Mirror the same dumps from dagua's pipeline ops (BuildGraphvizSFDPMatrixHierarchy,
the sequential-step op, prolongation/jitter ops -- find exact op names in
dagua/layout/ops/pipelines/sfdp.py and the ops it composes).
Compare stage-by-stage on asymmetric_hourglass_hub AND hexagonal_lattice_42, seed 100 (dagua
hierarchy currently: hourglass levels [14,8,4] K0=0.4383; hexagonal levels [42,24,12,6]
K0=0.3065 -- verify graphviz agrees or find the first mismatch). DOT input built exactly as
dagua/eval/competitors/graphviz_competitor.py does. Name the FIRST diverging quantity: which
level, which stage, which values.

PHASE 2 -- FIX OR FLOOR:
- If an op difference is found (expected): implement the smallest GATED fix in the worktree,
  scoped so only classic_sfdp behavior changes. Then verify: probe-path RMSD on the 2 trace
  graphs + 3 more from the cluster list (5 seeds) materially improves; zero regressions on 5
  previously-identical sfdp rows (byte-identical positions pre/post); pytest -k sfdp green.
  Commit on gates passing.
- ONLY if bisection exhausts every stage with NO op difference (all intermediate quantities
  match to float rounding and divergence still emerges): run the 1-ULP perturbation
  experiment -- nudge one initial coordinate by 1 ULP in the reference-matched dagua run and
  show the final-layout divergence pattern/magnitude reproduces the observed dagua-vs-reference
  gaps (RMSD, stress deltas of comparable size). Write the FLOOR DOSSIER with the bisection
  endpoint + perturbation tables. NO commit in this case.

DELIVERABLES:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_sfdp_conn_NOTES.md
(bisection trace tables, first-divergence naming, fix description w/ 7.0.5 cites OR floor
dossier, gate evidence, commit sha if any). Conventional commits on r76/sfdp-conn only if
gates pass. No push/merge. NO AI attribution in commits. ASCII only. NO runtime delegation:
dagua ops never invoke graphviz at runtime; the instrumented build is offline trace tooling.
</task>
<completeness_contract>
Done = first divergence NAMED with trace evidence AND (gated fix committed with gates green,
OR floor dossier with perturbation evidence and no commit, OR precise documented blocker).
Never claim floor without the perturbation experiment. Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Never dirty /home/jtaylor/projects/_references/graphviz's working tree
(build from a /tmp worktree/archive; remove the git worktree registration when done:
`git -C /home/jtaylor/projects/_references/graphviz worktree remove /tmp/gv750-trace --force`
at the end). Never touch other engines, eval scoring, reference runners. Never modify files
outside the dagua-sfdp-conn worktree except /tmp scratch.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
