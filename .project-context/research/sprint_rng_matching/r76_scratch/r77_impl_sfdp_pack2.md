<task>
r77-C1: sfdp packing -- close the spline-occupancy gap as far as possible without porting
spline routing. Context (READ FIRST): .project-context/research/sprint_rng_matching/
r75_findings/r76_IMPL_sfdp_disc_NOTES.md section C4d. The point-unit pack fix landed; 5
disconnected graph clusters remain divergent with the residual attributed to graphviz
packing polyominoes over SPLINE-routed edge boxes (pinfo.doSplines=1) while dagua packs
node boxes only. On 2 clusters dagua is ~5% WORSE (kitchen_sink_platform_graph 0.5047 vs
0.4791; multi_component_80 0.9154 vs 0.8662) -- JMT's quality bar demands we push those.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sfdp-pack2 (branch r77/sfdp-pack2, off
develop which has the pack fix). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

STEP 1 -- BOUND THE SPLINE EFFECT (instrumented, not inferred): `mkdir -p /tmp/gv750-pack2
&& git -C /home/jtaylor/projects/_references/graphviz archive 7.0.5 | tar -x -C
/tmp/gv750-pack2`. Build sfdp twice-instrumented: dump polyomino cell sets per CC (a) as-is
(doSplines=1) and (b) patched with doSplines=0. Run both on kitchen_sink_platform_graph +
multi_component_80 (seed 100). Compare final packed offsets a-vs-b: if (b) reproduces (a)'s
offsets (or nearly), SPLINES ARE NOT THE RESIDUAL -- find what is (margins? CL_OFFSET?
label boxes? sort order on these graphs?) and name it. If (a) differs materially from (b),
quantify: does dagua's current packing match (b)? If dagua==(b), the gap IS spline
occupancy and step 2 approximates it.
STEP 2 -- CLOSE WHAT'S CLOSABLE: implement the best non-spline approximation that moves
dagua toward (a): straight-line edge segment boxes rasterized into the polyomino occupancy
(graphviz's own poly_
cells use edge boxes -- read lib/pack/pack.c genPoly/edge handling for the exact cell
inflation), or label-box occupancy if step 1 fingers labels. Gate to graphviz-fidelity
disconnected sfdp only.

GATES (before commit): benchmark-path W gap to reference shrinks on kitchen_sink AND
multi_component (the quality-worse pair) with no regression on the other 3 clusters or any
previously-identical/equivalent sfdp row (hash gates as C4d: 33-row sample); pytest -k sfdp
green; ruff clean. KNOWN pre-existing failures (must not block): test_bench_large;
classic_fcose; double-border render smoke. Commit on r77/sfdp-pack2; bench the 8 disc
graphs (pattern of benchmark_100seed_r76_sfdp_fix3, output
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_sfdp_pack2) -- 0 errors.

DELIVERABLES: append "## Pack2: spline-occupancy bound + approximation" to
r76_IMPL_sfdp_disc_NOTES.md (the a-vs-b bound tables, what was implemented w/ pack.c cites,
before/after W, gate evidence, commit sha OR the dossier proving the remaining gap is
irreducibly spline-geometry). Clean /tmp/gv750-pack2. ASCII. NO AI attribution. No
push/merge.
</task>
<completeness_contract>
Done = the spline-effect BOUND measured (a-vs-b) AND (gated improvement committed + clean
bench, OR dossier showing dagua already matches the no-splines packer and the residual is
irreducibly spline geometry -- which upgrades the named cause from inference to
measurement). Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/sfdp-pack2 only. Never modify shared packer defaults,
other engines, eval scoring, reference runners, or the reference clone. Bench write to
benchmark_100seed_r77_sfdp_pack2 only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
