<task>
Run 4 small decisive experiments ordered by dagua's r75 adversarial critique (verdicts 18, 23, 26,
28 in .project-context/research/sprint_rng_matching/r75_findings/r75_ADVERSARIAL_VERDICTS.md; read
it + r75_mds_tails_{codex,sonnet}.md + r75_sugiyama_codex.md F5 for context). RESEARCH/PROBE ONLY:
no repo modifications, no commits; scratch in /tmp; in-memory monkeypatching allowed.

Repo: /home/jtaylor/projects/dagua (develop @ 89ed3c3). Installed python-igraph is 1.0.0 --
version-pin any igraph behavioral claim by tracing the INSTALLED wheel at runtime (the
_references/igraph source tree is unpinned; do not cite it as installed truth).

E1 -- igraph sugiyama LP objective (verdict 18): does installed igraph 1.0.0 use IN/IN degree
vectors (as the unpinned source suggests at sugiyama.c:588-615) or out-minus-in (as dagua's r74
fix assumed, dagua/layout/ops/sugiyama.py:495-543)? Construct 2-3 small directed DAGs where the
two objectives yield DIFFERENT optimal layer assignments (design them: asymmetric in/out degree
distributions), run installed igraph ig.layout('sugiyama') and compare realized y-layers against
both predictions. Verdict: dagua objective RIGHT / WRONG / indistinguishable on constructibles.
E2 -- connected classical_mds eigenspace test (verdict 23): for the 7 connected divergent graphs
(bipartite_4_3_4, center_port_backedge_hub, densenet_block, org_chart_1_5_4_8, petersen_10,
wide_single_layer_1_50_1, wide_3_50_3 -- from r75_targets_classical_mds.json), compute dagua's MDS
embedding subspace and the installed-igraph reference embedding subspace (top-2 eigenvectors of
the double-centered distance matrix): (a) do eigenvalue gaps show degenerate/near-degenerate top
eigenspaces (report lambda1,lambda2,lambda3 + gaps)? (b) is the reference embedding inside dagua's
eigenspace (principal-angle / subspace residual)? (c) does scipy eigh driver choice (evr/evx/evd)
change dagua's result? Verdict per graph: BASIS-ONLY (subspace matches, coordinates differ -> a
deterministic basis-selection port could fix) / GENUINE numerical floor (with the eigengap
evidence) / OTHER.
E3 -- maxent random_dag_50 first-divergence (verdict 26): the 3 remaining maxent rows are all
random_dag_50 (disconnected). Trace OGDF StressMinimization's initial layout for disconnected
input in the RUNNER's source (/home/jtaylor/tools/ogdf-src, StressMinimization.cpp:69-123
componentLayout/PivotMDS init + infinite-distance replacement avgEdgeCosts*sqrt(n) :94-100) vs
dagua's maxent pipeline init on the same graph (benchmark path, seed 42). Compare positions at
step 0 (before majorization): does the INITIALIZATION already diverge structurally (component
placement / distance fill), or only the iterations? Deliverable: first-divergence stage + minimal
fix sketch (NO blanket component splitting -- that was reverted in r74).
E4 -- neato disconnected pack/RNG probe (verdict 28): 2 rows (parallel_cycles_4x5, random_dag_50).
dagua seeds each component seed+i (dagua/layout/ops/pipelines/neato.py:1404-1419); graphviz uses
one process-wide stream. Probe: monkeypatch the component seeding to a single shared stream
(component_index=0 for all, or one sequential RNG), run benchmark path 5 seeds on both graphs, and
compare stress/Procrustes vs the saved graphviz_fdp/neato references (main repo eval_output
overlay dirs, read-only) against the unpatched run. Verdict: seeding-policy CONFIRMED-CAUSE /
KILLED (+ whether pack=false changes the picture, if the reference adapter honors it -- check
variants.py:877-883).

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r75_PROBE_tails_RESULTS.md --
per experiment: commands, raw numbers, verdict, recommended gated fix or explicit kill/floor
disposition (floors need the numeric evidence attached). ASCII only. Runtime budget ~45 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one experiment blocks, document and continue with the
others -- do not stall the batch.
</default_follow_through_policy>
