<task>
r76-D5: FORMAL FLOOR DOSSIERS (research/probe ONLY; no repo code changes; scratch in /tmp;
one output file). JMT's rule for divergent-with-known-cause dispositions: the cause must be
NAMED with evidence and quality must NOT be lesser. Two clusters have all the qualitative
evidence but lack the formal quantitative dossier. Produce it.

Repo: /home/jtaylor/projects/dagua (develop, read-only). ASCII only.

CLUSTER 1 -- MDS CONNECTED (14 rows, igraph_mds reference):
Context: r76_scratch/r76_probe_mds_gem_triage.md + r75_findings/r75_mds_tails_*.md found
machine-precision eigenvalue ties where the eigendecomposition flips coordinate signs/basis.
JMT RULING (binding): NO eigensolver vendoring; instead UPGRADE the label to "proven member
of reference equivalence class". Rows: eval_output/fidelity_definitive/r75_final.jsonl,
engine contains classical_mds, quality_identical_raw=false, no_canonical_reference!=true,
connected (disconnected excluded). For EACH distinct graph in those rows:
1. EIGENGAP TABLE: compute the double-centered distance matrix B's eigenvalues (the MDS
   input); report the gap between the 2nd/3rd (and any tied) eigenvalues in ULPs/relative
   terms -- document the degeneracy that makes basis selection arbitrary.
2. EQUIVALENCE-CLASS PROOF: take dagua's layout and the reference layout for 3 seeds;
   show an orthogonal transform WITHIN the degenerate eigenspace (sign flips/rotations
   spanning tied eigenvectors) maps one onto the other to near machine precision (report
   residual RMSD after the transform). That proves both are members of the same solution
   equivalence class of the same algorithm.
3. QUALITY PARITY: stress/crossings D vs R means from the ledger rows (should be ~equal).

CLUSTER 2 -- UMAP DISCONNECTED SPECTRAL (2 rows: random_dag_50::classic_umap_nn5,
random_dag_200::classic_umap_nn5):
Context: r75_findings/r76_IMPL_umap_NOTES.md (worktree copy is at
/home/jtaylor/.claude/worktrees/dagua-umap-port/.project-context/research/sprint_rng_matching/
r75_findings/r76_IMPL_umap_NOTES.md -- read the "Attempt 2" bisection: fuzzy graph, schedule,
curve, RNG state ALL match exactly; sole divergence = spectral init second-eigenvector basis
in a near-degenerate component eigenspace, max diff 0.376; even substituting the reference's
own compiled kernel leaves RMSD ~0.14). Produce:
1. EIGENGAP TABLE: for the fuzzy graph's normalized Laplacian per component (random_dag_50:
   components [52,45]), the eigengap around the eigenvectors used for 2D init -- quantify
   the near-degeneracy.
2. 1-ULP PERTURBATION EXPERIMENT (the missing piece): run dagua's umap twice on
   random_dag_50 (seed 100, nn5 params) -- once stock, once with ONE spectral-init
   coordinate nudged by 1 ULP. Report final-layout Procrustes RMSD between the two runs vs
   the dagua-vs-reference RMSD (from stage-1b, ~0.14-0.18). If perturbation RMSD is
   comparable, chaos amplification of basis-selection noise is PROVEN. Repeat for 3 seeds.
3. QUALITY PARITY: stage-1b W means (dagua 0.17919 vs ref 0.17935 on _50; 0.18340 vs
   0.17283 on _200) -- present with TOST context.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r76_FLOOR_DOSSIERS.md --
per-cluster: evidence tables, the formal disposition text (ready to paste into the ledger:
"proven member of reference equivalence class" for mds; "evidenced FP-chaos floor
(eigenspace basis selection), quality parity shown" for umap), commands used. Budget ~60 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if a computation is intractable in budget, document
the attempt and continue with the rest.
</default_follow_through_policy>
