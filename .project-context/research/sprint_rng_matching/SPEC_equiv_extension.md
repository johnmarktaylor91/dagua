<task>
Extend the EXISTING layout-equivalence module with the final two invariances. The trio (rigid +
automorphism + degenerate-eigenspace/stress) is already committed at f9d18e1 in
dagua/eval/equivalence_metrics.py (745 lines) + scripts/equivalence_report.py + tests.

READ the full spec at .project-context/research/sprint_rng_matching/SPEC_equivalence_metrics.md --
specifically the "FOLLOW-UP ADDITIONS" section at the bottom (Additions 4 and 5). Implement EXACTLY
those two by extending the existing module (do NOT rewrite the trio; add to it):

  Addition 4 -- per-connected-component rigid placement: component_aligned_rmsd (decompose via igraph
  connected components; per-component rotation+reflection+translation, single GLOBAL uniform scale;
  pooled RMSD; n_components; no-op == global RMSD for connected graphs). Own signal alongside
  aut_procrustes_rmsd.

  Addition 5 -- per-axis anisotropic scaling, OPT-IN: anisotropic_rmsd (align rot+refl+trans then fit
  per-axis scale_x,scale_y by least squares). GATED by FREE_ASPECT_ENGINES allowlist
  (default = {"classic_sugiyama"}); null/N/A for non-allowlisted engines (granting an unowned
  invariance hides bugs). NOTE: the trio found sugiyama petersen does NOT collapse under automorphism
  alone (0.85 -> 0.60) -- the anisotropic invariance is the expected fix; verify whether it now collapses.

Extend the PRACTICALLY_EQUIVALENT verdict disjunction: also pass if component_aligned_rmsd < 1e-3 OR
(engine in FREE_ASPECT_ENGINES AND anisotropic_rmsd < 1e-3). Keep emitting all raw signals.
</task>
<constraints>
- The 5-seed benchmark may STILL be running -- do NOT touch run_benchmark/competitors/variants/benchmark.py.
  Extend ONLY dagua/eval/equivalence_metrics.py + scripts/equivalence_report.py + the test file.
- No layout delegation (igraph for component-decomposition/automorphism analysis is fine). float64.
- Reuse the existing Procrustes/alignment helpers already in the module. Do NOT commit (CC commits).
</constraints>
<verification>
- Add tests: 2-component graph placed/oriented differently per component -> component_aligned_rmsd ~0
  while global RMSD large; sugiyama-like layout stretched independently in x/y -> anisotropic_rmsd ~0,
  AND a non-allowlisted engine does NOT get the anisotropic pass. Existing 4 tests must still pass.
- Rerun scripts/equivalence_report.py on the live holdouts (export
  LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:$LD_LIBRARY_PATH); report whether
  classic_sugiyama_default petersen now flips to PRACTICALLY_EQUIVALENT via anisotropic. Paste numbers.
- ruff + mypy clean on the changed files.
</verification>
<default_follow_through_policy>
Proceed autonomously; most reasonable low-risk interpretation. Report the sugiyama anisotropic
before/after as the key result, what you reused, and any choices.
</default_follow_through_policy>
