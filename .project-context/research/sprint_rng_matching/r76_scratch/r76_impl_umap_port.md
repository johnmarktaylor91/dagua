<task>
r76-C3: umap scalar-faithful SGD port. JMT BOUNDARY RULING (binding, 2026-07-02): "PUSH TO
EXHAUSTION: umap scalar-faithful SGD port (tau_rand xorshift stream is portable; target
distributional match, bit-exact if float32 order cooperates)." 7 combos remain divergent for
dagua's umap engine family. A rival-lab (Anthropic) critic will review the result.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-umap-port (branch r76/umap-port, fresh off
develop). Work ONLY here. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

PHASE 1 -- FIRST-DIVERGENCE TRACE (MANDATORY BEFORE ANY CODE CHANGE; adversarial verdict #24
from r75 requires it): on a tiny graph (3-5 nodes, 1 seed), instrument BOTH dagua's umap
pipeline (dagua/layout/ops/pipelines/ + the umap ops it composes) and the reference
umap-learn package (the INSTALLED version in this environment -- cite file:line from
site-packages; check `python3 -c "import umap; print(umap.__version__, umap.__file__)"`).
Compare stage by stage: (1) kNN/fuzzy simplicial set construction (rows/cols/vals), (2)
initialization (spectral or random -- exact values), (3) the FIRST SGD epoch (per-sample:
which edge, which negative samples, the tau_rand xorshift draws, the gradient values, the
clipped moves). Name the FIRST diverging quantity precisely. Write the trace table into the
notes file BEFORE changing code.

PHASE 2 -- SCALAR-FAITHFUL PORT: guided by the trace, port the reference behavior into
dagua's umap ops so the SGD loop is scalar-faithful: same edge-visitation schedule
(epochs_per_sample machinery), same tau_rand xorshift RNG stream and consumption ORDER for
negative sampling, same gradient forms/clipping, same float32 arithmetic where the reference
uses float32. Numba JIT vs python/torch op-order differences may block bit-exactness -- that
is acceptable ONLY after you demonstrate the streams/schedules match draw-for-draw and the
residual is pure float summation order (show one concrete example). TARGET RUNG:
statistically indistinguishable distributions (the benchmark's equivalence legs); bit-exact
is the stretch goal if float32 order cooperates.

FIND THE COMBOS: eval_output/fidelity_definitive/r75_final.jsonl -- rows with engine
containing umap, quality_identical_raw=false, no_canonical_reference!=true (expect 7).
Positions for comparison: per-combo freshest-dir overlay across
eval_output/benchmark_100seed_* dirs (r75_fixes and umap_realfix are the likely freshest for
umap).

GATES (all must pass before commit; else document honestly, leave uncommitted):
1. Phase-1 trace table exists with the named first divergence.
2. Probe evidence: on >=3 representative divergent combos' graphs (small ones), 5 seeds each,
   post-fix RMSD or distributional distance vs reference materially improves; zero
   regressions on 3 previously-IDENTICAL umap combos (verify via the same probe path).
3. pytest tests/ -k "umap" -x -q green; ruff clean on touched files.
4. NO runtime delegation: dagua ops must never import or invoke umap-learn/numba at runtime
   (r51/r58 incident class). Reference package is for offline tracing only.
5. Everything scoped so ONLY the umap engine family changes -- no shared-op behavior changes
   for other engines (if you must touch a shared op, add a gated parameter defaulting to
   current behavior).

DELIVERABLES: .project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_umap_NOTES.md
(trace table, what was ported with reference file:line cites, probe numbers, gate evidence,
commit sha). Conventional commits on r76/umap-port; re-add/re-commit through ruff-format until
`git log` SHOWS them. No push/merge. NO AI attribution in commits. ASCII only.
</task>
<completeness_contract>
Done = phase-1 trace named the first divergence AND (gates 1-5 pass with commits, OR a precise
documented blocker naming the exact non-portable construct with file:line cites and NO
commit). Never weaken a gate; never claim distributional match without probe numbers.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/umap-port only. Never touch other engines' pipelines, eval
scoring code, or reference runners. Never modify files outside the worktree.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
