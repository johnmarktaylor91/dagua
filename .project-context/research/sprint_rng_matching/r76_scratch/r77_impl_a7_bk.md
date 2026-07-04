<task>
r77-A7: igraph Brandes-Koepf X-STAGE parity -- the residual UNDER the now-solved rank LP.
Context (READ FIRST, in this worktree): .project-context/research/sprint_rng_matching/
r75_findings/r76_IMPL_igraph_NOTES.md (A3: ported the ordinal-edge Type-1 conflict quirk;
A6: GLPK rank parity landed -- rank vectors now MATCH installed igraph on the probe set,
including real_karate_34). With ranks exact, the remaining igraph-family divergence is the
BK coordinate assignment: real_karate_34 d_R 0.092, hexagonal_lattice_42 and
width_skew_late_merge still x-stage divergent. Bisect and port.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-igraph-glpk (branch r77/igraph-glpk, HEAD
a61bd2e -- contains the GLPK port; MERGED to develop already; keep committing here).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl. swiglpk is installed -- the GLPK path is active.

METHOD -- BISECT WITH MATCHED RANKS: on real_karate_34 + width_skew_late_merge (1 seed),
with ranks now identical to installed igraph, compare the ordering stage (should match
after A3's quirk port -- VERIFY), then the BK sub-stages against igraph's implementation
(igraph C source to /tmp/igraph-src, READ ONLY; the BK code lives with the sugiyama layout
source): vertical alignment (Type-1/Type-2 conflict marking -- A3 ported the ordinal quirk;
check Type-2 and the marking order), horizontal compaction (block placement order, class
shifts), the 4-directional runs and their BALANCING (median/average of the 4 candidate
coords -- igraph may use a specific combination or subset), and any final
normalization/anchor. If source reading is ambiguous, build an instrumented python-igraph
in a /tmp venv (fprintf in the BK code; pip install ./ from patched source; NEVER touch the
env's installed igraph) and dump per-node candidate coords per direction. Name the first
diverging quantity; port it gated to fidelity_mode="igraph".
NOTE: a prior broad repeated-igraph scan SEGFAULTED after ~75 consecutive layout calls --
batch reference invocations in fresh subprocesses to avoid it.

GATES (before commit):
1. d_R < 0.01 on >=7 of the 10-row A6 probe set (5 already pass via GLPK; the port should
   flip real_karate/hexagonal/width_skew classes), NO row leaving bit-exact/near.
2. Zero regressions: the 60+ bit-exact igraph rows sample (10 rows x 3 seeds)
   byte-identical; graphviz-fidelity rows byte-identical (5-row sample); no-swiglpk
   fallback tests still green.
3. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
   pre-existing failures (must not block): test_bench_large; classic_fcose; double-border
   smoke.
4. Commit on r77/igraph-glpk. Then FULL family bench: run_benchmark --engines
   classic_sugiyama --variants --max-nodes 0 --seeds 100 --seed-start 100 --workers 5
   --timeout 3600 --watchdog-timeout 7200 --output-dir
   /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_igraph_bk -- 0 errors.

DELIVERABLES: append "## A7: BK x-stage parity" to r76_IMPL_igraph_NOTES.md (bisection
tables, named quantity w/ source cite, before/after d_R on the probe set, gate evidence,
commit shas, bench Done line). ASCII. NO AI attribution. No push/merge. Clean /tmp scratch.
</task>
<completeness_contract>
Done = first BK divergence NAMED from bisection AND (gated port committed + clean bench, OR
a dossier proving the specific BK behavior is non-portable with the instrumented dump
shown). This is the LAST igraph work item -- an honest dossier ends the family either way.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/igraph-glpk only. NEVER modify the env's installed igraph.
Never touch graphviz-fidelity paths, eval scoring, reference runners. Bench write to
benchmark_100seed_r77_igraph_bk only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
