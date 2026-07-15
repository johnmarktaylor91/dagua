<task>
r78-G2: GEM ba_500 first-divergence trace -- the one missing artifact for 27 rows (24
SUPERIOR_DISTINCT gem rows + 3 DIVERGENT ba_500 gem rows share the suspected cause). READ
FIRST: .project-context/research/sprint_rng_matching/r75_findings/r78_SUPERIOR_NOTES.md
(this worktree, committed 3b704d9) -- its follow-up spec names exactly this: "a bounded
temporary OGDF runner trace for one 500-node row (ba_500 seed 100)".

CONTEXT: gem is near-bit-exact on small graphs post r76 round-budget fix (5e-08 RMSD). On
larger graphs (ba_500 class) dagua's layouts land in systematically BETTER basins
(superior) or diverge (the 3 ba_500 rows). Whether that's a nameable op (order/float path/
tie-break) or pure accumulated-float chaos has never been traced at this scale.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-superior (branch r78/superior).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

METHOD: instrumented runner copy in /tmp/gem-trace (scripts/ogdf_runner.cpp + the recipe in
scripts/rng_match/build_ogdf_runner.sh; NEVER modify the committed runner/ogdf-src). Dump
per-update for gem on ba_500 seed 100, gemRounds matched to the iters100 variant: node id,
impulse, temperature, position before/after, RNG draws. Mirror from dagua's gem pipeline
(same-process graph build). Diff update-by-update; find the FIRST diverging quantity.
- Nameable op -> port it gated to gem fidelity (gates: RMSD collapse on ba_500 3 seeds +
  byte-identity on 5 previously-identical gem rows + pytest -k gem green + ruff; commit).
- Pure float accumulation -> proof excerpt (first N matching updates, the divergence
  point, the accumulation mechanism) -> the 24 superior + 3 divergent rows get
  instrument-grade terminal evidence.

DELIVERABLES: append "## G2: gem ba_500 trace" to r78_SUPERIOR_NOTES.md (trace tables,
verdict, port+commit sha OR proof, gate evidence). ASCII. NO AI attribution. No push/merge.
Clean /tmp/gem-trace. KNOWN pre-existing failures (must not block): the standard 6-item
list. Commits on r78/superior AUTHORIZED on gate pass.
</task>
<completeness_contract>
Done = first-diverging update NAMED from the running instrumented trace AND (port committed
w/ gates, OR accumulation proof w/ excerpt). Source-reading-only is not acceptable.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/superior only. Never modify committed runner/ogdf-src/
other engines/eval scoring. No runtime reference invocation from dagua/layout.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices.
</default_follow_through_policy>
