<task>
R33 BIT-EXACT PURSUIT for graphviz primary 4 (dot, sfdp, neato, fdp).

Current 100-seed verdict:
- dot/sugiyama: TOST 0.25x, RMSD 0.031 (already very tight)
- neato: TOST 0.25x, RMSD 0.065
- fdp/fmmm: TOST 0.5x via R31, RMSD 0.07-0.18
- sfdp: TOST 0.25x, RMSD 0.08-0.10

Goal: push toward bit-exact (RMSD < 0.001) by reading lib/dotgen, lib/fdpgen, lib/neatogen, lib/sfdpgen C source deeper than prior rounds.

## R27 + R28 prior work

Already did:
- R27 line-by-line diffs for dot/neato/fdp/sfdp
- R28 sfdp fine-cooling/force-norm/recenter/quadtree fixes (3.3x improvement)
- R28 dot lattice spacing fix

So this round = remaining items past R27/R28.

## Read

- eval_output/algo_fidelity/round_27/{sfdp,dot,neato,fdp}/PLAN_*.md and DIFF files
- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/dot.c` (or dotinit.c)
- `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/spring_electrical.c`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neato.c`
- `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c`

Identify remaining divergences not addressed in R27/R28. Examples from R27 diffs:
- dot: network simplex rank assignment, mincross, network-simplex x-position, clusters
- neato: PCA initialization step
- fdp: derived-graph recursion, cluster expansion, overlap removal
- sfdp: sequential update path, matrix-based coarsening

These were all flagged as "wholesale rewrite" in R27/R28. The user explicitly said "leave no stone unturned" — so try them.

## Implementation

Pick ONE family with the highest expected ROI and go deep. Multi-commit via commit-safe.

Realistic scope: 1-2 features per family. e.g., dot's mincross + network simplex x-position (network simplex is ~500 LoC port).

## Verify

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot --seeds 30 --graphs <full bounded + medium> --output-dir eval_output/algo_fidelity/round_33/graphviz_bitexact/dot_post
# ... same for neato/fdp/sfdp
```

## Output
`eval_output/algo_fidelity/round_33/graphviz_bitexact/SUMMARY.md` with per-family before/after + which items ported.
</task>

<completeness_contract>
At minimum: 1 substantive port from at least 1 family. Document which residuals remain.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Read deeply.
</default_follow_through_policy>
