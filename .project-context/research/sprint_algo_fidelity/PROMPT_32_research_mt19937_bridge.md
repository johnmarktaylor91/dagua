<task>
R32 RESEARCH -- feasibility of porting igraph's MT19937 RNG into dagua.

## Why this matters

Multiple R31 plans (drl, graphopt, lgl) identified RNG mismatch as the dominant architectural floor:
- igraph uses C MT19937 (`igraph_rng_default`)
- dagua uses Python `random.Random` (also MT19937 but DIFFERENT state because of different seeding/state advance)
- Same seed != same draws

If we port igraph's MT19937 exactly, we get bit-exact paired seeds for drl/graphopt/lgl and likely close architectural floor.

## Tradeoff

CLAUDE.md and the project README emphasize: "PyTorch is the only required dependency." Adding igraph as a runtime dep breaks this.

BUT -- we could:
- Port the MT19937 STATE ADVANCE in pure Python/PyTorch (no dep on igraph C)
- Just need to match igraph's state encoding, seed schedule, and draw extraction exactly

## Your job

PURE RESEARCH. No code edits.

1. Read `/home/jtaylor/projects/_references/igraph/src/core/random/random.c` and `/home/jtaylor/projects/_references/igraph/src/core/random/mt19937.c` (if present; or check the actual filename).
2. Determine igraph's RNG state representation, seed function, advance function, draw function.
3. Write a feasibility plan: can we implement a pure-Python `IgraphMT19937` class that produces bit-exact draws from a seed? Estimate LoC.
4. Identify which dagua engines would benefit: drl (yes per R31), graphopt (yes), lgl (yes), maybe others.

## Output

`eval_output/algo_fidelity/round_32/mt19937_bridge/REPORT.md` with:
- igraph RNG implementation summary
- Pure-Python port feasibility (LoC estimate, complexity, edge cases)
- Per-engine impact estimate (which engines benefit, expected RMSD delta)
- Recommendation: proceed (do the port), defer (not worth it), or pass (architectural mismatch impossible)
</task>

<research_mode>
Diagnostic round. Output is the REPORT.md.
</research_mode>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Read deeply.
</default_follow_through_policy>
