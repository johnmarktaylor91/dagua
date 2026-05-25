<task>
R36 BIT-EXACT graphviz port. Sub-task: **dot / mincross (crossing minimization)**.

This is part of a 13-way parallel sprint to push graphviz dot/neato/fdp/sfdp
from strong_equivalent (RMSD ~0.03) to bit-exact (RMSD <1e-3).

## Your sub-task

Port the mincross (crossing minimization) component of graphviz dot from C to Python/PyTorch,
inside dagua under `fidelity_mode`. Keep existing behavior intact by default;
only change behavior when `fidelity_mode` is enabled.

## Read these reference C files line-by-line

- /home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c (median heuristic + alt iteration + transposition)

## Dagua target files

- dagua/layout/ops/pipelines/dagua_native.py or new module
- dagua/layout/ops/pipelines/sugiyama.py

## Implementation

1. Read the reference C files end-to-end.
2. Implement the mincross (crossing minimization) logic in Python/PyTorch under `fidelity_mode`.
3. Add unit tests with golden vectors captured from the reference behavior
   (or document if golden capture is infeasible).
4. Run focused pytest (`tests/test_pipeline_<family>.py` + your new test file).
5. Verify against bounded subset: run live_compare for the parent family if your
   component is invokable end-to-end. If not (you're a sub-component), just
   verify your tests pass.

Focus: take ranks from dot_rank component, produce ordering within each rank to minimize edge crossings. Implement `graphviz_mincross(ranks, edges, iterations=24)` -> ordering. Median heuristic + transposition refinement per the C source.

## Scope

- DO NOT TOUCH: render/styles, cluster sprint files, existing fidelity_report
  outputs or benchmark_100seed_final results.
- Stage commits with explicit `git add`.
- Use commit-safe wrapper: `bash scripts/commit-safe.sh -m "..."`
- Commit format: `feat(layout): round 36 dot_mincross -- <terse>`. Multi-commit OK.

## Output

`eval_output/algo_fidelity/round_36/dot_mincross/SUMMARY.md` with:
- Source files read
- Implementation summary
- Tests added
- Any blockers / interface assumptions for the integration codex
</task>

<completeness_contract>
Either implement the component OR document explicit interface decisions if you
can't isolate it from sibling sub-components. Integration codex will wire up.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Read deeply.
</default_follow_through_policy>
