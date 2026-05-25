<task>
R35 NEW ENGINE: add OpenOrd to dagua.

OpenOrd (Martin et al. 2011) is a force-directed multilevel layout known
for very large graphs. Used in Gephi. Adds another fidelity-paired family.

## Your job

### Phase A: Reference

Check if a Python OpenOrd reference is available:
- `python -c "import openord; print(openord.__file__)"` (likely no)
- igraph has `layout_drl` (which IS OpenOrd!) — drl is openord. So we already have a reference.

Wait — verify: drl IS the implementation of openord per igraph docs. If so,
adding openord variants would duplicate drl variants. Confirm or deny via:
```
grep -i openord /home/jtaylor/projects/_references/igraph/src/layout/drl/*.cpp
```

If openord == drl, document and skip implementation (no new engine).
If openord is genuinely different (some Gephi-specific variant), implement.

### Phase B: Implement (if confirmed distinct)

Add `dagua/layout/ops/pipelines/openord.py` following the same pattern as
fcose/yifanhu R33 additions. Variants per Gephi defaults:
- openord_default
- openord_liquid (liquid stage emphasis)
- openord_simmer
- openord_crunch

### Phase C: Verify

Bounded smoke test (no reference unless we find one).

## Output
`eval_output/algo_fidelity/round_35/openord/SUMMARY.md` with either:
- "openord == drl, no new engine added" (with proof), OR
- working openord pipeline + tests

Use commit-safe wrapper.
</task>

<completeness_contract>
Verify openord vs drl relationship. Either implement distinct openord OR document equivalence.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
