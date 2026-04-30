<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 27 RESEARCH + LINE-BY-LINE diff for **classic_linlog**.

`classic_linlog` (5 variants in dagua/eval/variants.py) has **NO reference
comparator** at all -- the variants registry shows `original_engine=None`
for every linlog variant. This means linlog has never been compared against
any original implementation.

## Your job (two phases)

### Phase 1: Find an upstream LinLog reference

The LinLog energy model is by Andreas Noack (2003). Original sources:
- "Energy-Based Clustering of Graphs with Nonuniform Degrees" (Noack 2005)
- Reference implementation: search the web for Noack's original LinLogLayout
  Java code (likely on his Cottbus university page or GitHub mirror)
- Alternative: igraph has `layout_drl` which is LinLog-derived but distinct
- Alternative: NetworkX or networkit may have a linlog layout

Use:
```
gh search repos linlog layout
gh search code "linlog" --filename layout.py
```
Plus targeted web search via the exa MCP if available, or `WebSearch` for the
Java source.

If you find an upstream reference:
- Determine if it's installable in the eval environment (Python wrapper, JAR,
  pip package, anything)
- If installable, ADD a competitor adapter under
  `dagua/eval/competitors/` and a corresponding entry in
  `dagua/eval/variants.py` for the five linlog variants
- If not installable, document the absence in the diff doc

### Phase 2: Line-by-line if reference available

If you successfully added a reference, then do the standard adversarial
line-by-line:
- Dagua linlog: `dagua/layout/ops/pipelines/linlog.py`
- Reference: whatever upstream you found

Write the diff doc at
`.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_linlog.md`
covering:
- Reference search results (with URLs)
- Whether you successfully wired the reference; if yes, how
- Line-by-line diff (if reference available) with same format as other R27 docs
- Baseline measurement using the new reference (if available)

### If no reference is installable

Document in the diff doc:
- All references searched
- Why none could be installed
- A `principled_residual: source_unavailable` classification with full
  rationale
- Suggested future-work for a manual port if anyone wants to do it later

## Scope constraints

- DO NOT TOUCH render/styles
- Adapter additions and variants.py changes ARE in scope (that's the work)
- Stage commits explicitly with `git add <specific paths>`; commit format
  `feat(fidelity): round 27 linlog -- <terse desc>`
- NO algorithmic edits to dagua/layout/ops/pipelines/linlog.py — that's R28
  if a reference is found

## Tests

If you add a reference adapter:
- `pytest tests/test_eval/ -k "linlog" --tb=short -q` (or wherever competitor
  tests live; create a new test file if needed)

</task>

<completeness_contract>
- Either: (a) reference found AND wired AND diff document with baseline, OR
  (b) principled_residual documentation with thorough search trail.
- The diff doc is mandatory either way.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Don't give up on Phase 1
without exhausting reasonable search paths.
</default_follow_through_policy>
