<task>
R33 REFERENCE AUDIT — NeuLay + sgd2 + any other potentially-missing reference adapters.

R31 infra_recovery codex flagged: "Upstream NeuLay and SGD2 multi references are unavailable in this environment." If true, those engines will NEVER produce paired data regardless of dagua-side improvements.

## Your job

1. Inspect `dagua/eval/competitors/` for all reference adapters:
   - `neulay_competitor.py` (or similar)
   - `sgd2_competitor.py` / `sgd2_multi_competitor.py`
   - Any other engine with reference issues

2. For each, verify the reference can actually be imported and run:
   ```bash
   python -c "import <reference_lib>; print('ok')"
   ```

3. If a reference is missing/broken:
   - Try to install it (pip install)
   - If not pip-installable, document in a TODO and skip
   - If a Python wrapper would suffice, write one

4. Re-run a small focal test to confirm reference produces actual positions:
   ```bash
   python -c "
   from dagua.eval.competitors import get_competitor
   c = get_competitor('neulay')
   from dagua.eval.graphs import get_test_graphs
   tg = [t for t in get_test_graphs() if t.name == 'linear_3layer_mlp'][0]
   r = c.layout(tg.graph, seed=42)
   print(r)
   "
   ```

## Output

`eval_output/algo_fidelity/round_33/reference_audit/SUMMARY.md` documenting:
- Which references work
- Which don't and why
- Any installation/wiring fixes made
- Per-engine recommendation for next steps

## Scope

- DO TOUCH: dagua/eval/competitors/*.py if you find fixable bugs (e.g., import path wrong, adapter not registered)
- Use commit-safe wrapper for commits.
</task>

<completeness_contract>
Audit + document. Fix any that are cheap. Commit working changes.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
