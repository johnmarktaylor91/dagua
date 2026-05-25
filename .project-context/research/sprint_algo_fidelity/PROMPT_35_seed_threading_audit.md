<task>
R35 end-to-end seed-threading audit.

R28 caught a `del seed` bug in `dagua/eval/competitors/ogdf_competitor.py` that
silently dropped seed for stochastic OGDF engines, making cached positions
identical across "different" seeds.

R33 lgl_graphopt verified OGDF adapters all preserve seed now. But there may
be other adapters with similar drops.

## Your job

Audit EVERY adapter in `dagua/eval/competitors/*.py`. For each:

1. Find the `layout(graph, seed=None)` and `layout_with_variant(graph, seed=None, variant_params)` methods.
2. Trace the `seed` argument from method signature to:
   - reference library call (e.g., `igraph.Graph.layout_lgl(seed=...)`)
   - subprocess invocation (e.g., `ogdf_runner --seed ...`)
   - layout function (e.g., `nx.spring_layout(seed=...)`)
3. Verify NO step drops seed silently. Look for:
   - `del seed`
   - `seed = None`
   - Missing seed kwarg in downstream call
   - Hardcoded `seed=42` overriding caller-provided seed

4. For each issue: fix it. Add regression test asserting seed actually changes output.

## Verification

After fixes, run a quick verification:
```bash
for engine in <stochastic-engines>; do
    python -c "
    from dagua.eval.competitors import get_competitor
    from dagua.eval.graphs import get_test_graphs
    c = get_competitor('$engine')
    g = [t for t in get_test_graphs() if t.name == 'small_world_100'][0].graph
    r1 = c.layout(g, seed=42)
    r2 = c.layout(g, seed=43)
    same = (r1.pos - r2.pos).abs().max().item() < 1e-6
    print('$engine: pos differs across seeds:', not same)
    "
done
```

A stochastic engine that returns IDENTICAL output for different seeds has a
broken seed thread.

## Output
`eval_output/algo_fidelity/round_35/seed_audit/SUMMARY.md` with per-engine status.

Use commit-safe wrapper.
</task>

<completeness_contract>
Audit every adapter. Fix every seed-thread bug. Document clean ones.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
