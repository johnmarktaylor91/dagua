<task>
R34 PORT linlog (Andreas Noack's LinLog algorithm) from public paper to Python.

R27 marked linlog as "source_unavailable" because Noack's LinLogLayout is LGPL
Java with no Python wrapper. User said "leave NOTHING on the table". Port.

## Approach

The LinLog algorithm is described in:
- Noack, "Energy-Based Clustering of Graphs with Nonuniform Degrees" (2003)
- Noack, "Modularity clustering is force-directed layout" (2009)

The algorithm is a force-directed layout with logarithmic attractive force and
power-law repulsion. Pseudocode is in the paper, ~100 lines of Python.

Public source repos to check:
- https://github.com/iVis-at-Bilkent/Linloglayout-lib (LGPL Java)
- https://github.com/Tulip-Dev/tulip/tree/master/plugins/layout/LinLog (LGPL C++)
- Any others via `gh search code "LinLog" language:python` or `language:javascript`

## Your job

### Phase A: Find a Python or readable reference

Search: `gh search repos linloglayout`, `gh search code "linlog" path:python`.
If a Python implementation exists (even partial), use it as the spec.

### Phase B: Port

Implement `dagua/eval/competitors/linlog_competitor.py` with a fresh
`LinLogReference` class. Algorithm:
1. Random init in [0, 1)^2
2. For each iteration:
   - For each pair of nodes, compute repulsive force based on `r^a` (where `a`
     is repuExponent, typically 1.0)
   - For each edge, compute attractive force based on `r^a * log(r)` for LinLog
     mode, or `r^a` for classic
   - Sum forces per node, move proportionally
3. Apply Barnes-Hut for repulsion if N > threshold

Variant parameters (from existing dagua variants):
- attrExponent (1.0 default = LinLog, 2.0 = quadratic)
- repuExponent (0.0 default = log repulsion, 0.5 = power)
- steps (100, 300, 500)

### Phase C: Wire as reference

Register `linlog` competitor pointing to your port.
Update `dagua/eval/variants.py` so `classic_linlog_*` variants pair against
`linlog` as `is_true_original=True`.

### Phase D: Verify

```bash
python scripts/algo_fidelity_live_compare.py classic_linlog linlog --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_34/linlog_port/post
```

Expected: real RMSD numbers for 5 classic_linlog_* variants (currently no
reference comparator at all).

## Implementation

Use commit-safe wrapper.

## Output
`eval_output/algo_fidelity/round_34/linlog_port/SUMMARY.md`.
</task>

<completeness_contract>
Either: (a) working linlog reference producing positions, OR (b) documented
blocker with thorough search trail (but the algorithm is paper-spec'd; this
should be implementable).
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
