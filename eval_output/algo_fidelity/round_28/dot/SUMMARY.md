# Round 28 dot Summary

## Changes

- Updated `dagua/layout/ops/pipelines/dagua_native.py::_dot_lattice_lp` to use
  Graphviz-dot-compatible point-unit spacing constants:
  - `nodesep = 18.0`
  - rank center separation `ranksep = 72.0`
- Stopped deriving `_dot_lattice_lp` spacing from mean node width/height.
- Updated the modified function docstring to document the point-unit spacing
  behavior.

Code delta: `dagua_native.py` changed by 24 insertions and 12 deletions.

## Verification

Baseline source: copied Round 27 bounded benchmark artifacts into
`eval_output/algo_fidelity/round_28/dot/baseline`.

Post-fix command:

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_28/dot/post_fix
```

Post-fix output:

```text
Wrote 2325 rows to eval_output/algo_fidelity/round_28/dot/post_fix/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_28/dot/post_fix/multi_seed_summary.json
graphs: 5
median: 0.006317
p25: 0.000000
p75: 0.007816
p95: 0.037107
worst: mixed_width_labels 0.044430
```

Per-graph medians were unchanged from Round 27:

| graph | baseline | post_fix | delta |
|---|---:|---:|---:|
| `linear_3layer_mlp` | `0.0000000063` | `0.0000000063` | `0.0000000000` |
| `parallel_multiedge_bundle` | `0.0000000000` | `0.0000000000` | `0.0000000000` |
| `nested_shallow_enc_dec` | `0.0078164181` | `0.0078164181` | `0.0000000000` |
| `tl_mlp_3layer` | `0.0063167145` | `0.0063167145` | `0.0000000000` |
| `mixed_width_labels` | `0.0444298722` | `0.0444298722` | `0.0000000000` |

Interpretation: item 1 was applied, but this bounded
`classic_sugiyama` vs `graphviz_dot` comparison did not move. The likely reason
is that the optional `_dot_lattice_lp` polish path is not exercised by this
benchmark path, so the accepted outcome is no regression.

## Residuals

`principled_residual: large_rewrite_required`

The remaining Round 27 dot gaps are explicitly out of Round 28 scope:
network-simplex ranking, mincross fidelity, x-position network simplex,
cluster-aware constraints, flat/self/multiedge dot classification, and
aspect/ratio scaling. These require wholesale algorithm and metadata plumbing,
not a line-local spacing fix.

## Assumptions

- Treated the R3 Sugiyama defaults as the point-unit source of truth:
  `nodesep=18.0` and rank center separation `72.0`.
- Used the Round 27 baseline artifacts as the pre-fix baseline because the code
  change was already applied before running the post-fix comparator.

## Concerns

- The modified path is still an optional native polish candidate, while the
  requested bounded comparison runs `classic_sugiyama`; therefore metric
  immobility is expected and should not be interpreted as evidence that the
  spacing fix is dead code globally.
