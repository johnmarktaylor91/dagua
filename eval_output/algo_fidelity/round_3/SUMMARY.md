# Round 3 Summary

## Outcome: COMMITTED

Round 3 confirmed that live `classic_sugiyama` vs cached `graphviz_dot` is
deterministic on the current node-size context: two baseline runs produced
identical `live_rmsd.csv` files. The remaining dot-family divergence was not
caused by layer assignment or in-layer ordering on the diagnostic graphs. The
high-signal coordinate lever was a default spacing mismatch: direct Sugiyama
pipeline calls were using unit `rank_sep` and `node_sep`, while graphviz dot's
cached geometry is in point units.

The fix changes only the direct classic Sugiyama pipeline defaults to
dot-compatible point spacing: a 72 pt rank center distance and an 18 pt
horizontal bounding-box gap. This preserves explicit caller/config overrides
and leaves the Brandes-Koepf implementation intact.

| Graph / metric | Before | After | Delta |
|---|---:|---:|---:|
| dot family median | 0.341942 | 0.019116 | -0.322826 |
| mixed_width_labels | 0.404615 | 0.016176 | -0.388439 |
| shape_and_routing_matrix | 0.456349 | 0.019158 | -0.437190 |
| small_label_storm | 0.485187 | 0.028078 | -0.457108 |
| linear_3layer_mlp | 0.000000 | 0.000000 | +0.000000 |
| parallel_multiedge_bundle | 0.000000 | 0.000000 | +0.000000 |
| nested_shallow_enc_dec | 0.007816 | 0.007816 | +0.000000 |

Recommended Round 4 family: `fdp`, because Round 1 identified it as the next
worst graphviz family and its errors were uniformly above the 0.15 floor.
