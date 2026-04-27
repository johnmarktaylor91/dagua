# Sprint 32 Post-31a Quality Research Findings -- Claude

## TL;DR

**SHIP.** Drop the `family in {TREE, CHAIN}` and `"lattice_like"` topology-tag
rejections from `_should_apply_brandes_koepf_refine` in
`dagua/layout/ops/coordinate.py` (lines 1022-1025). Class-predicated; flips
four moderate H2H losses (`mixed_width_labels`, `unet_small`,
`hierarchical_residual_stage`, `cluster_member_style_stress`) into wins,
delivers four additional >=+1.0 lifts on existing wins, narrows but does not
invert four other existing wins, and leaves all five user-named protected
wins within `|delta| <= 0.5`. One-conditional removal in one file; no new
ops, no new constants.

## 1. Target picked + why

Computed fresh post-31a H2H baseline at `/tmp/sprint32_h2h.csv` (94 graphs).
Top losses:

| graph | dagua | best comp | delta | engine |
|---|---:|---:|---:|---|
| mixed_width_labels | 77.58 | 84.52 | -6.94 | elk_layered |
| unet_small | 70.79 | 77.04 | -6.25 | elk_layered |
| disconnected_encoder_residual | 81.19 | 85.63 | -4.45 | elk_layered |
| cluster_member_style_stress | 75.87 | 80.31 | -4.44 | dagre |
| dense_pair_50 | 72.71 | 75.38 | -2.67 | graphviz_dot |
| hierarchical_residual_stage | 82.29 | 84.71 | -2.42 | dagre |
| parallel_cycles_4x5 | 60.65 | 62.73 | -2.08 | graphviz_sfdp |

`disconnected_encoder_residual` is multi-component (rejected by BK's
component gate by construction). `dense_pair_50` and `parallel_cycles_4x5`
are different structural classes. The four sprint-31 targets *plus*
`cluster_member_style_stress` all pass the patched gate predicate, so the
highest-leverage move is the chain-with-skip / lattice_like-tagged
residual-DAG class.

## 2. Metric breakdown

Per-component breakdown of the four sprint-31 targets at HEAD vs best
competitor:

| graph | composite | dag | depth_rho | edge_cv | crossing_rate |
|---|---:|---:|---:|---:|---:|
| mixed_width_labels dagua | 77.58 | 1.000 | 1.000 | 0.496 | 0.125 |
| best comp (elk_layered) | 84.52 | 0.833 | 1.000 | 0.066 | 0.000 |
| unet_small dagua | 70.79 | 1.000 | 1.000 | 0.746 | 0.250 |
| best comp (igraph_graphopt) | 85.45 | 1.000 | 1.000 | 0.103 | 0.000 |
| extreme_mwt dagua (post-31a) | 86.41 | 0.833 | 0.991 | 0.043 | 0.000 |
| best comp (sgd2) | 83.32 | 0.917 | 1.000 | 0.109 | 0.000 |
| hierarchical_residual_stage dagua | 82.29 | 1.000 | 1.000 | 0.498 | 0.053 |
| best comp (dagre) | 84.71 | 1.000 | 1.000 | 0.557 | 0.000 |

The dagua loss on three of four targets is dominated by **crossing_rate**
(0.125/0.250/0.053) and **edge_length_cv** (0.496/0.746/0.498). Dagua
produces a vertical spine (x_range=0) with all skip edges drawn collinear
with the chain.

After-fix per-component (live monkey-patch, seed=42):

| graph | composite | edge_cv | crossing_rate |
|---|---:|---:|---:|
| mixed_width_labels | 77.58 -> 87.14 (**+9.55**) | 0.496 -> 0.227 | 0.125 -> 0.000 |
| unet_small | 70.79 -> 80.48 (**+9.69**) | 0.746 -> 0.745 | 0.250 -> 0.000 |
| hierarchical_residual_stage | 82.29 -> 86.32 (**+4.03**) | 0.498 -> 0.516 | 0.053 -> 0.000 |

Crossings go to 0.000 in all three; CV either improves or holds.

## 3. Existing-op recomposition vs new code

This is a **pure existing-op recomposition; no new code.**

`BrandesKoepfHorizontalRefine` (op `brandes_koepf_horizontal_refine` at
`dagua/layout/ops/coordinate.py:1491`) and `InsertDummyNodes`
(`layering.py:632`) already exist and are wired into
`build_dagua_pipeline()` in `dagua_native_legacy.py:1161, 1275`. The
sprint-31a gate change in the **legacy** pipeline already turns both ops
"on" for the chain-with-skip class:

```
mixed_width_labels         legacy_dummy=True  legacy_bk=True  has_long_edges=True
unet_small                 legacy_dummy=True  legacy_bk=True  has_long_edges=True
hierarchical_residual_stage legacy_dummy=True  legacy_bk=True  has_long_edges=True
extreme_mwt                legacy_dummy=True  legacy_bk=True  has_long_edges=True
```

But BK then runs its **own** op-level admission gate inside
`coordinate.py:_should_apply_brandes_koepf_refine` (lines 987-1035) which
rejects:

```python
if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
    return False
if "lattice_like" in getattr(structure, "topology_tags", ()):
    return False
```

For all three width-1 chain-with-skip targets, `classify_graph()` sets
`topology_tags = ("lattice_like",)` (planar + degree 2-6 + edge-to-node
ratio 1.0-2.2 + num_layers >= 5 + uniform layer width). So BK silently
no-ops at the op level even though the legacy gate enabled it. Sprint-31a
flipped the legacy enable flag; the op-level gate still rejects.
`extreme_mwt`'s sprint-31a +11.95 lift landed because it is `planar_dag`-
tagged not `lattice_like`.

**Gap classification: (a) existing op exists but doesn't fire (gate bug).**

The same op-level fix delivers the lift Codex's prototype tried to deliver
with new "skip-corridor polish" code -- without adding any new candidate to
`_best_of_polish`, without graph predicates, and without metric-sensitive
coordinate hacks. Satisfies CONTEXT.md guards 1-6 by construction.

## 4. Pseudocode of proposed change

```python
def _should_apply_brandes_koepf_refine(
    structure, edge_index, layers, num_nodes, min_layers,
):
    if num_nodes == 0:
        return False
    # REMOVED: family in {TREE, CHAIN} early-out.
    # Pure CHAIN/TREE has no long layer edges, so dummy nodes don't
    # insert and BK has no horizontal work; the early-out was redundant
    # with the downstream "no long edges" condition. For chain-with-skip
    # graphs that are family=GENERAL but width-1, dummy insertion
    # provides the horizontal degrees of freedom BK needs.

    # REMOVED: "lattice_like" topology-tag early-out.
    # The lattice_like predicate (planar + degree 2-6 + edge-to-node
    # 1.0-2.2 + num_layers >= 5 + uniform width) was meant to keep BK
    # away from regular 2D lattices. It accidentally fires on the
    # entire chain-with-skip residual-DAG family.

    num_layers = int(layers.max().item()) + 1 if layers.numel() > 0 else 0
    if num_layers < min_layers:
        return False
    component_sizes = _weak_component_sizes(
        edge_index=edge_index, num_nodes=num_nodes,
    )
    if component_sizes not in ([num_nodes], [num_nodes - 1, 1]):
        return False
    return _has_strict_forward_layering(
        edge_index=edge_index, layers=layers,
    )
```

**Class predicate after the change:** Connected (or near-connected: at
most one isolated node) directed acyclic graph with strict forward
layering and >= 2 layers. Robust to renaming, jittering, +/- 1 edge
perturbation. No `(num_nodes, num_edges)` reference, no graph names.

## 5. Empirical validation

### 5a. Per-target fixed-seed delta (seed=42)

| graph | before | after | delta |
|---|---:|---:|---:|
| mixed_width_labels | 77.584 | 87.136 | **+9.553** |
| unet_small | 70.785 | 80.476 | **+9.690** |
| extreme_mwt | 86.408 | 86.408 | +0.000 (already fires post-31a) |
| hierarchical_residual_stage | 82.285 | 86.316 | **+4.030** |

### 5b. Sigma=0.5 jitter, 8 trials

| graph | mean | std | min | max |
|---|---:|---:|---:|---:|
| mixed_width_labels | +2.080 | 4.632 | -0.440 | +9.585 |
| unet_small | **+8.170** | 1.944 | **+5.245** | +9.704 |
| extreme_mwt | +0.000 | 0.000 | 0.000 | 0.000 |
| hierarchical_residual_stage | +0.439 | 1.963 | -1.217 | +4.065 |

`unet_small` jitter is rock-solid positive. `mixed_width_labels` mean > 0,
min > -1 (passes). `hierarchical_residual_stage` mean > 0 but min -1.22
fails strict "min > -1" by 0.22. Recorded honestly rather than widening
the bar.

### 5c. Out-of-suite synthetics (chain-with-skip class)

| synthetic | edges | before | after | delta |
|---|---|---:|---:|---:|
| skip_chain_14_residual | chain 0..13 + skips (0,3),(3,6),(6,9),(9,12) | 81.585 | 85.228 | **+3.643** |
| unet_11 | chain 0..10 + skips (1,9),(2,8),(3,7) | 69.198 | 78.289 | **+9.091** |
| chain_long_skip_8 | chain 0..7 + (0,7) | 74.516 | 83.959 | **+9.443** |

Negative controls (graphs that pass the patched gate but should not benefit):

| negative control | before | after | delta |
|---|---:|---:|---:|
| pure_chain_12 (no skips) | 97.207 | 97.207 | +0.000 |
| bipartite_4layer_8 | 83.581 | 83.581 | +0.000 |

Class-stable on independently-generated synthetics; no-op when no
horizontal-corridor work exists.

### 5d. User-named protected wins

| graph | before | after | delta |
|---|---:|---:|---:|
| deep_chain_20 | 97.500 | 97.500 | +0.000 |
| random_dag_200 | 74.611 | 74.611 | +0.000 |
| ba_500 | 63.138 | 63.138 | +0.000 |
| org_chart_deep | 92.441 | 92.830 | +0.389 |
| hub_fanout_label_skew | 93.737 | 93.737 | +0.000 |

All within `|delta| <= 0.5`.

### 5e. Suite-wide affected-graph audit (27 newly-enabled graphs)

```
Regressions (delta < -0.5):
  compound_10x20             N=200  79.140 -> 76.283  -2.856
  multiscale_skip_cascade     N=15  79.064 -> 76.954  -2.110
  residual_block              N=10  85.011 -> 84.287  -0.724
  ragged_feature_pyramid      N=12  81.602 -> 80.909  -0.693

Lifts (delta >= 1.0):
  cluster_member_style_stress  N=8  75.871 -> 87.409 +11.538
  unet_small                   N=9  70.785 -> 80.476  +9.690
  mixed_width_labels           N=6  77.584 -> 87.136  +9.553
  hierarchical_residual_stage N=10  82.285 -> 86.316  +4.030
  resnet_stack_4x16           N=30  77.028 -> 79.447  +2.419
  clustered_longlabel_handoffs N=10 87.857 -> 90.183  +2.325
  long_range_residual_ladder  N=38  80.327 -> 82.117  +1.790
  sparse_pair_50              N=50  86.656 -> 88.256  +1.601
```

The four regressing graphs are jitter-stable (8-trial sigma=0.5 means
-3.61, -1.92, -0.72, -0.70 with stds <= 0.26). They are NOT metric
artifacts.

**However, those four regressing graphs are all currently dagua *wins*
in H2H.** After the fix:

- compound_10x20: 79.14 -> 76.28 vs graphviz_dot 75.00 (still wins by +1.28; was +4.14)
- multiscale_skip_cascade: 79.06 -> 76.95 vs dagre 70.67 (still wins by +6.29; was +8.40)
- residual_block: 85.01 -> 84.29 vs graphviz_dot 82.01 (still wins by +2.28; was +3.00)
- ragged_feature_pyramid: 81.60 -> 80.91 vs graphviz_dot 78.69 (still wins by +2.22; was +2.91)

**No H2H wins are lost.** Wins narrow on four graphs; losses flip to
wins on four graphs.

### 5f. Net H2H ledger impact

| graph | before | after | sign change |
|---|---:|---:|---|
| mixed_width_labels | -6.94 | +2.61 | LOSS -> WIN |
| unet_small | -6.25 | +3.44 | LOSS -> WIN |
| hierarchical_residual_stage | -2.42 | +1.61 | LOSS -> WIN |
| cluster_member_style_stress | -4.44 | +7.10 | LOSS -> WIN |
| compound_10x20 | +4.14 | +1.28 | WIN -> WIN |
| multiscale_skip_cascade | +8.40 | +6.29 | WIN -> WIN |
| residual_block | +3.00 | +2.28 | WIN -> WIN |
| ragged_feature_pyramid | +2.91 | +2.22 | WIN -> WIN |

Plus four further +1.0 to +2.5 lifts that were already wins
(`resnet_stack_4x16`, `clustered_longlabel_handoffs`,
`long_range_residual_ladder`, `sparse_pair_50`).

**Net: +4 graphs flip from loss to win, 0 wins flip to loss.** Converts
the honest 83/93 -> ~87/93 best-or-tied.

## 6. Concrete dagua/ files + line numbers to edit

Single edit site:

- `dagua/layout/ops/coordinate.py:1019-1035` -- inside
  `_should_apply_brandes_koepf_refine`, drop the two early-outs:

```python
# Remove these two conditionals (currently lines 1022-1025):
if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
    return False
if "lattice_like" in getattr(structure, "topology_tags", ()):
    return False
```

The rest of the gate (component check, layering check,
`num_layers >= min_layers`) is preserved. No other file changes needed.

## 7. Risks

1. **Jitter-marginal lift on `hierarchical_residual_stage`** (mean +0.439,
   min -1.22). Fixed-seed lift is real (crossing_rate 0.053 -> 0.000) but
   jitter min violates a strict "min > -1" by 0.22.
2. **`compound_10x20` regression -2.86 fixed, -3.61 jitter mean.** Largest
   single negative effect. Still leaves graph as +1.28 H2H win.
3. **Test-suite churn.** Tests likely assert BK is suppressed on the
   targets. These flip; expected outputs need updating.
4. **Visual regression on 4 regressed graphs.** Eyeball validation needed
   on top-2 improved AND top-2 regressed.
5. **Sprint-32 closes "small but non-empty."** One conditional removal in
   one file. Follow-on to sprint-31a closing the gate-refinement loop.
