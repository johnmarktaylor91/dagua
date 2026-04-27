# Sprint-31 Findings -- Hierarchical DAGs with Skip Edges (Claude)

Branch HEAD `702d7f5` on `feat/bench-and-aesthetics`. Read-only on `dagua/`.
Metric: `dagua.metrics.composite(dagua.metrics.full(...))` with
`node_sizes=[[40,20]]*N`, post-fix `segments_intersect` that counts
collinear-overlap as crossing.

## TL;DR

- **Don't ship a polish-style fix.** The class-gated `skip_corridor_polish`
  prototyped helps `mixed_width_labels` (+8.60), `extreme_mixed_width_transformer`
  (+6.41), `hierarchical_residual_stage` (+3.34), and synthetic `skip_chain_12`
  (+5.63), but does **not** lift `unet_small` (-0.96 fixed, **-1.51 jitter mean,
  jitter-stable negative**) and regresses `synth_wnw_ladder_8` (-2.36) and
  `synth_resnet_4blocks` (+0.15). Wins land via picker margin; regressions get
  absorbed by picker rejection. That is the picker-margin-as-acceptance pattern
  CONTEXT.md guard 4 prohibits.
- **The real bug is upstream in the layered-DAG pipeline gates.**
  `_should_use_native_dummy_nodes` and `_should_apply_brandes_koepf_refine`
  (in `dagua/layout/ops/pipelines/dagua_native_legacy.py` lines 242 and 277)
  both early-out when `max_layer_width <= 1`. Three of the four target graphs
  classify as max_layer_width=1 because longest-path layering puts every chain
  node on its own layer. Skip edges have nowhere to be routed; the gradient
  pipeline collapses x to mean.
- **Recommended action: drop the `max_layer_width <= 1` early-out from both
  gates.** One conditional removed in each. The fix re-enables Sugiyama dummy
  nodes + Brandes-Koepf horizontal coordinate assignment on the chain-with-skip
  family where they have direct theoretical justification, with zero new
  constants, no new polish, and the original sprint-19 invariants in
  `test_native_dummy_nodes_improve_hexagonal_lattice_composite` likely
  restorable.
- **Class predicate (for either path):** connected DAG, longest-path layering
  >= 4 layers, max layer width <= 4, at least one edge spans >= 2 layers.
  This is "skinny pipeline with skip edges" -- residual / encoder-decoder /
  U-Net topologies. Real structural family, not an N+E fingerprint.
- **If Fix A doesn't lift the targets enough, "no further principled fix
  found" is the honest sprint-31 conclusion.** The polish (Fix B) is recorded
  below for posterity, not recommended for merge.

## 1. Per-metric diagnosis

### mixed_width_labels (N=6, E=6)

Default `dagua.layout(g)`:
```
x        : ( 0.00,  218.84)
MHA      : ( 0.00,  375.25)
LayerNorm: ( 0.00,  575.89)
+        : ( 0.00,  769.83)
ReLU     : ( 0.00, 1184.64)
out      : ( 0.00, 1555.22)
```

Every node at x=0. Edge `x -> +` (span 3) is collinear with chain edges
`MHA -> LN -> +`; the post-fix `segments_intersect` correctly counts them as
crossings.

| metric | value |
|---|---:|
| crossing_rate | 0.125 |
| edge_length_cv | 0.496 |
| dag_consistency | 1.000 |
| depth_spearman_rho | 1.000 |
| edge_straightness_mean_deg | 0.0 |
| angular_res_mean_deg | 108.0 |
| **composite** | **77.58** |

Loss to elk (84.52, delta -6.94) is the crossing term. ELK puts skip endpoints
(x and +) on column A (x=12) and chain interior (MHA, LN) on column B (x=32) --
non-collinear.

### unet_small (N=9, E=11)

Default x=0.00 for every node. Layer assignments:
```
input(0) enc1(1) enc2(2) enc3(3) bottleneck(4)
dec3(5)  dec2(6) dec1(7) output(8)
```
Long edges: enc1->dec1 (span 6), enc2->dec2 (span 4), enc3->dec3 (span 2).
`crossing_rate=0.250`. **Composite=70.79**.

ELK (77.04) bends the chain into a U-shape: input(12,12), enc1(12,112),
enc2(32,212), enc3(52,312), bottleneck(72,412), dec3(52,512), dec2(32,612),
dec1(12,712), output(12,812). Encoder steps right, decoder back left. enc_k
and dec_k share x; each skip edge becomes vertical and parallel to the chain.

### extreme_mixed_width_transformer (N=10, E=12)
Layer widths `[1,1,1,3,1,1,1,1]`. Long edge `x -> +` (span 5). Default
x_range=841 (Q/K/V already spread). Loss to graphviz_dot (74.46 vs 77.99) is
the long skip running through Q/K/V's column.

### hierarchical_residual_stage (N=10, E=11)
All chain. x_range=0. Long edges `stem.conv -> stage1.add` (span 3) and
`stage1.add -> stage2.add` (span 3). Loss to dagre (82.29 vs 84.71) from
residuals colliding with chain.

## 2. Pipeline-behavior diagnosis (load-bearing)

`dagua.layout.graph_classify.classify_graph` reports:
```
mixed_width_labels              : num_layers=6,  max_layer_width=1
unet_small                      : num_layers=9,  max_layer_width=1
extreme_mixed_width_transformer : num_layers=8,  max_layer_width=3
hierarchical_residual_stage     : num_layers=10, max_layer_width=1
```

Both gates contain:
```python
if max_layer_width is not None and max_layer_width <= 1:
    return False
```

So both routes that *would* horizontally separate skip edges from the chain
are skipped on chain-shaped layerings. Reproduced directly: setting
`force_pipeline='layered_dag'` explicitly produces x_range=0 on
`mixed_width_labels`, `unet_small`, and `hierarchical_residual_stage` (and 982
on extreme_mwt, but composite still drops to 72.48 -- worse than default's
74.46). The infrastructure exists; the gate is wrong for chain-with-skip.

The original guard makes sense for genuine chains (`deep_chain_20`: composite
97.50 with x_range=0 is optimal). For chain-with-skip, dummies + BK *should*
fire: the long edge passes through "phantom layers" and routing it through
dummies in those layers gives BK the freedom to push the dummies away from
chain x.

## 3. Recommended fix

### Fix A (preferred): refine the gates

```python
def _should_use_native_dummy_nodes(config, structure, edge_index, layer_assignments):
    if not bool(getattr(config, "insert_dummy_nodes", True)):
        return False
    if structure is None or not bool(getattr(structure, "is_directed_acyclic", True)):
        return False
    if int(getattr(structure, "num_components", 1)) != 1:
        return False
    if int(getattr(structure, "num_layers", 0)) <= 1:
        return False
    if (layer_assignments is None
            or int(layer_assignments.shape[0]) < _DUMMY_NODE_MIN_NODES):
        return False
    # REMOVED: max_layer_width <= 1 early-out.
    # Width-1 layerings still benefit from dummy-node insertion when
    # long (span > 1) edges exist; dummies can be placed off-spine via
    # Brandes-Koepf, breaking colinearity with the chain.
    if "dense_dag" in getattr(structure, "topology_tags", ()):
        return False
    return _has_long_layer_edges(edge_index=edge_index,
                                 layer_assignments=layer_assignments)


def _should_apply_brandes_koepf_refine(config, structure, layer_assignments):
    if not bool(getattr(config, "brandes_koepf_refine", True)):
        return False
    # REMOVED: max_layer_width <= 1 short-circuit.
    # BK is meaningful when long edges + dummies exist on a width-1
    # layering: those dummies become candidates for horizontal
    # separation. For pure chains (no long edges), BK is a no-op.
    return True
```

Activation predicate is implicit: DAG, single component, >= 2 layers, at least
one span > 1 edge. Independent of N/E/edge-set/constants. Captures the entire
encoder-decoder / residual / skip-graph family.

### Fix B (fallback, NOT recommended): skip-corridor polish (~80 LOC)

[Polish prototype omitted -- see in-conversation report; recommended NOT
shipped because guard 4 violations.]

## 4. Empirical validation (Fix B polish only -- Fix A not measured)

| graph | N | E | gate | before | after | delta | jitter mean (sigma=0.5, 8 trials) |
|---|---:|---:|---|---:|---:|---:|---:|
| mixed_width_labels | 6 | 6 | True | 77.58 | 86.18 | **+8.60** | +1.13 (std 4.33, min -1.38) |
| unet_small | 9 | 11 | True | 70.79 | 69.82 | -0.96 | **-1.51** (std 1.47, max -0.95) |
| extreme_mixed_width_transformer | 10 | 12 | True | 74.46 | 80.87 | **+6.41** | **+6.15** (std 1.73, min +4.19) |
| hierarchical_residual_stage | 10 | 11 | True | 82.29 | 85.62 | **+3.34** | +1.39 (std 1.74, min -1.89) |
| synth_wnw_ladder_8 | 8 | 10 | True | 73.38 | 71.03 | -2.36 | -- |
| synth_resnet_4blocks | 18 | 21 | True | 85.23 | 85.38 | +0.15 | -- |
| synth_skip_chain_12 | 12 | 13 | True | 74.68 | 80.31 | **+5.63** | -0.48 (std 2.64) |
| random_dag_200 | 383 | 300 | False | 74.40 | 74.40 | 0.00 | -- |
| deep_chain_20 | 22 | 21 | False | 97.50 | 97.50 | 0.00 | -- |
| org_chart_deep | 79 | 78 | False | 92.44 | 92.44 | 0.00 | -- |
| hub_fanout_label_skew | 10 | 13 | False | 93.74 | 93.74 | 0.00 | -- |

### Reading the table
- Gate correctly excludes all four protected wins (delta 0.00).
- On in-class targets: 2 of 4 jitter-positive (mixed_width, extreme_mwt).
  hierarchical_residual_stage's positive jitter mean (+1.39) has a negative
  jitter min (-1.89). **unet_small is jitter-stable negative** (max -0.95).
- On synthetics: skip_chain_12 jitter-erodes (+5.63 fixed -> -0.48 mean),
  wnw_ladder regresses (-2.36), resnet neutral.
- Polish moves nodes by O(pitch_x)~=60. mixed_width's std 4.33 vs mean 1.13
  means the lift IS partially metric artifact.

## 5. Recommended action

**Recommend Fix A: drop the `max_layer_width <= 1` guards in
`_should_use_native_dummy_nodes` and `_should_apply_brandes_koepf_refine`.**

Why:
1. Re-enables existing principled machinery (BK 2002, Sugiyama et al. 1981) on
   a class with direct theoretical justification.
2. One-conditional-removal change in each of two functions. No new constants,
   no new gate, no new `_best_of_polish` candidate.
3. BK *will* push dummy nodes off chain x because that's its objective. Chain
   stays straight; long edges get their own corridor.
4. Doesn't interact with picker margin. Pipeline output improves before any
   polish runs; existing 16 principled candidates operate on the better
   baseline.
5. If gate fix produces under-spread layouts on intentional width-1 chains,
   existing principled candidates pick the better of {improved, baseline}
   via composite score.

**Do NOT ship Fix B (skip-corridor polish):**
1. Helps only 2 of 4 targets jitter-stably; regresses unet_small jitter-stably;
   relies on picker rejection to hide synth_wnw_ladder/synth_resnet
   regressions -- exact pattern CONTEXT.md guards 4-5 prohibit.
2. Adds polish code in `dagua_native.py` after sprint-30 just trimmed it.
3. Mechanism semantically overlaps with what BK is supposed to do; if BK runs
   it should produce better output than this hand-coded corridor allocation.

**If Fix A is impractical, "no further principled fix found" is the correct
sprint-31 outcome.**

## 6. Risks / things to watch with Fix A

- `deep_chain_20` (97.50 baseline, no skip edges): `_has_long_layer_edges`
  returns False (all spans=1), so dummy nodes still don't fire. BK is a no-op
  on width-1 layers. Should pass through unchanged. **Verify after fix lands.**
- Lift on `extreme_mixed_width_transformer` may swing picker toward principled
  BK and away from gradient pipeline's existing Q/K/V spread. Test suite
  already covers it.
- If "long edges" predicate is satisfied on a graph the current pipeline
  handles well (e.g. cyclic feedback that becomes DAG after back-edge
  removal), BK might run on a layout it wasn't meant for. Existing
  `_should_apply_brandes_koepf_refine` already gates on `is_acyclic`; confirm
  continues to require acyclicity.

## Relevant file paths

- `dagua/layout/ops/pipelines/dagua_native_legacy.py` -- lines 242 and 277,
  the two `max_layer_width <= 1` guards to remove (Fix A target).
- `dagua/layout/ops/pipelines/dagua_native.py` -- `_run_native_problem` (line
  298) dispatches to `build_native_layered_dag_pipeline`; `_best_of_polish`
  (line 2558) is where Fix B would integrate.
- `dagua/layout/ops/pipelines/native_layered_dag.py` -- thin shim that calls
  `dagua_native_legacy.build_dagua_pipeline` with `insert_dummy_nodes` and
  `brandes_koepf_refine` set; relies on those gate predicates.
- `dagua/layout/graph_classify.py` -- line 383 computes `max_layer_width`
  from `_analyze_layers`. Reports `max_layer_width=1` for all four targets
  except extreme_mwt.
- `dagua/metrics.py` -- lines 146-200 contain the post-fix `segments_intersect`
  (with collinear-overlap branch) that the diagnosis depends on.
- `dagua/eval/graphs.py` -- definitions of the four target graphs at lines
  758, 862, 978, 1323.
