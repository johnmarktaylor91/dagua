## 1. Executive summary

- CONFIRMED: Graphviz SFDP 7.0.5 ignores the benchmark adapter's `theta` and `maxiter` attributes. Thus `classic_sfdp_theta04`, `classic_sfdp_theta08`, and `classic_sfdp_steps200` compare dagua parameter changes against Graphviz default output, not matching originals.
- CONFIRMED: The broad "reference overlap removal distorts positions" H1 is not the main bucket cause. Default Graphviz output equals `overlap=prism0`/`prism1` on probes, and source shows `prism0` has zero overlap iterations.
- HYPOTHESIS: disconnected targets diverge because Graphviz reuses one mutable `spring_electrical_control` across components and then uses `packSubgraphs`, while dagua reruns independent component pipelines and imports neato's packer.
- CONFIRMED: the r74 `p_neg2` clamp is active and makes dagua `p=-2` identical to default on probes; Graphviz also collapses `repulsiveforce=-2` to default. It is not the cause of remaining `p_neg2` divergences.
- CONFIRMED: dagua's Graphviz permutation port uses raw modulo where Graphviz 7.0.5 now uses rejection-sampled `gv_random`; however, no rejection occurred for tested bounds up to 10000 from seed 1, so this is not a cheap high-impact fix for these small/medium graphs.
- I could not explain 58 connected default-like targets; they need deeper force/coarsening parity work.

## 2. Findings ranked by expected combo-count impact

### 1. CONFIRMED: `theta` and `maxiter` original-side params are no-ops in Graphviz SFDP

Expected impact: 47/126 targets by direct variant count: `classic_sfdp_steps200` 18, `classic_sfdp_theta04` 14, `classic_sfdp_theta08` 15.

Code evidence:

- Adapter/variant side forwards these attributes: `dagua/eval/variants.py:1594-1610` maps theta variants to original params `{"maxiter": 500, "theta": 0.4/0.8, "repulsiveforce": -1.0}`; `dagua/eval/variants.py:1627-1632` maps `steps200` to original param `{"maxiter": 200, "theta": 0.6, "repulsiveforce": -1.0}`.
- Graphviz adapter blindly emits variant params as `-G` graph attrs in `dagua/eval/competitors/graphviz_competitor.py:503-536`.
- Graphviz SFDP source only reads `K`, `repulsiveforce`, `levels`, `smoothing`, `quadtree`, `beautify`, `overlap_shrink`, `rotation`, and `label_scheme` in `sfdpinit.c:200-224`; no `maxiter` or `theta` read exists.
- Graphviz Barnes-Hut theta is a file-scope constant: `_references/graphviz/lib/sfdpgen/spring_electrical.c:41-43` defines `static const double bh = 0.6`.
- Graphviz maxiter default is set once in `_references/graphviz/lib/sfdpgen/spring_electrical.c:51-65`; iteration loops consume `ctrl->maxiter` at `spring_electrical.c:249` and stop at `spring_electrical.c:369`, but no SFDP attr updates it.
- Dagua actually uses these params: `dagua/layout/ops/pipelines/sfdp.py:857-930` passes `steps` and `theta` into Graphviz-fidelity refine/prolong ops; `layout_sfdp_pipeline` exposes them at `dagua/layout/ops/pipelines/sfdp.py:1038-1049`.

Command evidence:

```text
$ python3 /tmp/r75_sfdp_theta_maxiter_probe.py
{'maxiter': 500, 'theta': 0.6, 'repulsiveforce': -1.0} 0.03302329108211094
{'maxiter': 200, 'theta': 0.6, 'repulsiveforce': -1.0} 0.03302329108211094
rms_to_base 4.176623870052053e-16
{'maxiter': 500, 'theta': 0.4, 'repulsiveforce': -1.0} 0.03302329108211094
rms_to_base 4.176623870052053e-16
{'maxiter': 500, 'theta': 0.8, 'repulsiveforce': -1.0} 0.03302329108211094
rms_to_base 4.176623870052053e-16
{'maxiter': 500, 'theta': 0.6, 'repulsiveforce': -2.0} 0.03302329108211094
rms_to_base 4.176623870052053e-16
```

Cheapest decisive follow-up: run the definitive scorer only for those 47 with reference labels collapsed to `graphviz_sfdp__for__classic_sfdp_default` and/or remove `theta`/`steps200` from true-original status. Runtime should be minutes because positions already exist for default references.

Fix sketch:

- Comparison-side fix: mark `classic_sfdp_theta04`, `classic_sfdp_theta08`, and `classic_sfdp_steps200` as non-true-original/proxy variants, or stop comparing them to Graphviz as if those attrs exist.
- Dagua-side alternative, if sprint lead wants only canonical Graphviz: force `theta=0.6` and `steps=500` for Graphviz-fidelity variants. This would erase intended dagua parameter variants, so it needs sign-off.

Risk:

- Low risk to bit-exact combos if handled in variant metadata/scoring only.
- High semantic risk if changing dagua behavior, because it intentionally exposes `steps`/`theta` as dagua params.

### 2. HYPOTHESIS: disconnected component state and packing mismatch

Expected impact: up to 43/126 disconnected targets. These include every target for `disconnected_encoder_residual`, `disconnected_label_cycle_collage`, `kitchen_sink_platform_graph`, `multi_component_80`, `parallel_cycles_4x5`, `random_dag_200`, plus 5/6 for `random_dag_50` and 2 for `random_bipartite_60`.

Code evidence:

- Graphviz creates connected components and reuses the same `spring_electrical_control ctrl` through all subgraphs: `_references/graphviz/lib/sfdpgen/sfdpinit.c:238-240` initializes/tunes one control, `sfdpinit.c:268-287` loops `sfdpLayout(sg, &ctrl, pad)` and then calls `packSubgraphs`.
- `ctrl->K` is mutated on the first component when negative: `_references/graphviz/lib/sfdpgen/spring_electrical.c:284-286` and `spring_electrical.c:571-573`.
- `ctrl->random_start`, `ctrl->K`, `ctrl->adaptive_cooling`, and `ctrl->step` are mutated during multilevel prolongation: `_references/graphviz/lib/sfdpgen/spring_electrical.c:1173-1179`.
- Dagua splits components before building the normal pipeline: `dagua/layout/ops/pipelines/sfdp.py:1125-1140`.
- Dagua lays each component via a fresh recursive `layout_sfdp_pipeline` call: `dagua/layout/ops/pipelines/sfdp.py:1004-1025`. The docstring/comment says the same seed is reused per component because Graphviz resets `srand`, but it does not model Graphviz's shared mutable `ctrl->K` and post-component state.
- Dagua imports neato component helpers and packer: `dagua/layout/ops/pipelines/sfdp.py:15-19`; the packer itself is in `dagua/layout/ops/pipelines/neato.py:569-657`.

Target evidence:

```text
disconnected 43 stressbetter 24 crossfail 32 medianrel -0.011786864407584714 max 67.97274386574723 min -0.9997786813923855
graphs [('disconnected_encoder_residual', 6), ('disconnected_label_cycle_collage', 6), ('kitchen_sink_platform_graph', 6), ('multi_component_80', 6), ('parallel_cycles_4x5', 6), ('random_dag_200', 6), ('random_dag_50', 5), ('random_bipartite_60', 2)]
```

Probe evidence, same competitor wrappers as benchmark:

```text
PAIR classic_sfdp_default parallel_cycles_4x5 {"D_cross": 0, "D_stress": 0.11858536283536335, "R_cross": 0, "R_stress": 0.11175323042361876, "proc": 1.0300278736348902}
PAIR classic_sfdp_default disconnected_label_cycle_collage {"D_cross": 0, "D_stress": 0.23053906338367353, "R_cross": 0, "R_stress": 0.20307620897865572, "proc": 0.9428837147885281}
```

Cheapest decisive experiment: implement a scratch monkeypatch or temporary branch that threads one shared Graphviz-control state through `_layout_graphviz_sfdp_components` and disables final normalization until after `packSubgraphs`-like packing, then rerun 5 seeds for `parallel_cycles_4x5`, `disconnected_label_cycle_collage`, and `multi_component_80`. Estimated runtime: 15-25 minutes if done as a local patch plus `scripts/run_benchmark.py --variants --graphs ... --engines classic_sfdp_default,graphviz_sfdp__for__classic_sfdp_default --seeds 42 43 44 45 46`.

Fix sketch:

- Port Graphviz's component loop more literally for SFDP fidelity mode: one `SFDPGraphvizControl` object for all components, preserving `K`, `random_start`, `adaptive_cooling`, and `step` transitions.
- Either port `packSubgraphs` more directly or verify the imported neato packer against Graphviz `packSubgraphs` on component bboxes.

Risk:

- Medium. Prior blanket component fixes broke exact combos in other algorithms. Restrict to `layout_sfdp_pipeline(..., fidelity_mode="graphviz")` and disconnected graphs only.
- Existing connected bit-exact/equivalent SFDP combos should be untouched if the branch remains behind `len(components) > 1`.

### 3. CONFIRMED: broad H1 overlap-removal theory is killed for current default reference

Expected impact: H1-style adapter correction (`overlap=false`) should not be applied as a blanket fix. It would change the reference away from Graphviz default and can make disconnected output different for the wrong reason.

Code evidence:

- Adapter extracts final JSON positions after running `dot -Tjson -Ksfdp`: `dagua/eval/competitors/graphviz_competitor.py:381-444`.
- Graphviz default calls `graphAdjustMode(g, &am, "prism0")` when built with GTS: `_references/graphviz/lib/sfdpgen/sfdpinit.c:241-245`.
- If mode is prism, SFDP sets `ctrl.overlap = am.value` and `ctrl.initial_scaling = am.scaling`: `sfdpinit.c:250-253`.
- The overlap remover returns immediately when `ntry == 0`: `_references/graphviz/lib/neatogen/overlap.c:517-528`. Thus default `prism0` has no iterative overlap smoothing; only initial scaling can run, which normalized stress fits away.
- Graphviz still does rotation and final postprocess after layout: `_references/graphviz/lib/sfdpgen/spring_electrical.c:1188-1200`, `_references/graphviz/lib/sfdpgen/sfdpinit.c:295`.

Command evidence:

```text
OVERLAP asymmetric_hourglass_hub {"default_stress": 0.03302329108211094, "false_stress": 0.03302495002234454, "prism0_stress": 0.03302329108211094, "prism1_stress": 0.03302329108211094, "rms_default_false": 1.9050380569127023e-05, "rms_default_prism0": 4.176623870052053e-16, "rms_default_prism1": 4.176623870052053e-16}
OVERLAP parallel_cycles_4x5 {"default_stress": 0.11175323042361876, "false_stress": 0.09635783856805144, "prism0_stress": 0.11175323042361876, "prism1_stress": 0.11175323042361876, "rms_default_false": 1.0292680692512257, "rms_default_prism0": 3.0431420878697893e-16, "rms_default_prism1": 3.0431420878697893e-16}
OVERLAP disconnected_label_cycle_collage {"default_stress": 0.20307620897865572, "false_stress": 0.1881198015929672, "prism0_stress": 0.20307620897865572, "prism1_stress": 0.20307620897865572, "rms_default_false": 0.3905941611605939, "rms_default_prism0": 2.1228933251263786e-16, "rms_default_prism1": 2.1228933251263786e-16}
OVERLAP parallel_multiedge_bundle {"default_stress": 2.571449000600326e-06, "false_stress": 2.53225500107902e-06, "prism0_stress": 2.571449000600326e-06, "prism1_stress": 2.571449000600326e-06, "rms_default_false": 2.5007083506494918e-05, "rms_default_prism0": 2.0771095305691559e-16, "rms_default_prism1": 2.0771095305691559e-16}
```

Fix sketch:

- Do not set `overlap=false` in the adapter as a bucket-wide fix.
- If the sprint lead wants "spring-electrical pre-postprocess" as the reference target, that is a reference-definition change, not a fidelity fix.

Risk:

- High if misapplied. `overlap=false` changes disconnected outputs and could invalidate already-equivalent combos against actual Graphviz defaults.

### 4. CONFIRMED but low expected impact: RNG bounded integer mismatch

Expected impact: probably near zero for these targets unless a graph/coarsening level is large enough to hit a rejection in `gv_random`. It is still a real source mismatch.

Code evidence:

- Graphviz 7.0.5 `gv_permutation` calls `gv_random(i + 1)`: `_references/graphviz/lib/util/random.c:15-30`.
- `gv_random` uses rejection sampling through `random_small`: `_references/graphviz/lib/util/random.c:35-58` and `random.c:85-95`.
- Graphviz SFDP coarsening uses `gv_permutation(m)`: `_references/graphviz/lib/sfdpgen/Multilevel.c:102`.
- Dagua Graphviz-fidelity hierarchy seeds coarsening with `GraphvizRandom(seed=1)`: `dagua/layout/ops/pipelines/sfdp.py:348-380`.
- Dagua `GraphvizRandom.random` implements rejection sampling: `dagua/layout/ops/sfdp.py:200-228`.
- Dagua `GraphvizRandom.permutation` intentionally uses raw modulo: `dagua/layout/ops/sfdp.py:247-253`. That comment is inconsistent with Graphviz 7.0.5 source.

Command evidence:

```text
$ python3 /tmp/r75_sfdp_permutation_probe.py
3 True rejects 0 firstdiff None
4 True rejects 0 firstdiff None
7 True rejects 0 firstdiff None
14 True rejects 0 firstdiff None
20 True rejects 0 firstdiff None
80 True rejects 0 firstdiff None
97 True rejects 0 firstdiff None
100 True rejects 0 firstdiff None
200 True rejects 0 firstdiff None
383 True rejects 0 firstdiff None
1000 True rejects 0 firstdiff None
10000 True rejects 0 firstdiff None
```

Fix sketch:

- Change `GraphvizRandom.permutation` to call `self.random(index + 1)`.
- Add a targeted regression comparing the first permutation for seed 1 against a tiny C or Python port of Graphviz `gv_random`.

Risk:

- Low for current target sizes based on the no-rejection probe, but nonzero for any already bit-exact very-large SFDP combos if a rejection occurs. Gate with a before/after scan of bit-exact SFDP combos.

### 5. CONFIRMED: r74 `p_neg2` clamp is active and not causing the bucket

Expected impact: do not revert commit 6f8cff5. `classic_sfdp_p_neg2` divergences are default-SFDP divergences under another variant label, except where disconnected/component handling or other force parity issues apply.

Code evidence:

- Variant maps dagua `repulsive_exponent=-2.0` to Graphviz `repulsiveforce=-2.0`: `dagua/eval/variants.py:1615-1622`.
- Dagua clamps values below default `-1.0` in Graphviz-fidelity mode: `dagua/layout/ops/pipelines/sfdp.py:411-433`; it is applied in `build_sfdp_pipeline` at `sfdp.py:911-916`.
- Graphviz parses `repulsiveforce` with minimum `0.0`, then negates: `_references/graphviz/lib/sfdpgen/sfdpinit.c:211-213`; if effective `p >= 0`, spring code falls back to `p=-1`: `_references/graphviz/lib/sfdpgen/spring_electrical.c:287`, `spring_electrical.c:574`.

Command evidence:

```text
PAIR classic_sfdp_default asymmetric_hourglass_hub {"D_stress": 0.03627356962234667, "R_stress": 0.03302329108211094, "proc": 0.5455870468999658}
PAIR classic_sfdp_p_neg2 asymmetric_hourglass_hub {"D_stress": 0.03627356962234667, "R_stress": 0.03302329108211094, "proc": 0.5455870468999658}
...
{'maxiter': 500, 'theta': 0.6, 'repulsiveforce': -2.0} 0.03302329108211094
rms_to_base 4.176623870052053e-16
```

Risk:

- Reverting the clamp would make dagua intentionally differ from Graphviz for negative `repulsiveforce` inputs and likely increase divergence.

## 3. Single-fix conversion estimate

- A comparison-side metadata/scoring fix for `theta04`, `theta08`, and `steps200` can remove or reclassify 47 targets. It cannot make them bit-exact because the reference invocation never exercised those parameters.
- A disconnected-component fidelity fix could plausibly move up to 43 targets, but overlaps with the 47 parameter targets. The non-overlap maximum is 19 disconnected default-like targets (`default`, `graphviz_fidelity`, `p_neg2`).
- H1-style `overlap=false` should convert zero targets under the current "Graphviz default" reference definition and risks making the comparison less canonical.
- The RNG permutation fix is correctness debt, not an expected mass converter for this bucket.

## 4. Target combos I could not explain

I could not explain these 58 connected, default-like targets after removing the confirmed no-op-parameter variants and the disconnected component hypothesis:

```text
binary_tree::classic_sfdp_p_neg2
asymmetric_hourglass_hub::classic_sfdp_p_neg2
asymmetric_hourglass_hub::classic_sfdp_graphviz_fidelity
asymmetric_hourglass_hub::classic_sfdp_default
cluster_member_style_stress::classic_sfdp_p_neg2
braided_feedback_tails::classic_sfdp_p_neg2
broken_symmetry_residual_pair::classic_sfdp_p_neg2
clustered_longlabel_handoffs::classic_sfdp_p_neg2
deep_chain_20::classic_sfdp_p_neg2
densenet_block::classic_sfdp_p_neg2
extreme_mixed_width_transformer::classic_sfdp_p_neg2
grid_5x5::classic_sfdp_p_neg2
grid_rect_6x8::classic_sfdp_p_neg2
hexagonal_lattice_42::classic_sfdp_default
hexagonal_lattice_42::classic_sfdp_graphviz_fidelity
hexagonal_lattice_42::classic_sfdp_p_neg2
hierarchical_residual_stage::classic_sfdp_p_neg2
hub_skip_superfan::classic_sfdp_p_neg2
interleaved_cluster_crosstalk::classic_sfdp_p_neg2
kitchen_sink_hybrid_net::classic_sfdp_p_neg2
linear_3layer_mlp::classic_sfdp_p_neg2
long_range_residual_ladder::classic_sfdp_p_neg2
mixed_width_labels::classic_sfdp_p_neg2
long_skip_only_24::classic_sfdp_p_neg2
multiscale_skip_cascade::classic_sfdp_p_neg2
nested_cluster_label_stack::classic_sfdp_p_neg2
nested_shallow_enc_dec::classic_sfdp_p_neg2
outerplanar_dag_20::classic_sfdp_p_neg2
dense_pair_50::classic_sfdp_p_neg2
ragged_feature_pyramid::classic_sfdp_p_neg2
planar_60::classic_sfdp_p_neg2
planar_60::classic_sfdp_graphviz_fidelity
real_karate_34::classic_sfdp_default
real_karate_34::classic_sfdp_graphviz_fidelity
real_karate_34::classic_sfdp_p_neg2
recurrent_feedback_cell::classic_sfdp_p_neg2
real_lesmis_77::classic_sfdp_default
real_lesmis_77::classic_sfdp_graphviz_fidelity
real_lesmis_77::classic_sfdp_p_neg2
residual_block::classic_sfdp_p_neg2
shape_and_routing_matrix::classic_sfdp_p_neg2
small_label_storm::classic_sfdp_p_neg2
resnet_stack_4x16::classic_sfdp_p_neg2
tl_cnn_small::classic_sfdp_p_neg2
tl_mlp_3layer::classic_sfdp_p_neg2
sparse_pair_50::classic_sfdp_p_neg2
tl_resnet_2block::classic_sfdp_p_neg2
sierpinski_42::classic_sfdp_p_neg2
tl_transformer_1layer::classic_sfdp_p_neg2
transformer_full_4h_2l::classic_sfdp_p_neg2
transformer_layer::classic_sfdp_p_neg2
weighted_chain_20::classic_sfdp_default
weighted_chain_20::classic_sfdp_p_neg2
weighted_chain_20::classic_sfdp_graphviz_fidelity
triangular_lattice_36::classic_sfdp_p_neg2
weighted_karate_34::classic_sfdp_default
weighted_karate_34::classic_sfdp_graphviz_fidelity
weighted_karate_34::classic_sfdp_p_neg2
```

Likely next places to inspect for those: exact Graphviz sparse matrix row order and duplicate-edge handling in `makeMatrix`, the `average_edge_length` quirk already ported at `dagua/layout/ops/sfdp.py:808-826`, and small-graph sequential update parity against `_references/graphviz/lib/sfdpgen/spring_electrical.c:599-660`.
