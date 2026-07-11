# W-C p_neg2 reference-oracle report

## Disposition

`classic_sfdp_p_neg2` is scoreable. Graphviz 7.0.5 can express the same
internal repulsive exponent as Dagua's `repulsive_exponent=-2.0`, but the
correct graph attribute is:

```text
repulsiveforce=2.0
```

The old `repulsiveforce=-2.0` was not p=-2. Graphviz clamps negative
`repulsiveforce` to `0.0`, negates that to internal `p=-0.0`, and the solver
then falls back to `p=-1`.

## Source evidence

Pinned Graphviz 7.0.5 source:

- `lib/sfdpgen/sfdpinit.c:239`:
  `ctrl->p = -1.0*late_double(g, agfindgraphattr(g, "repulsiveforce"), -AUTOP, 0.0);`
- `lib/common/utils.c:82-95`: `late_double(..., minimum)` returns `minimum`
  when the parsed value is below that minimum. For `repulsiveforce`, the minimum
  is `0.0`.
- `lib/sfdpgen/spring_electrical.c:519`, `688`, `864`, `1167`, `1352`:
  nonnegative internal `p` is reset to `-1`.
- `lib/sfdpgen/spring_electrical.c:719-725` and `924-950`: the repulsive force
  uses `KP * delta / pow(dist, 1 - p)`. With `p=-2`, this is the p=-2
  physical exponent.

Therefore:

- `repulsiveforce=-2` -> `late_double(..., minimum=0)` returns `0` ->
  internal `p=-0` -> solver fallback `p=-1`.
- `repulsiveforce=2` -> internal `p=-2` -> force denominator `dist^3`.

## Empirical evidence

Local binary:

```text
sfdp - graphviz version 7.0.5 (20221231.0122)
```

Seeded `sfdp -Tplain` probe on a fixed 16-node graph with
`start=100, maxiter=500, theta=0.6`:

| repulsiveforce | sha256 prefix | spread | relation to default |
|---:|---|---|---|
| default | `80f7d62d3377` | width 3.030211, height 1.249511, mean_r 0.995156 | default |
| -2 | `80f7d62d3377` | width 3.030211, height 1.249511, mean_r 0.995156 | byte-identical |
| -1 | `80f7d62d3377` | width 3.030211, height 1.249511, mean_r 0.995156 | byte-identical |
| 0 | `80f7d62d3377` | width 3.030211, height 1.249511, mean_r 0.995156 | byte-identical |
| 1 | `80f7d62d3377` | width 3.030211, height 1.249511, mean_r 0.995156 | byte-identical |
| 2 | `bcc4aead19dd` | width 2.239011, height 0.981911, mean_r 0.740322 | distinct |
| 3 | `0788d32551b` | width 2.441411, height 1.195711, mean_r 0.777583 | distinct |

Verbose Graphviz confirmed the internal exponents:

```text
default: repulsive and attractive exponents: -1.000 1.000
-2:      repulsive and attractive exponents: -0.000 1.000
2:       repulsive and attractive exponents: -2.000 1.000
```

## Dagua semantics check

`dagua/layout/ops/pipelines/sfdp.py` implements the same force law:

- `_sfdp_force_scales(..., repulsive_exponent=-2.0)` computes
  `repulsive_scale = K ** 3` and the Graphviz-style attractive scale exponent
  `(2 - p) / 3 = 4/3`.
- `_SFDPGraphvizSequentialStep` divides the repulsive delta vector by
  `distance ** (1 - repulsive_exponent)`. For `p=-2`, that is `distance ** 3`,
  matching Graphviz's p=-2 force expression.

Important caveat: the current Graphviz-fidelity pipeline also has
`_graphviz_repulsive_exponent`, whose docstring was written around the old
negative-attribute oracle and collapses `repulsive_exponent < -1` to `-1`.
This task explicitly prohibited touching `sfdp.py`, so I left that code alone.
The corrected reference rescore below is therefore an honest oracle correction,
not an algorithm-code repair.

## Code change

Changed `dagua/eval/variants.py`:

```diff
- {"maxiter": 500, "theta": 0.6, "repulsiveforce": -2.0}
+ {"maxiter": 500, "theta": 0.6, "repulsiveforce": 2.0}
```

Added a focused registry regression test in `tests/test_variant_registry.py`.

## Reference bench

Command shape:

```text
python scripts/run_benchmark.py \
  --seeds 35 --seed-start 100 --seed-refs graphviz_sfdp \
  --output-dir eval_output/benchmark_100seed_r79_pneg2_refs \
  --resume --workers 8 \
  --engines graphviz_sfdp__for__classic_sfdp_p_neg2 \
  --graphs <101 p_neg2 graphs> \
  --timeout 300 --watchdog-timeout 600 --variants
```

Result:

```text
3535 total, 3535 ok, 0 skipped, 0 errors, 0 timeouts
```

Coverage: 101 graphs x 35 seeds, seeds 100-134, engine
`graphviz_sfdp__for__classic_sfdp_p_neg2`.

## Rescore

Output:

```text
eval_output/fidelity_definitive/per_combo_r79_pneg2.jsonl
```

The first scorer attempt found stale reimplementation tensors for
`random_dag_50` and `random_dag_200` in the old baseline directory
(`97/383` nodes versus current `50/200`). The successful rescore overlaid the
existing SFDP repair/top-up directories before the corrected reference:

```text
benchmark_100seed_escalation_final
benchmark_100seed_r76_sfdp_fix
benchmark_100seed_r76_sfdp_fix2
benchmark_100seed_r76_sfdp_fix3
benchmark_100seed_r77_sfdp_pack2
benchmark_100seed_r77_randomdag
benchmark_100seed_r79_sfdp_scale
benchmark_100seed_r79_pneg2_refs
```

Result:

```text
101 rows written
94 Mode A rows
7 INSUFFICIENT_DATA rows
0 NO_CANONICAL_REFERENCE rows
```

## Tier movement

Ledger comparison used `scripts/build_definitive_ledger.py` with
`causes_r78.json`. Old tiers are from `per_combo_r78_merged.jsonl`; new tiers
are from `per_combo_r79_pneg2.jsonl`.

Old p_neg2 tiers:

| tier | rows |
|---|---:|
| DISTRIBUTIONAL_EQUIVALENT | 88 |
| QUALITY_EQUIVALENT | 5 |
| DIVERGENT_NAMED_CAUSE | 5 |
| POSITIONAL_IDENTICAL | 2 |
| INSUFFICIENT_DATA | 1 |

New p_neg2 tiers:

| tier | rows |
|---|---:|
| DIVERGENT_UNEXPLAINED | 43 |
| DISTRIBUTIONAL_EQUIVALENT | 27 |
| QUALITY_EQUIVALENT | 17 |
| SUPERIOR_DISTINCT | 7 |
| INSUFFICIENT_DATA | 7 |

Movement:

| movement | rows |
|---|---:|
| DISTRIBUTIONAL_EQUIVALENT -> DISTRIBUTIONAL_EQUIVALENT | 23 |
| DISTRIBUTIONAL_EQUIVALENT -> DIVERGENT_UNEXPLAINED | 40 |
| DISTRIBUTIONAL_EQUIVALENT -> INSUFFICIENT_DATA | 6 |
| DISTRIBUTIONAL_EQUIVALENT -> QUALITY_EQUIVALENT | 14 |
| DISTRIBUTIONAL_EQUIVALENT -> SUPERIOR_DISTINCT | 5 |
| DIVERGENT_NAMED_CAUSE -> DISTRIBUTIONAL_EQUIVALENT | 4 |
| DIVERGENT_NAMED_CAUSE -> QUALITY_EQUIVALENT | 1 |
| INSUFFICIENT_DATA -> INSUFFICIENT_DATA | 1 |
| POSITIONAL_IDENTICAL -> DIVERGENT_UNEXPLAINED | 1 |
| POSITIONAL_IDENTICAL -> QUALITY_EQUIVALENT | 1 |
| QUALITY_EQUIVALENT -> DIVERGENT_UNEXPLAINED | 2 |
| QUALITY_EQUIVALENT -> QUALITY_EQUIVALENT | 1 |
| QUALITY_EQUIVALENT -> SUPERIOR_DISTINCT | 2 |

New divergent-unexplained rows:

```text
binary_tree
broken_symmetry_residual_pair
chung_lu_150
cluster_member_style_stress
clustered_longlabel_handoffs
clustered_medium_5x20
compound_10x20
compound_dag_5x30
dense_pair_50
densenet_block
grid_20x20
grid_rect_6x8
hexagonal_lattice_42
long_range_residual_ladder
long_skip_only_24
nested_cluster_label_stack
outerplanar_dag_20
petersen_10
planar_60
protein_ppi_200
ragged_feature_pyramid
random_dag_200
real_karate_34
real_lesmis_77
recurrent_feedback_cell
regular_3_30
residual_block
resnet_stack_4x16
rgg_100
rgg_500
sbm_4x30
shape_and_routing_matrix
sierpinski_42
small_label_storm
sparse_pair_50
tl_cnn_small
tl_mlp_3layer
tl_resnet_2block
transformer_full_4h_2l
transformer_layer
triangular_lattice_36
weighted_clusters_3x10
weighted_karate_34
```

## Verification

Commands run:

```text
sfdp -Tplain probes for repulsiveforce default/-2/-1/0/1/2/3
python scripts/run_benchmark.py ... p_neg2 refs
python scripts/definitive_fidelity_analysis.py ... p_neg2 rescore
python scripts/build_definitive_ledger.py ... old/new tier comparison
```

The requested full quality gates are listed in the task summary rather than
this report because they were run after the report was written.
