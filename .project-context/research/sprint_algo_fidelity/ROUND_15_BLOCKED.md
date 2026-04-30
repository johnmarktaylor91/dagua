# Round 15 Blocked -- GraphOpt vs igraph

Status: BLOCKED
Family: graphopt
Date: 2026-04-30

## Reason

The requested primary Dagua ops file, `dagua/layout/ops/graphopt.py`, does not
exist. The Round 15 missing-context gate explicitly says to abort if the Dagua
graphopt ops file is not found, so no implementation change was made.

GraphOpt implementation code does exist, but it is split across shared op
modules:

- `dagua/layout/ops/pipelines/graphopt.py`
- `dagua/layout/ops/force.py`
- `dagua/layout/ops/init.py`
- `dagua/layout/ops/postprocess.py`

Only `dagua/layout/ops/pipelines/graphopt.py` is in Round 15's allowed edit
scope. The most plausible first lever found during diagnosis is in
`dagua/layout/ops/init.py`, which is outside the allowed files.

## Baseline

Command:

```text
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_15/baseline_small
```

Output:

```text
Wrote 75 rows to eval_output/algo_fidelity/round_15/baseline_small/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_15/baseline_small/multi_seed_summary.json
graphs: 5
median: 0.067702
p25: 0.018174
p75: 0.067702
p95: 0.260675
worst: tl_mlp_3layer 0.308918
```

## Diagnosis

igraph GraphOpt source:

- Defaults match the task prompt: `niter=500`, `node_charge=0.001`,
  `node_mass=30`, `spring_length=0`, `spring_constant=1`,
  `max_sa_movement=5`.
- `COULOMBS_CONSTANT` is `8987500000.0`.
- Repulsion is `COULOMBS_CONSTANT * q^2 / d^2` for pairs with
  `0 < d < 500`.
- Spring force uses `abs(distance - spring_length)`, applies half the force to
  each endpoint, and with default `spring_length=0` pulls connected nodes
  together proportional to distance.
- Movement is `force / node_mass`, clamped per axis to `max_sa_movement`.
- Random initialization calls `igraph_layout_random()`, which samples both
  coordinate columns uniformly from `[-1, 1]`.

Dagua observations:

- Pipeline defaults in `dagua/layout/ops/pipelines/graphopt.py` match igraph.
- `_GRAPHOPT_COULOMBS_CONSTANT = 8_987_500_000.0` in
  `dagua/layout/ops/force.py` matches igraph.
- The main GraphOpt force step in `dagua/layout/ops/force.py` implements the
  inverse-square repulsion, 500-unit repulsion cutoff, half-strength spring
  endpoint contribution, and per-axis movement clamp.
- `GraphOptInitializePositions` in `dagua/layout/ops/init.py` initializes with
  `random.Random(seed).random()`, i.e. `[0, 1]`, while igraph initializes in
  `[-1, 1]`.

## Deferred Lever

The conservative next lever is to align GraphOpt random initialization with
igraph's `[-1, 1]` range, then rerun the same small subset. That requires
editing `dagua/layout/ops/init.py` or introducing a GraphOpt-specific init op
in an allowed GraphOpt-owned file. Under this round's scope constraints, the
safe action is to stop and report the blocked scope mismatch.

## Tests

No code change was made, so no pytest target was run. The required live
comparator baseline completed cleanly.
