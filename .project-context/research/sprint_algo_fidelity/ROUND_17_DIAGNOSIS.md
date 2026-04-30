# Round 17 Diagnosis -- NeuLay

Date: 2026-04-30
Family: neulay
Status: diagnosed residual

## Environment

The requested upstream probe did not find a live NeuLay package:

```text
python -c "import torch_geometric; import neulay" 2>&1 | head -5
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'neulay'
```

`torch_geometric` is importable, but `neulay`/`NeuLay` is not. Therefore
`dagua/eval/competitors/neulay_competitor.py` cannot execute the independent
reference implementation in this environment.

Cached target positions are present in `eval_output/benchmark_full/positions/`.
For the requested five graphs, cached `neulay` target tensors exist for all
five graphs, but cached `classic_neulay` records exist for only two graphs in
the benchmark result index. The live comparator's selector requires both a
cached target and a cached Dagua-side record, so the requested baseline command
ran on:

```text
linear_3layer_mlp
parallel_multiedge_bundle
```

## Local Implementation Read

Files read:

```text
dagua/layout/ops/neulay.py
dagua/layout/ops/pipelines/neulay.py
dagua/eval/competitors/neulay_competitor.py
dagua/layout/classic/neulay.py
dagua/eval/variants.py
dagua/eval/competitors/classic_competitor.py
```

Dagua's pipeline is already structurally NeuLay-shaped:

- `NeuLaySeedRNG` seeds PyTorch, NumPy, and CUDA RNGs.
- `_ResGCN` uses a residual GCN with `N x 100` learnable node features, a
  `100 -> 100` sparse normalized-adjacency GCN with `tanh`, a `100 -> 3`
  second GCN, and a final `[h0, h1, h2] -> dim` projection.
- `_build_normalized_adjacency` makes the graph undirected, adds self-loops,
  and applies symmetric degree normalization.
- The GCN phase and direct phase both optimize elastic edge energy plus
  Gaussian KD-tree repulsion using RMSprop.
- Adaptive magnitude is `100 * N^(1/3) * radius`.
- The registered `classic_neulay` competitor passes the benchmark defaults:
  `steps=20000`, `gcn_steps=2000`, `use_gcn=True`, `lr=0.1`, `radius=0.4`.

The local classic module documents that this architecture matches the upstream
`NeuLay-2.py` script, but the actual upstream package source is not installed
locally, so that claim cannot be independently verified in Round 17.

## Highest-Confidence Divergence

No high-confidence one-line implementation divergence could be confirmed from
installed source. The remaining gap is most likely one of:

- upstream source drift from the locally documented `NeuLay-2.py` architecture;
- optimizer/RNG stream differences caused by PyTorch, PyG, or package version;
- objective detail mismatch, especially edge multiplicity handling or loss
  normalization, which cannot be checked against source here;
- cached-reference data mismatch: the comparison uses historical upstream
  tensors rather than a live upstream run in the current environment.

One local inconsistency was found: `layout_neulay_pipeline()` has a public
signature default `lr=0.01`, while the registered benchmark competitor and
variant registry use NeuLay default `lr=0.1`. This does not affect the Round 17
live comparator because `classic_neulay` passes `lr=0.1` explicitly.

## Baseline

Command:

```text
python scripts/algo_fidelity_live_compare.py classic_neulay neulay \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_17/baseline_small
```

Output:

```text
Wrote 30 rows to eval_output/algo_fidelity/round_17/baseline_small/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_17/baseline_small/multi_seed_summary.json
graphs: 2
median: 0.122014
p25: 0.113229
p75: 0.130799
p95: 0.137827
worst: linear_3layer_mlp 0.139584
```

Graph-level TOST:

```text
linear_3layer_mlp: equivalent_at_1.5x and equivalent_at_2x
parallel_multiedge_bundle: not_equivalent through 2x
```

The within-target floor is high for `linear_3layer_mlp`:

```text
within neulay mean: 0.174630
dagua-vs-neulay median: 0.139584
```

The floor is near zero for `parallel_multiedge_bundle`:

```text
within neulay mean: 0.000916
dagua-vs-neulay median: 0.104444
```

## Decision

No code lever was applied. With the upstream package unavailable and only two
selected cached comparison graphs, a code change would be tuning against sparse
historical outputs rather than confirming a source-level NeuLay mismatch.

Classification:

```text
principled_residual: source_unavailable_cached_reference_floor
```
