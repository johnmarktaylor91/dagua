# SmartGD and DeepGD Fidelity

SmartGD and DeepGD are neural graph drawing methods, so trained-weight
availability defines the fidelity target. The upstream SmartGD repository
cloned at `/tmp/smartgd-ref` ships `generator_stress_only.pt` and
`generator_xing_only.pt`.

The Dagua SmartGD port in `dagua/layout/ops/pipelines/smartgd.py` reimplements
the generator architecture directly: NNConv edge-conditioned layers, dynamic
relative edge features, residual generator blocks, shortest-path edge features,
and stress-based coordinate rescaling. Runtime layout does not import the
cloned reference repository.

Verification command:

```bash
python scripts/verify_smartgd_deepgd_fidelity.py
```

Expected scope:

- Port correctness is checked by loading the same checkpoint into the reference
  generator and Dagua model, feeding identical prepared tensors, and comparing
  forward outputs up to Procrustes alignment.
- Layout quality is reported with sampled stress, edge crossings, and
  neighborhood preservation.
- Bit-exact trained layout claims beyond same-weight forward equivalence are not
  made; SmartGD and DeepGD are learned and depend on checkpoint choice plus
  preprocessing.

Current SmartGD residual:

- Pretrained checkpoint availability: true when `/tmp/smartgd-ref` is present.
- First divergent stage: dynamic edge-feature router. The reference wraps this
  path in TorchScript tracing; the Dagua port loads weights strictly and matches
  closely by Procrustes residual, but not bit-exactly.
