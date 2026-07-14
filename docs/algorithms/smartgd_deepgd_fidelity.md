# SmartGD and DeepGD Fidelity

SmartGD and DeepGD are neural graph drawing methods, so trained-weight
availability defines the fidelity target. The upstream SmartGD repository
cloned at `~/tools/dagua-refs/smartgd` ships `generator_stress_only.pt` and
`generator_xing_only.pt`. The upstream DeepGD repository cloned at
`~/tools/dagua-refs/deepgd` ships `model_stress_only.pt`.

The Dagua SmartGD port in `dagua/layout/ops/pipelines/smartgd.py` reimplements
the generator architecture directly: NNConv edge-conditioned layers, dynamic
relative edge features, residual generator blocks, shortest-path edge features,
and stress-based coordinate rescaling. Runtime layout does not import the
cloned reference repository.

The Dagua DeepGD port in `dagua/layout/ops/pipelines/deepgd.py` uses the same
ported generator architecture with DeepGD checkpoint defaults. The upstream
SmartGD source confirms SmartGD is subsequent work built on DeepGD, and the
generator source trees differ only in package names and comments.

Verification command:

```bash
python scripts/verify_smartgd_deepgd_fidelity.py
```

Expected scope:

- Port correctness is checked by loading the same checkpoint into the reference
  generator and Dagua model, feeding identical prepared tensors, and comparing
  forward outputs for bit-exact tensor equality. Rotation-invariant Procrustes
  residuals are still reported as a diagnostic.
- Deterministic inference is enforced with `torch.manual_seed(seed)`,
  `torch.use_deterministic_algorithms(True)`, deterministic CuDNN settings, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; CUDA is used when available.
- Layout quality is reported with sampled stress, edge crossings, and
  neighborhood preservation.

Current SmartGD residual:

- Pretrained checkpoint availability: true when
  `~/tools/dagua-refs/smartgd/generator_stress_only.pt` is present.
- Fidelity tier: positional bit-exact on the verifier corpus.
- First divergent stage: none. The previous mismatch was traced to `_DEFAULT_EPS`
  in the dynamic edge-feature expansion; upstream SmartGD/DeepGD both define
  `EPS = 1e-5`.

Current DeepGD residual:

- Pretrained checkpoint availability: true when
  `~/tools/dagua-refs/deepgd/model_stress_only.pt` is present.
- Fidelity tier: positional bit-exact on the verifier corpus.
- First divergent stage: none.
