<task>
R35 universal NumPy-RNG sweep.

Many R31/R32/R33 fixes replaced `torch.Generator` with `np.random.RandomState`
per engine (in `fidelity_mode`) because most reference impls use NumPy. Do
a sweep to find any STOCHASTIC dagua engine that still uses `torch.Generator`
for randomness while its reference is NumPy-based.

## Your job

1. Grep `dagua/layout/ops/` for `torch.Generator`, `manual_seed`,
   `torch.randn`, `torch.rand`, etc.
2. For each, identify the engine and its reference's RNG family
   (check `dagua/eval/competitors/<ref>_competitor.py`).
3. If reference uses NumPy or `random.Random` and dagua uses torch, add a
   `fidelity_mode` path that uses NumPy. Keep the torch default for backward
   compat / performance.
4. Add regression tests asserting bit-exact draw sequence equivalence under
   `fidelity_mode`.

## Skip
- Engines already converted in R31/R32/R33 (umap, drl, tsnet, graphopt, etc.)
- Engines where torch is genuinely required (e.g., dagua native autograd)

## Verification

For each sweep target, run bounded live_compare and confirm no regression:
```bash
python scripts/algo_fidelity_live_compare.py classic_<engine> <reference> --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_35/numpy_rng_sweep/<engine>
```

## Output
`eval_output/algo_fidelity/round_35/numpy_rng_sweep/SUMMARY.md` with engines
swept, RNG before/after, RMSD deltas.

Use commit-safe wrapper.
</task>

<completeness_contract>
Sweep all stochastic engines. Apply fidelity_mode where impactful. Document where not.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
