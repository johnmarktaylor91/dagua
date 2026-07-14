# CoRe-GD Fidelity

CoRe-GD is a neural graph layout method, so trained-weight availability defines
the fidelity target. The upstream repository cloned at `/tmp/coregd-ref` ships
pretrained checkpoints in `checkpoints/`, including `core_rome.pt`.

The Dagua port in `dagua/layout/ops/pipelines/coregd.py` reimplements the
reference architecture directly: encoder MLP, GRU/GIN/GAT edge convolutions,
iterative sigmoid coordinate decoding, and positional rewiring via KNN,
Delaunay, or radius graph overlays. Runtime layout does not import the cloned
reference repository.

Verification command:

```bash
python scripts/verify_coregd_fidelity.py
```

Expected scope:

- Port correctness is checked by loading the same pretrained checkpoint into the
  reference model and Dagua model, feeding identical prepared PyG data, and
  comparing forward outputs.
- Layout quality is reported with sampled stress, edge crossings, and
  neighborhood preservation.
- Bit-exact trained layout claims beyond that forward equivalence are not made;
  CoRe-GD is learned and depends on checkpoint choice plus randomized features.
