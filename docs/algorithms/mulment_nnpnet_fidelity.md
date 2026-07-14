# MulMent + NNP-NET Fidelity Notes

## MulMent

- Reference: `karlsruhedraw/KaDraw`, multilevel coarsening plus MaxEnt-Stress local refinement.
- Reference runtime: `kadraw` target built in `/tmp/KaDraw/build` and ran single-threaded with `--seed 7` on `examples/btree.graph`; full `all` target failed because `graphchecker` links object files that reference Cairo symbols but does not link `cairo`.
- Port tier: quality-faithful. The Dagua pipeline uses Dagua heavy-edge coarsening metadata, optimizes the coarsest graph, projects positions through the hierarchy, and applies a KaDraw-style fixed-point MaxEnt local optimizer.
- Runtime delegation: none. The production pipeline does not import subprocess/FFI adapters.
- RNG: seeded Torch CPU generators for coarsening, coarsest initialization, and uncoarsening jitter.

## NNP-NET

- Reference: `IlanHartskeerl/NNP-NET`, pivot/PMDS embedding plus tsNET teacher and neural projection.
- Reference runtime: default build failed at the executable link step with `undefined reference to symbol 'pthread_create@@GLIBC_2.2.5'` / `DSO missing from command line`; rebuilding in `/tmp/NNP-NET/build-pthread` with explicit `-pthread` flags succeeded and ran single-threaded on `TestGraphs/3elt.mtx`. The reference CLI exposes no seed option for the Keras training path, so reference RNG could not be pinned from the command line.
- Port tier: structural-port. The Dagua pipeline implements farthest-pivot graph-distance features, deterministic farthest teacher sampling, a tsNET teacher stage, and deterministic ridge projection in place of the reference Keras MLP.
- Runtime delegation: none. The production pipeline does not call the reference binary or import TensorFlow/Keras.
- RNG: tsNET teacher initialization is seed-pinned; projection is deterministic.

Run:

```bash
python scripts/verify_mulment_nnpnet_fidelity.py
```
