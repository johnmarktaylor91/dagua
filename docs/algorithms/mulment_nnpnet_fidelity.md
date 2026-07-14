# MulMent + NNP-NET Fidelity Notes

## MulMent

- Reference: `karlsruhedraw/KaDraw`, multilevel coarsening plus MaxEnt-Stress local refinement.
- Reference runtime: `kadraw` target built in `/tmp/KaDraw/build` and ran single-threaded with `--seed 7` on `examples/btree.graph`; full `all` target failed because `graphchecker` links object files that reference Cairo symbols but does not link `cairo`.
- Port tier: coarsener-port. The Dagua pipeline now builds a KaDraw-style size-constrained label-propagation hierarchy, optimizes the coarsest graph, projects positions through the hierarchy, and applies a KaDraw-style fixed-point MaxEnt local optimizer. A standalone KaDraw probe on the 12-node hierarchy fixture matches the first two identity levels and first diverges at level 3 (`12->7` reference vs `12->6` Dagua), so the remaining gap is narrowed to LP tie/update semantics rather than Dagua heavy-edge matching.
- Runtime delegation: none. The production pipeline does not import subprocess/FFI adapters.
- RNG: KaDraw-compatible MT19937/libstdc++ tie-break stream for coarsening; seeded Torch CPU generators remain for coarsest initialization.

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
