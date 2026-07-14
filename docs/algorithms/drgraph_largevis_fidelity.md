# DRGraph + LargeVis fidelity

Implementation: native Python/PyTorch-compatible port of the LargeVis and DRGraph graph-layout source loops. Shared code covers geodesic KNN similarity construction, alias-table edge sampling, degree^0.75 negative sampling, and sampled SGD updates.

Named residual stage: `reference_runtime_rng`. Both references use GSL `rand48` seeded with `314159265`, but this environment could not link GSL, so no reference coordinates were available for runtime residuals.

## Reference build/run

- LargeVis clone: `/tmp/LargeVis`; documented compile command failed: `fatal error: gsl/gsl_rng.h: No such file or directory`.
- DRGraph clone: `/tmp/DRGraph`; documented build first required changing `Boost_USE_STATIC_LIBS` to `OFF` for the local shared Boost install, then failed at link: `cannot find -lgsl` and `cannot find -lgslcblas`.
- Single-thread reference runs were therefore blocked before execution.

## DRGraph license text found in repository

No top-level `LICENSE` or `COPYING` file exists in the cloned `ZJUVAG/DRGraph` snapshot. Source files include mixed third-party notices:

- `src/algorithm/maxheap.h` and `src/algorithm/fastcommunity_mh.cc`: "This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version."
- `src/algorithm/kmeans.h`: MIT-style permission notice beginning "Permission is hereby granted, free of charge, to any person obtaining a copy of this software...".
- `src/ANNOY/annoylib.h`: Apache License, Version 2.0 notice.

## Results

| algorithm | graph | tier | self residual | sampled stress | quality |
| --- | --- | --- | ---: | ---: | --- |
| largevis | chain_5 | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 3.21294e-16 | 0.334294 | ACCEPTABLE |
| largevis | cycle_4 | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 2.5488e-16 | 0.306827 | ACCEPTABLE |
| largevis | diamond | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 2.5488e-16 | 0.224512 | ACCEPTABLE |
| largevis | grid_3x3 | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 4.64104e-16 | 0.514017 | ACCEPTABLE |
| drgraph | chain_5 | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 2.72302e-16 | 0.430011 | ACCEPTABLE |
| drgraph | cycle_4 | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 4.00297e-16 | 0.296696 | ACCEPTABLE |
| drgraph | diamond | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 3.23162e-16 | 0.302867 | ACCEPTABLE |
| drgraph | grid_3x3 | SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED | 2.02636e-16 | 0.42287 | ACCEPTABLE |

## Notes

- Production pipelines do not call adapters, subprocesses, or reference clones.
- LargeVis graph mode uses graph geodesics as the high-dimensional distance space, matching the project t-SNE/UMAP graph-layout adaptation pattern.
- DRGraph multilevel scaffolding is exposed as an API parameter, but this native port runs the deterministic single-level optimizer; this is the main remaining fidelity gap after the blocked reference runtime.
