# Bertault Fidelity

`algorithm="bertault"` implements the default OGDF `BertaultLayout` force loop
locally, without runtime delegation to `scripts/ogdf_runner`.

## Reference

The reference runner uses `ogdf/misclayout/BertaultLayout.h` from the prebuilt
OGDF install under `/home/jtaylor/tools/ogdf`. The runner-owned initial layout is
the same seeded `std::rand() % 1000 / 10.0` coordinate grid used by the existing
OGDF fidelity runners.

## Implemented Stages

1. Seeded initial coordinates matching `scripts/ogdf_runner.cpp`.
2. Default required edge length from the initial average edge length.
3. `10 * N` iterations unless an explicit iteration budget is supplied.
4. Node-node repulsion.
5. Adjacent-node attraction.
6. Node-edge repulsion for projections inside the edge segment.
7. Bertault section radius updates for inside/outside edge projections.
8. Serial node movement clipped by the active section radius.

OGDF's optional `impred` preprocessing path is not enabled by the default
constructor and is not part of this pipeline.

## Fidelity Tier

The local port reaches the `NUMERIC` tier on the pinned small graph set:
maximum absolute residual is below `1e-6` against the compiled runner, and the
rotation-invariant Procrustes residual is printed by
`scripts/verify_bertault_fidelity.py`.

Named residual: `python-float-loop-order`. The first divergence is after the
force accumulation/move stage, where Python and compiled C++ evaluate the same
double-precision arithmetic with slightly different instruction ordering.

## Verification

Run:

```bash
python scripts/verify_bertault_fidelity.py
python -m pytest tests/test_pipeline_bertault.py -q
```
