# OGDF runner rebuild

## Version and location

- OGDF source: official `https://github.com/ogdf/ogdf.git` tag `foxglove-202510`
- Source checkout: `/home/jtaylor/tools/ogdf-src`
- Build directory: `/home/jtaylor/tools/ogdf-build`
- Install prefix: `/home/jtaylor/tools/ogdf`
- Installed headers include `/home/jtaylor/tools/ogdf/include/ogdf/basic/Graph.h`
- Installed static libs include `/home/jtaylor/tools/ogdf/lib/libOGDF.a` and `libCOIN.a`

Rebuild with:

```bash
scripts/rng_match/build_ogdf_runner.sh
```

The script clones the official tagged OGDF release, installs it in user space,
rebuilds `scripts/ogdf_runner`, and runs the matched-parameter checks.

## Verification on 2026-06-02

Environment:

```bash
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:$LD_LIBRARY_PATH
```

Before rebuild, using a temporary pre-patch runner compiled from `HEAD`:

- `python scripts/rng_match/check_engine.py classic_gem_iters100 --seeds 1`
  - max RMSD: `1.149744208e+00`
- `python scripts/rng_match/check_engine.py classic_fmmm_steps100 --seeds 1`
  - max RMSD: `1.395544671e+00`

After rebuild:

- `python scripts/rng_match/check_engine.py classic_gem_iters100 --seeds 1`
  - max RMSD: `7.008235088e-08`
- `python scripts/rng_match/check_engine.py classic_gem_iters100`
  - max RMSD: `7.965414923e-08`
- `python scripts/rng_match/check_engine.py classic_gem_iters500`
  - max RMSD: `8.246905089e-08`
- `python scripts/rng_match/check_engine.py classic_fmmm_steps100 --seeds 1`
  - max RMSD: `1.390498235e+00`
- `python scripts/rng_match/check_engine.py classic_fmmm_steps100`
  - max RMSD: `1.390498235e+00`

Direct runner sanity checks confirm the rebuilt binary honors parameters:

- `gemRounds=1` vs `gemRounds=100` raw RMSD on a 6-node path:
  `1.625227514e+02`
- `fmmmFixedIterations=1` vs `fmmmFixedIterations=100` raw RMSD on a 6-node path:
  `1.229620538e+02`

GEM is now under the harness `1e-7` threshold. FMMM still diverges despite the
runner honoring `fmmmFixedIterations`; this points to current FMMM reference
behavior or port alignment rather than the runner ignoring matched parameters.
