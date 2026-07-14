# GRIP Fidelity

## Status

GRIP is implemented as `algorithm="grip"` in `dagua.layout.ops.pipelines.grip`.
The production pipeline is native Python/PyTorch and does not call the original
runtime, a cloned source tree, GUI wrappers, or competitor adapters.

## Reference Runtime

- Project page: `https://www2.cs.arizona.edu/~kobourov/GRIP/`
- Paper basis: Gajer, Goodrich, and Kobourov, "A Fast Multi-Dimensional
  Algorithm for Drawing Large Graphs" / GRIP descriptions.
- Source status: the available source archive does not state a license.
- Build attempt: downloaded to `/tmp/dagua_grip_reference` and ran `make`.
- Build result: failed at `MesaPlot.c` because `GL/glut.h` is unavailable in the
  execution environment.
- Runtime result: not established. The available application is GUI/Tcl/OpenGL
  oriented, so no seedable batch adapter was used for this build.

Because the reference did not run, current tier is quality-tier clean-room rather
than bit/similarity-exact against the original C implementation.

## Implemented Stages

- MIS filtration: seeded greedy maximal independent set filtration, with the
  paper's exclusion radius `2**i` for the `V_i -> V_{i+1}` step.
- Intelligent initial placement: coarsest anchors are placed first, then new
  vertices are placed from graph-nearest already placed anchors using the
  two-circle/trilateration construction from the paper.
- Local refinement: per-level Fruchterman-Reingold-style refinement restricted
  to graph-nearest level neighborhoods using the paper's
  `avg_degree * N / |V_i|` neighborhood schedule.

## Verification

Run:

```bash
python scripts/verify_grip_fidelity.py
```

Expected report shape:

```text
GRIP fidelity verification
reference_runtime: build_failed (missing GL/glut.h; GUI/Tcl runtime not established)
reference_license: unlicensed source archive; clean-room implementation from paper
first_divergent_stage: reference-runtime
named_residual: procrustes_rmsd
mis_init_status: isolated deterministic pins pass in tests/test_pipeline_grip.py
no_delegation_guards: pass
path6: residual=0.000e+00 tier=quality-tier-clean-room quality=42.33 mis_sizes=[6, 3]
cycle6: residual=2.343e-16 tier=quality-tier-clean-room quality=28.67 mis_sizes=[6, 3]
diamond_tail: residual=2.591e-16 tier=quality-tier-clean-room quality=40.58 mis_sizes=[5, 3]
two_components: residual=0.000e+00 tier=quality-tier-clean-room quality=43.16 mis_sizes=[6, 4, 2]
```

The residual is a native public-adapter vs composed-pipeline self-check because
the original runtime was not available as a seedable oracle.

## Test Pins

`tests/test_pipeline_grip.py` covers:

- registry and op registration;
- seeded MIS filtration order on a path;
- intelligent placement's circle-equation case;
- a small cycle numeric layout pin;
- seed determinism and seed sensitivity;
- public adapter vs composed pipeline equivalence;
- no-delegation import guard;
- invalid-parameter validation.

## Known Divergence

The clean-room implementation follows the published algorithmic structure, but
it does not claim exact parity with Roman Yusufov's unlicensed C/OpenGL source.
The first unresolved divergent stage is the reference runtime itself: it did not
build in this environment, and no seedable batch interface was established.
