# ISOM Fidelity

## Reference Runtime

- Reference: JUNG 2.1.1 `edu.uci.ics.jung.algorithms.layout.ISOMLayout`
- Runtime status: JUNG jar ran in this environment through
  `dagua/eval/competitors/isom_competitor.py`
- Jars: downloaded on demand to `/tmp/dagua-isom-jung`; no jars are committed
- Determinism: `RandomLocationTransformer` receives the requested seed, and the
  Java runner seeds `Math.random` by opening `java.lang` for the JUNG random
  training-point path
- RNG: local `JavaRandom` matched pinned JDK `Random.nextDouble()` samples

## Source-Port Surface

The Dagua `algorithm="isom"` pipeline ports the JUNG epoch loop directly:

- seedable Java-random initialization over a 600 x 600 layout region
- random training point at `10 + Math.random() * width/height`
- winner selection by first nearest node in vertex iteration order
- graph-distance BFS neighborhood through undirected JUNG neighbors
- node update factor `adaption / 2^distance`
- exponential adaptation cooling with `coolingFactor = 2`
- radius decay every 100 epochs until `minRadius = 1`

The production pipeline does not call Java or any competitor adapter at runtime.
`tests/test_pipeline_isom.py` includes an AST no-delegation guard for this.

## Verification Results

Command:

```bash
python scripts/verify_isom_fidelity.py
```

Observed output, 2026-07-14:

```text
ISOM fidelity verification
reference_runtime: JUNG jar
rng_matched_java_random: True
model_status: source-faithful JUNG ISOMLayout.java port
path3_short: residual=3.468e-16 tier=bit/similarity-exact quality=26.11
path5_default_window: residual=2.006e-16 tier=bit/similarity-exact quality=44.17
branch6: residual=1.934e-16 tier=bit/similarity-exact quality=46.53
disconnected8: residual=2.874e-16 tier=bit/similarity-exact quality=37.63
cycle_chord6: residual=6.834e-17 tier=bit/similarity-exact quality=33.74
first_divergent_stage: none
```

Residual is rotation/translation/scale/reflection-invariant Procrustes RMSD from
`dagua.eval.equivalence_metrics.procrustes_rmsd`.

## Current Tier

Overall tier: bit/similarity-exact against the runnable JUNG jar on the
verification corpus.

The remaining practical limitation is that deterministic JUNG verification uses
Java reflection to seed `Math.random`; the production Python pipeline has no
such limitation because it uses the local Java RNG port directly.
