# r75 FMMM Probe Results

Date: 2026-07-01
Repo: `/home/jtaylor/projects/dagua`
HEAD observed: `89ed3c3`
Mode: research/probe only; dagua source was monkeypatched in memory from `/tmp/r75_probe_fmmm.py`.

## Commands

```bash
python /tmp/r75_probe_fmmm.py --output /tmp/r75_probe_fmmm_results.json
MPLCONFIGDIR=/tmp/mplconfig python /tmp/r75_probe_fmmm.py --output /tmp/r75_probe_fmmm_results.json
```

The first command failed before any experiment because the scratch script used a direct dotted
`import dagua.layout.ops.pipelines.fmmm as fmmm`, which collided with the public `dagua.layout`
attribute. The rerun used `importlib.import_module("dagua.layout.ops.pipelines.fmmm")` and completed.

Raw JSON is at `/tmp/r75_probe_fmmm_results.json`.

Benchmark path used:

```python
get_competitor("classic_fmmm").layout_with_variant(
    graph, timeout=120, seed=seed,
    variant_params={"steps": 10, "fidelity_mode": True},
)
get_competitor("ogdf_fmmm").layout_with_variant(
    graph, timeout=120, seed=seed,
    variant_params={"fixed_iterations": 10},
)
```

Graphs: `deep_chain_20`, `grid_5x5`, `weighted_chain_20`,
`asymmetric_hourglass_hub`. Seeds: `42,43,44,45,46`.

## Position Quantization Check

Verdict: both sides quantize to integer coordinates during FMMM fidelity iterations.

- Dagua: `_ogdf_fmmm_force_iteration` calls `_ogdf_fmmm_adjust_positions(...)`
  before repulsion each iteration. `_ogdf_fmmm_adjust_positions` floors every x/y
  coordinate when `final_floor=True`, which is the default used by the force loop.
- OGDF: `FMMMLayout::adjust_positions` floors every x/y coordinate in the
  `AllowedPositions::Integer` path. `scripts/ogdf_runner.cpp` does not override
  `allowedPositions`, so the default OGDF behavior applies.
- Consequence: the coincidence census is testing the relevant integer-floored
  state, not an all-float dagua-only state.

## Experiment 1: Coincident-Node Repulsion Trigger Census

Instrumentation: wrapped `_ogdf_fmmm_tensor_repulsive_forces` in memory and counted
off-diagonal exact zero-distance pairs at each call before delegating to the original
function. Counts below are unordered node pairs summed across all calls.

| graph | seed | repulsion calls | unordered zero-pair total | trigger calls | classic stress |
|---|---:|---:|---:|---:|---:|
| deep_chain_20 | 42 | 130 | 0 | 0 | 0.840535027 |
| deep_chain_20 | 43 | 130 | 0 | 0 | 0.842156397 |
| deep_chain_20 | 44 | 130 | 0 | 0 | 0.826168562 |
| deep_chain_20 | 45 | 130 | 0 | 0 | 0.805679875 |
| deep_chain_20 | 46 | 130 | 0 | 0 | 0.818422252 |
| grid_5x5 | 42 | 130 | 0 | 0 | 0.671770224 |
| grid_5x5 | 43 | 130 | 0 | 0 | 0.700089455 |
| grid_5x5 | 44 | 130 | 0 | 0 | 0.632918487 |
| grid_5x5 | 45 | 130 | 0 | 0 | 0.631851239 |
| grid_5x5 | 46 | 130 | 0 | 0 | 0.694694114 |
| weighted_chain_20 | 42 | 130 | 0 | 0 | 0.825791304 |
| weighted_chain_20 | 43 | 130 | 0 | 0 | 0.830182587 |
| weighted_chain_20 | 44 | 130 | 0 | 0 | 0.811354803 |
| weighted_chain_20 | 45 | 130 | 0 | 0 | 0.780110712 |
| weighted_chain_20 | 46 | 130 | 0 | 0 | 0.788748867 |
| asymmetric_hourglass_hub | 42 | 130 | 0 | 0 | 0.666297225 |
| asymmetric_hourglass_hub | 43 | 130 | 0 | 0 | 0.674683563 |
| asymmetric_hourglass_hub | 44 | 130 | 0 | 0 | 0.654122804 |
| asymmetric_hourglass_hub | 45 | 130 | 0 | 0 | 0.666009545 |
| asymmetric_hourglass_hub | 46 | 130 | 0 | 0 | 0.649920871 |

Verdict: KILLED for these requested failing rows. Dagua and OGDF differ in source
behavior for exactly coincident repulsion, but the trigger never fired in any of the
20 benchmark-path runs. Do not implement the jitter fallback as an r75 fix for these
rows.

Recommended minimal gated fix: explicit kill/no code change for this sprint. Keep the
finding as a future guardrail only if a broader census finds actual zero-pair triggers.

## Experiment 2: Oscillation-Damping Angle Formula

Patch: replaced tensor oscillation damping's angle computation with OGDF's literal
subtraction form:

```python
angles = atan2(force_y, force_x) - atan2(previous_y, previous_x)
angles = where(angles < 0, angles + 2*pi, angles)
buckets = ceil(angles / 0.52359878)
```

This matches `GenericPoint::angle` range behavior: negative differences are wrapped
once into `[0, 2*pi)`. The bucket index was clamped to dagua's existing `[0, 13]`
safety range.

| graph | seed | base-vs-swap RMSD | base-vs-OGDF RMSD | swap-vs-OGDF RMSD | moved toward? | base stress | swap stress | OGDF stress |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| deep_chain_20 | 42 | 6.598e-17 | 0.051886660 | 0.051886660 | false | 0.840535027 | 0.840535027 | 0.854246404 |
| deep_chain_20 | 43 | 4.104e-17 | 0.093424689 | 0.093424689 | false | 0.842156397 | 0.842156397 | 0.851210878 |
| deep_chain_20 | 44 | 4.120e-17 | 0.069794358 | 0.069794358 | false | 0.826168562 | 0.826168562 | 0.850124914 |
| deep_chain_20 | 45 | 5.874e-17 | 0.123445780 | 0.123445780 | false | 0.805679875 | 0.805679875 | 0.838818576 |
| deep_chain_20 | 46 | 3.316e-17 | 0.080423803 | 0.080423803 | false | 0.818422252 | 0.818422252 | 0.834454381 |
| grid_5x5 | 42 | 6.419e-17 | 0.039013548 | 0.039013548 | false | 0.671770224 | 0.671770224 | 0.639065032 |
| grid_5x5 | 43 | 4.419e-17 | 0.139466730 | 0.139466730 | false | 0.700089455 | 0.700089455 | 0.641179543 |
| grid_5x5 | 44 | 0.000e+00 | 0.002043296 | 0.002043296 | false | 0.632918487 | 0.632918487 | 0.632867343 |
| grid_5x5 | 45 | 5.628e-17 | 0.001225687 | 0.001225687 | false | 0.631851239 | 0.631851239 | 0.631859181 |
| grid_5x5 | 46 | 1.210e-17 | 0.098165093 | 0.098165093 | false | 0.694694114 | 0.694694114 | 0.642111099 |
| weighted_chain_20 | 42 | 4.883e-17 | 0.049336648 | 0.049336648 | false | 0.825791304 | 0.825791304 | 0.830930258 |
| weighted_chain_20 | 43 | 4.036e-17 | 0.088496489 | 0.088496489 | false | 0.830182587 | 0.830182587 | 0.827042095 |
| weighted_chain_20 | 44 | 2.511e-17 | 0.074261796 | 0.074261796 | false | 0.811354803 | 0.811354803 | 0.822930530 |
| weighted_chain_20 | 45 | 4.140e-17 | 0.105969757 | 0.105969757 | false | 0.780110712 | 0.780110712 | 0.815406218 |
| weighted_chain_20 | 46 | 2.227e-17 | 0.091169568 | 0.091169568 | false | 0.788748867 | 0.788748867 | 0.815174913 |
| asymmetric_hourglass_hub | 42 | 4.703e-17 | 0.041737981 | 0.041737981 | false | 0.666297225 | 0.666297225 | 0.666324578 |
| asymmetric_hourglass_hub | 43 | 3.539e-17 | 0.085154618 | 0.085154618 | false | 0.674683563 | 0.674683563 | 0.666071274 |
| asymmetric_hourglass_hub | 44 | 4.088e-17 | 0.022554218 | 0.022554218 | false | 0.654122804 | 0.654122804 | 0.663196852 |
| asymmetric_hourglass_hub | 45 | 3.901e-17 | 0.037456534 | 0.037456534 | false | 0.666009545 | 0.666009545 | 0.663742341 |
| asymmetric_hourglass_hub | 46 | 5.351e-17 | 0.031541897 | 0.031541897 | false | 0.649920871 | 0.649920871 | 0.665017383 |

Verdict: KILLED. The swap is effectively a numerical no-op here: base-vs-swap
RMSD is zero to about `1e-17`, normalized stress is unchanged, and no row moves
toward the OGDF reference.

Recommended minimal gated fix: explicit kill/no code change. The cross/dot formula
does not explain these rows.

## Cheap First-Divergence Dump

Dagua intermediate positions were captured by wrapping `_ogdf_fmmm_force_iteration`
for `deep_chain_20`, seed 42. The first occurrence of iterations `1,2,3,5,10`
is from the main force loop. Later duplicate iteration labels in the raw JSON come
from postprocessing/fine-tuning phases that also call the same force-iteration
function with local phase iteration numbers.

OGDF intermediate positions were not captured because `scripts/ogdf_runner.cpp`
supports only final-output JSON. It has no flag or payload key for iteration dumps.

Compact dagua dump:

| iteration | bbox x_min | bbox x_max | bbox y_min | bbox y_max | first five positions |
|---:|---:|---:|---:|---:|---|
| 1 | 1.4 | 431.6 | 4.4 | 449.6 | `[[194.568,414.136],[96.447,379.927],[404.559,310.897],[310.176,81.417],[231.569,80.860]]` |
| 2 | 1.9 | 430.1 | 4.9 | 448.2 | `[[193.137,414.275],[96.894,378.854],[403.117,309.797],[310.350,81.836],[230.138,79.722]]` |
| 3 | 2.8 | 428.3 | 5.7 | 446.3 | `[[191.275,414.556],[97.788,377.709],[401.233,308.598],[310.694,82.674],[228.274,78.449]]` |
| 5 | 12.1 | 417.1 | 14.8 | 435.4 | `[[180.714,418.608],[107.146,374.791],[389.926,305.423],[313.584,91.772],[217.046,73.956]]` |
| 10 | 79.5 | 289.2 | 75.6 | 444.3 | `[[175.545,444.331],[231.713,369.495],[289.056,280.508],[289.187,168.756],[203.011,91.736]]` |

Full dagua snapshots are in `/tmp/r75_probe_fmmm_results.json`.

## Bottom Line

Verdict 9: KILLED for the requested rows. No coincident-node repulsion triggers occurred.

Verdict 10: KILLED. The OGDF angle-subtraction formula does not move the layouts or
metrics in practice on the requested rows.

Minimal recommendation: do not land either fix. Continue the r75 FMMM search elsewhere;
these two hypotheses are not causes for the probed failures.
