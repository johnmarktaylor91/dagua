# Round 2 Diagnosis Only

No Sugiyama source fix was applied. The live baseline does not match Round 1
cached RMSDs within the required 0.005 tolerance, so applying an algorithm fix
would measure against a moving target.

The actionable finding is infrastructure-level: `scripts/algo_fidelity_live_compare.py`
now mirrors the benchmark harness by computing node sizes before live competitor
runs and falls back to local TorchLens graph cache files when optional TorchLens
is not installed. This makes the live comparator runnable end-to-end, but it
also exposes that current node-size measurement differs from the cached
benchmark positions.

Recommended unblock: regenerate `eval_output/benchmark_full/positions` for
`classic_sugiyama` and `graphviz_dot` with the current render/text measurement
stack, or serialize the original benchmark graph node-size tensors alongside
positions so live comparisons can exactly replay the cached benchmark context.
