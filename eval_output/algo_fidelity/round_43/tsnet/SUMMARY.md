# Round 43 tsNET Summary

## Reference Source Lines

The task reference checkout path
`/home/jtaylor/projects/_references/scikit-learn/sklearn/manifold/` was not
present on this machine. I used the installed sklearn source at
`/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/`.

- `_joint_probabilities`: `_t_sne.py` lines 38-68. Casts distances to
  `float32`, calls `_utils._binary_search_perplexity`, symmetrizes, normalizes,
  and returns condensed `P`.
- `_kl_divergence`: `_t_sne.py` lines 128-202. Uses SciPy
  `pdist(..., "sqeuclidean")` condensed ordering, `Q = dist / (2 * sum(dist))`,
  BLAS-backed `np.dot` for KL, and `squareform((P - Q) * dist)` for gradients.
- `_kl_divergence_bh`: `_t_sne.py` lines 205-298 and
  `_barnes_hut_tsne.pyx` lines 37-91, 94-159, and 162-220. Barnes-Hut uses
  sparse `P`, float32 positions/forces, and the Cython quad-tree gradient.
- `_gradient_descent`: `_t_sne.py` lines 301-444. Owns momentum, adaptive
  gains, 50-iteration convergence checks, and resets optimizer state per call.
- `TSNE._fit`: `_t_sne.py` lines 855-942 and 1013-1018. Uses
  `learning_rate=max(N / early_exaggeration / 4, 50)`, squares precomputed
  distances for non-euclidean metrics, computes exact `P`, and initializes with
  NumPy `RandomState` at scale `1e-4`.
- `TSNE._tsne`: `_t_sne.py` lines 1043-1094. Runs 250 early-exaggeration
  iterations at momentum `0.5`, then disables exaggeration and runs the late
  phase at momentum `0.8`.
- `scipy.spatial.distance.pdist`: installed at
  `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/scipy/spatial/distance.py`;
  sklearn exact relies on its condensed vector ordering.

## Implementation

- Replaced the public `fidelity_mode=True` sklearn `TSNE(...)` delegation in
  `dagua/layout/ops/pipelines/tsnet.py` with a local exact-port path.
- Added fidelity selectors `True`, `"sklearn"`, and `"exact"`.
- Added a local sklearn-compatible `RandomState` seed bridge.
- Ported sklearn exact `_kl_divergence` with SciPy condensed `pdist`,
  `squareform`, NumPy double distance accumulation, and float32 parameter/
  gradient dtype matching sklearn random initialization.
- Ported sklearn `_gradient_descent` phase boundaries, including separate
  optimizer buffers for early and late phases.

## Verification

Direct `_kl_divergence` parity probe:

```text
False nan nan 0.0 float32 float32
True 1.9046779868141257 1.9046779868141257 0.0 float32 float32
```

Requested smoke vs `sklearn.manifold.TSNE(method="exact")` at perplexity `30`,
`max_iter=300`, 4 topologies x 3 seeds:

```text
path overall mean RMSD:      5.120337248361495e-17
star overall mean RMSD:      5.882816849922894e-17
clustered overall mean RMSD: 4.914082315119092e-17
grid overall mean RMSD:      1.7435331042404337e-17
overall mean RMSD:           4.4151923794109785e-17
overall max RMSD:            6.856775184376102e-17
max absolute coordinate diff: 0.0
```

Round 41 smoke harness after the local port:

```text
overall: before_mean=0.300040925, after_mean=0.000000000, after_max=0.000000000
```

## Verdict

Bit-exact for the requested smoke target. The residual floor for the ported
exact fidelity path is `0.0` coordinate max-absolute difference against sklearn
on the smoke matrix, below the `<0.001` completeness target.
