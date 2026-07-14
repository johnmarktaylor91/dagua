# PaCMAP + Word2VecGD Fidelity

## PaCMAP

- Reference inspected: `pacmap 0.9.1` from PyPI (`YingfanWang/PaCMAP`, Apache-2.0).
- Adaptation: dense graph geodesic distances are used as PaCMAP's high-dimensional feature matrix.
- Native implementation: `dagua/layout/ops/pipelines/pacmap.py`.
- Runtime delegation: production pipeline does not import `pacmap`; only the evaluation adapter does.
- Fidelity pins:
  - nearest, mid-near, and further pair counts follow PaCMAP small-sample reorganization;
  - deterministic mid-near/further pair sampling mirrors PaCMAP's legacy `np.random.seed(...)` reseeding;
  - optimizer matches the package core within the observed float32/Numba accumulation floor.

## Word2VecGD

- Reference inspected: `mlyann/graphv_nn` cloned under `/tmp/graphv_nn`.
- Reference shape: Python `random.choice` random walks, gensim skip-gram Word2Vec embeddings, then normalized stress evaluation.
- Native implementation: `dagua/layout/ops/pipelines/word2vecgd.py`.
- Runtime delegation: production pipeline does not import `graphv_nn` or `gensim`.
- Fidelity pins:
  - random-walk corpus order matches the reference's node-major walk generation and Python RNG;
  - skip-gram training is deterministic under seed;
  - cosine-stress placement reduces the native normalized objective.

## Verification

Run:

```bash
python scripts/verify_pacmap_word2vecgd_fidelity.py
```

The script prints per-algorithm tier and quality score plus no-delegation guard status.
