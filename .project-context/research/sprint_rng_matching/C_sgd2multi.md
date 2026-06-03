Close sgd2_multi to bit-exact (real RNG gap, tractable). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY. Pipeline:
dagua/layout/ops/pipelines/sgd2_multi.py. Reference: sgd2_multi_ref (s_gd2 / S_GD2). Now runs.
Current: batch128 9.6e-6 (CLOSE), lr001 0.03, with_aspect 0.05, default 0.08, others up to 1.09.
The gap is s_gd2's stochastic SGD: it SHUFFLES the term order each epoch via a PRNG. Read s_gd2
source (installed: ~/anaconda3/envs/py311/lib/python3.11/site-packages/s_gd2/) for its exact RNG
(seed handling + per-epoch shuffle) + the SGD step schedule + arithmetic order. Reimplement the
GENERATOR (NO replay, NO delegation). Verify python scripts/rng_match/check_engine.py
classic_sgd2_multi_default (+ batch128/lr001/etc). Drive <1e-7; document any chaotic residual.
