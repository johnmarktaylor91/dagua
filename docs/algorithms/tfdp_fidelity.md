# t-FDP fidelity

Implementation: native torch exact t-force loop with PMDS/random initialization.

## Reference blocker

t-FDP reference failed for single_node: /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/tmp/tfdp-ref/source_code/utils.py:21: RuntimeWarning: invalid value encountered in divide
  b_k = b_k1 / b_k1_norm
Traceback (most recent call last):
  File "<string>", line 40, in <module>
ZeroDivisionError: division by zero


## Exact mode

| graph | residual | tier | stress delta | neighborhood delta |
| --- | ---: | --- | ---: | ---: |
| single_node | nan | REFERENCE_BLOCKED | nan | nan |
| small_chain | 0.0226823 | POSITIONAL | 0.0127504 | 0 |
| diamond | 0.24478 | DISTRIBUTIONAL | 0.115418 | 0 |
| cycle_4 | 0.65777 | DISTRIBUTIONAL | 0.18055 | 0 |
| grid_3x3 | 0.479476 | DISTRIBUTIONAL | 0.0380881 | 0 |
| disconnected | 0.478177 | DISTRIBUTIONAL | 0.0881646 | 0 |

## FFT mode

The public `force_mode='fft'` hook is present, but native Dagua currently falls back to the exact force evaluator. The reference FFT path is a pyFFTW/Numba interpolation implementation and remains a named gap.
