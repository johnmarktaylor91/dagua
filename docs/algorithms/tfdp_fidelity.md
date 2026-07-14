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
| small_chain | 4.5993e-07 | BIT/SIMILARITY_EXACT | 1.4869e-06 | 0 |
| diamond | 2.38796e-05 | POSITIONAL | 1.53594e-06 | 0 |
| cycle_4 | 2.98772e-06 | POSITIONAL | 4.70942e-06 | 0 |
| grid_3x3 | 3.23354e-06 | POSITIONAL | 2.83023e-06 | 0 |
| disconnected | 2.9798e-06 | POSITIONAL | 2.14485e-06 | 0 |

## FFT mode

The public `force_mode='fft'` hook is present, but native Dagua currently falls back to the exact force evaluator. The reference FFT path is a pyFFTW/Numba interpolation implementation and remains a named gap.
