# Round 33 Reference Adapter Audit

## Scope

Audited `dagua/eval/competitors/` reference adapters with focus on NeuLay,
`s_gd2`, and `(SGD)^2` multi-reference wiring. Also checked the rest of the
registered competitor adapters through `available()` so missing references are
not hidden behind registration.

## Import checks

| Reference | Import command/result | Status |
| --- | --- | --- |
| PyTorch Geometric | `python -c "import torch_geometric; print('ok')"` -> ok | works |
| NeuLay module | `python -c "import neulay; print('ok')"` -> `ModuleNotFoundError` | broken |
| NeuLay module, case variant | `python -c "import NeuLay; print('ok')"` -> `ModuleNotFoundError` | broken |
| `s_gd2` | `python -c "import s_gd2; print('ok')"` -> ok | works |
| UMAP | `python -c "import umap; print('ok')"` -> ok | works |
| sklearn | `python -c "import sklearn; print('ok')"` -> ok | works |
| igraph | `python -c "import igraph; print('ok')"` -> ok | works |
| NetworkX | `python -c "import networkx; print('ok')"` -> ok | works |
| FA2 | `python -c "import fa2; print('ok')"` -> ok | works |
| ForceAtlas2 alternate package | `python -c "import forceatlas2; print('ok')"` -> `ModuleNotFoundError` | not needed; `fa2` works |

## Adapter availability

`get_competitors()` showed every registered adapter available except:

| Adapter | `available()` | Reason |
| --- | ---: | --- |
| `neulay` | false | No importable `neulay` or `NeuLay` package exposes `layout_neulay` / `layout`. |
| `sgd2_multi_ref` | false | `/tmp/graph-drawing` exists, but required Python sources `gd2.py` and `criteria.py` are absent. |

All other registered adapters reported `available=True`, including classic
Dagua-backed baselines, Graphviz, igraph, NetworkX, dagre, ELK, Cytoscape fCoSE,
Gephi YifanHu, OGDF, FA2, `sgd2`, `sgd2_mds`, t-SNE, and UMAP.

## Focal layout checks

Using a small six-node chain equivalent to the `linear_3layer_mlp` shape:

| Adapter | Result |
| --- | --- |
| `neulay` | returned `pos=None`, error `upstream NeuLay reference is not installed` |
| `sgd2` | returned finite positions with shape `[6, 2]` |
| `sgd2_mds` | returned finite positions with shape `[6, 2]` |
| `sgd2_multi_ref` | returned `pos=None`, error `missing upstream SGD2 source files at /tmp/graph-drawing: gd2.py, criteria.py` |

The exact requested `get_test_graphs()` focal command did not complete within
30 seconds in this checkout while other Round 33 agents were modifying layout
registry files; it reached graph collection but timed out before calling the
NeuLay adapter. The direct NeuLay adapter check above confirms the same
reference-side failure without depending on graph collection.

## Install attempts

Tried low-risk installs:

- `python -m pip install neulay`: package is already installed as editable
  `neulay==0.0.0`, but import still fails.
- `python -m pip install NeuLay`: resolves to the same editable
  `neulay==0.0.0`, import still fails.
- `python -m pip install git+https://github.com/csabath95/NeuLay.git`: fails
  because the upstream repo has neither `setup.py` nor `pyproject.toml`.
- `python -m pip install git+https://github.com/tiga1231/graph-drawing.git`:
  fails because the upstream repo has neither `setup.py` nor `pyproject.toml`.

The installed editable NeuLay distribution points at `/tmp/neulay_pkg/neulay`,
but that path does not exist. This is an environment-level broken wrapper, not a
Dagua adapter import typo.

## Wiring fixes made

No adapter code changes were made.

Reason: `neulay_competitor.py` is already explicit about requiring an
independent upstream callable and must not fall back to Dagua's own NeuLay
implementation. The cloned NeuLay reference under
`/home/jtaylor/projects/_references/NeuLay` is research script/notebook code
with hard-coded top-level dataset loading and no reusable importable layout
entry point, so a correct wrapper is non-trivial and should be a separate
reference-port task.

`sgd2_multi_competitor.py` already reports the missing upstream files clearly.
The `/tmp/graph-drawing` clone is present, but current checkout contents do not
include `gd2.py` or `criteria.py`; they are not recoverable through pip install
because the repo is not a Python package.

## TODO / next steps

| Engine | Recommendation |
| --- | --- |
| NeuLay | Build and install a real local wrapper package around the cloned reference, or replace the adapter target with a committed wrapper that ports `NeuLay-2.py` into a side-effect-free callable. Keep it independent from Dagua's native NeuLay implementation. |
| `sgd2_multi_ref` | Locate the historical upstream commit/artifact containing `gd2.py` and `criteria.py`, or port those modules from the original notebook into `/tmp/graph-drawing`/a committed reference wrapper. Until then `(SGD)^2` multi cannot produce paired reference rows. |
| `sgd2` / `sgd2_mds` | No action; the installed `s_gd2` package imports and returns positions. |
| Other adapters | No immediate action from this audit; availability checks passed. Re-run targeted layout smoke tests after concurrent Round 33 registry work settles. |
