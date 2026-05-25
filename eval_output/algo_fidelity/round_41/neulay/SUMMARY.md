# Round 41 NeuLay Bit-Exact Push

## Reference Source Lines

- `/home/jtaylor/projects/_references/NeuLay/old_code/NeuLay-2.py:244-270`: GCN loop computes `outputs = net(inp)`, refreshes KD-tree pairs on `epoch % 5 == 0`, then performs `optimizer.step()`.
- `/home/jtaylor/projects/_references/NeuLay/old_code/NeuLay-2.py:276-280`: direct phase starts from `torch.nn.Parameter(outputs.detach())` and continues `epoch1` from the final GCN `epoch`.
- `/home/jtaylor/projects/_references/NeuLay/old_code/NeuLay-2.py:286-294`: direct KD-tree refresh uses the continued `epoch1 % 5` cadence, then steps RMSprop.
- `/home/jtaylor/projects/_references/NeuLay/old_code/NeuLay-2.py:317`: persisted output is `outputs1`, the pre-step direct forward value from the final direct iteration.

## Sub-Component Diagnosis

Smoke harness: path/star/clustered/grid, seeds 1/2/3, `gcn_steps=30`, `fdl_steps=50`, `dim=3`, query radius `4`, Procrustes RMSD after centering/unit-Frobenius alignment.

Dominant residuals were not RNG, node order, edge order, or force constants. The remaining gap came from reference loop state:

- Initialization/RNG: matched.
- Force kernel constants: matched (`radius=.4`, `magnitude=100*N**(1/3)*radius`, KD-tree radius `4`).
- Iteration order: edge order was not dominant for these simple undirected smoke graphs.
- Convergence/loop state: dominant. Dagua used post-step `model()` output after GCN and reset direct loop cache/window; the reference uses stale pre-step `outputs`, carries `pairs`, carries the rolling loss window, and continues epoch parity into direct optimization.
- Output finalization: reference writes stale pre-step `outputs1`, not the post-step parameter value.

## Port Summary

Changed `dagua/layout/ops/neulay.py` only. For `fidelity_mode="old_code"`:

- Cache the final GCN pre-step output and use it as the direct-phase initial position.
- Preserve the final GCN KD-tree pair cache and rolling loss window for the direct phase.
- Continue the direct KD-tree refresh cadence from the final GCN epoch.
- Cache the final direct pre-step output and return it in finalization.

Default non-fidelity behavior is unchanged.

## Smoke RMSD

Before:

| topology | seed 1 | seed 2 | seed 3 |
| --- | ---: | ---: | ---: |
| path | 0.015194 | 0.015443 | 0.021140 |
| star | 0.036789 | 0.113669 | 0.024664 |
| clustered | 0.010323 | 0.004455 | 0.012915 |
| grid | 0.012237 | 0.006138 | 0.018385 |

Before mean: `0.024295`; max: `0.113669`.

After:

| topology | seed 1 | seed 2 | seed 3 |
| --- | ---: | ---: | ---: |
| path | 0.00000006 | 0.00000005 | 0.00000004 |
| star | 0.00000004 | 0.00000008 | 0.00000004 |
| clustered | 0.00000007 | 0.00000004 | 0.00000007 |
| grid | 0.00000005 | 0.00000003 | 0.00000007 |

After mean: `0.00000005`; max: `0.00000008`; raw max absolute coordinate delta was `0` for all smoke rows.

## Final Verdict

Bit-exact for the smoke harness. The tiny nonzero Procrustes values are alignment/SVD numerical noise despite zero raw coordinate delta.

## Verification Notes

Passed:

```text
python -m py_compile dagua/layout/ops/neulay.py
ruff check dagua/layout/ops/neulay.py dagua/layout/ops/pipelines/neulay.py --fix
All checks passed!
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Blocked by unrelated in-flight workspace edits:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
TypeError: Cannot overwrite attribute __setattr__ in class InitializeGEMPositions
```

Global ruff is also blocked outside NeuLay:

```text
ruff check .
E501 dagua/layout/ops/pipelines/stress_majorization.py:563
E501 dagua/layout/ops/tsnet.py:178
```
