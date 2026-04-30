# Round 18 Residual -- tsNET vs sklearn TSNE

Date: 2026-04-30
Classification: `stochastic_floor_match_with_low_floor_exception`

## Verdict

No code change was committed. The only high-confidence small lever,
sklearn-compatible NumPy random initialization, did not meet the commit
criterion and was reverted.

Baseline median:

```text
0.337116
```

Post-fix median after the attempted initialization alignment:

```text
0.343569
```

The attempted lever regressed the median by `0.006453`, so it missed the
required `>= 0.03` improvement. Aggregate graph-level TOST did not improve:
four of five graphs were already equivalent at `0.5x`, and
`parallel_multiedge_bundle` remained `not_equivalent`.

## Why This Is Residual

Most of the requested subset is already inside the sklearn stochastic floor:

| Graph | Baseline median | Reference within-seed median | TOST `0.5x` |
|---|---:|---:|---|
| `linear_3layer_mlp` | 0.397908 | 0.429070 | equivalent |
| `mixed_width_labels` | 0.322619 | 0.397920 | equivalent |
| `nested_shallow_enc_dec` | 0.397908 | 0.429070 | equivalent |
| `tl_mlp_3layer` | 0.337116 | 0.378084 | equivalent |
| `parallel_multiedge_bundle` | 0.169341 | 0.000001 | not equivalent |

The remaining failure is a low-floor exception on a tiny symmetric multiedge
graph, where sklearn Barnes-Hut t-SNE collapses to nearly identical layouts
across seeds but dagua still explores different basins.

## Attempted Lever

Changed `TsnetInitializePositions` locally to use:

```text
np.random.RandomState(seed).standard_normal((N, 2)).astype(np.float32) * 1e-4
```

This matches sklearn's `init='random'` implementation in sklearn `1.8.0`.
The comparison result worsened, so the change was reverted.

## Remaining Architectural Floor

The remaining gap is not a clean hyperparameter mismatch. The likely floor is
implementation-level:

- sklearn uses its own Barnes-Hut objective and gradient descent machinery.
- dagua uses PyTorch autograd over a dense full probability matrix.
- sklearn builds sparse nearest-neighbor probabilities in the Barnes-Hut path,
  even when the small-graph neighbor count includes all other nodes.
- sklearn can stop based on gradient/progress checks every 50 iterations;
  dagua runs exactly the requested step count.

A faithful fix would likely require a sklearn-style objective/optimizer
reimplementation rather than another scalar hyperparameter alignment.
