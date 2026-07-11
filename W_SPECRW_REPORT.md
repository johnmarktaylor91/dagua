# Classic spectral random-walk closure

## Outcome

The remaining random-walk divergences were a fixable sparse-eigensolver mismatch, not an
incorrect random-walk Laplacian, an incorrect right-eigenvector back-map, or a mathematical
quality floor.

The refreshed spectral artifact is:

- `eval_output/fidelity_definitive/per_combo_r79_spectral.jsonl`
- 420 rows, all uniquely keyed, with no scoring errors
- 417 `INVARIANCE_EQUIVALENT`, 3 `QUALITY_EQUIVALENT`, 0 `DIFFERENT`
- `classic_spectral_random_walk`: 105/105 `INVARIANCE_EQUIVALENT`

## Root cause

Both Dagua and the independent `nx_spectral_random_walk` adapter construct
`L_rw = I - D^-1 A` correctly. Dagua's former sparse fidelity path transformed that matrix to
the symmetric similar problem, solved it with `eigsh`, and mapped the orthonormal eigenvectors
back with `D^-1/2`. The reference adapter instead solved `L_rw` directly with nonsymmetric
`eigs` and used its right eigenvectors.

The two approaches select the same invariant subspace on the affected connected graphs, but
they do not select the same basis inside a repeated eigenvalue. Nonsymmetric ARPACK can return
a non-orthogonal basis, so substituting the symmetric solve changes the final geometry even
after Procrustes alignment.

Direct instrumentation found:

| Graph | Selected repeated eigenvalue | Principal angles between selected subspaces | Dagua/reference residual norms |
| --- | ---: | ---: | ---: |
| `grid_50x50` | `1.01694931e-3` (multiplicity 2) | `1.48e-11`, `1.48e-12` degrees | approximately `1e-15` |
| `small_world_500` | `3.68431291e-4` (multiplicity 2) | `8.89e-12`, `1.54e-12` degrees | approximately `1e-15` |

Thus the eigenspaces and equations were correct. The divergence came from Dagua substituting
an orthonormal `eigsh` basis for the reference adapter's `eigs` right-eigenvector basis.

The disconnected `er_500` row was also checked as the likely third residual. It has a repeated
zero mode, but the corrected paired solver call closes it as well.

## Reference independence

`dagua/eval/competitors/networkx_competitor.py` contains no `dagua.layout` import. The
`nx_spectral_random_walk` adapter is necessarily a NetworkX-backed extension because NetworkX
does not expose a public random-walk normalization option for `spectral_layout`; it builds the
adjacency with NetworkX and performs the documented SciPy eigenproblem independently.

The adapter previously left ARPACK's start implicit. That made repeated-eigenspace bases depend
on process RNG state. The fix gives both independent implementations the same explicit,
process-stable start-vector rule without introducing a runtime oracle-to-Dagua dependency.

## Implementation

- `dagua/layout/ops/embed.py`: removed the random-walk symmetric-similarity substitution from
  NetworkX-fidelity mode. Sparse random-walk layouts now use the existing nonsymmetric
  `eigs(..., which="SR", ncv=..., v0=...)` path and its right eigenvectors.
- `dagua/eval/competitors/networkx_competitor.py`: stabilized the independent random-walk
  reference's `eigs` call with the same explicit start-vector rule.
- `tests/test_layout/test_spectral_fidelity.py`: replaced the symmetric-substitution regression
  expectation with exact nonsymmetric solver-setup coverage and an oracle-independence/start
  parity check.

The removed similarity-transform helpers became unreachable after the solver correction and
were deleted as part of the scoped fix.

## Fresh benchmark and score

Benchmark cache:

- `eval_output/benchmark_r79_spectral_rwfix`
- 105 graphs x 8 paired engines = 840 deterministic runs
- 840 ok, 0 skipped, 0 errors, 0 timeouts

Commands:

```bash
python scripts/run_benchmark.py --variants \
  --engines <four classic spectral variants and four paired references> \
  --graphs <the 105-graph spectral corpus> \
  --workers 4 --timeout 300 --watchdog-timeout 900 \
  --output-dir eval_output/benchmark_r79_spectral_rwfix

python scripts/definitive_fidelity_analysis.py --mode deterministic \
  --refresh-dir eval_output/benchmark_r79_spectral_rwfix \
  --combos-file /tmp/dagua_r79_combos/combos_r79_spectral.txt \
  --workers 6 --overwrite \
  --output eval_output/fidelity_definitive/per_combo_r79_spectral.jsonl
```

Target evidence:

| Graph | Verdict | Toolkit distance | Stress D/R | Crossings D/R | Neighborhood D/R |
| --- | --- | ---: | ---: | ---: | ---: |
| `grid_50x50` | invariant | `1.951e-16` | `0.0205561 / 0.0205561` | `0 / 0` | `0.7426 / 0.7426` |
| `small_world_500` | invariant | `2.189e-16` | `0.0331399 / 0.0331399` | `2455 / 2455` | `0.8982 / 0.8982` |
| `er_500` | invariant | `2.904e-16` | `0.8688004 / 0.8688004` | `126513 / 126513` | `0.0646217 / 0.0646217` |

## Scope

No changes were made to `sugiyama.py`, `fmmm.py`, `tsnet.py`, or `causes_r78.json` by this
work. Unrelated working-tree changes in those areas were excluded from the scoped commit.

## Verification

- `ruff check dagua/layout/ops/embed.py dagua/eval/competitors/networkx_competitor.py tests/test_layout/test_spectral_fidelity.py --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed with no issues.
- `pytest tests/test_layout/test_spectral_fidelity.py tests/test_pipeline_spectral.py -x --tb=short -q`: 29 passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: stopped on an unrelated
  concurrently modified Sugiyama order assertion after 121 passes.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: stopped on
  the existing Graphviz competitor mock-signature mismatch after 170 passes and 1 xfail.
- `ruff check .`: the scoped files pass; the repository-wide command reports 21 existing
  errors in untracked `.project-context/research` and `.research` scripts. Those user-owned
  files were left unchanged.
