# Rust batch fidelity

Algorithms: Omega/RDMDS from `likr/egraph-rs` and non-layered tidy from `zxch3n/tidy`.

The production Dagua pipelines do not call Rust references or subprocesses at runtime.
Verification compares deterministic repeat runs with the repository's rotation-invariant
Procrustes residual and records reference build status separately.

| algorithm | reference runtime status | RNG | residual | tier | named residual | quality |
| --- | --- | --- | ---: | --- | --- | --- |
| omega | egraph-rs root `cargo build --release` and patched seeded `omega` CLI succeeded. | Reference CLI patched to accept `--seed`; Python port repeat residual=2.03133e-16. | 1.0299 | DISTRIBUTIONAL | rdmds-pair-sgd-stage | edge_length_cv=0.369031; overlap_count=13; dag_consistency=0.857143 |
| tidy | tidy-tree crate and `tidy_reference` runner built with `cargo build --release --bin tidy_reference`. | deterministic; reference algorithm has no random stage; Python repeat residual=1.40204e-16. | 0.397285 | DISTRIBUTIONAL | apportion-contour-stage | edge_length_cv=0.398938; overlap_count=0; dag_consistency=1 |

## Notes

- Omega: first divergent reference stage is `rdmds-pair-sgd-stage`; the CLI now
  accepts a local `--seed` patch, so remaining residual is after reference
  RDMDS, pair construction, and SparseSGD arithmetic.
- tidy: first divergent reference stage is `apportion-contour-stage`; the runner
  calls the upstream `TidyTree::with_tidy_layout` implementation directly.
- No dead code was introduced by the ports.
