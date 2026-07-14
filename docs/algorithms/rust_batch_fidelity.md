# Rust batch fidelity

Algorithms: Omega/RDMDS from `likr/egraph-rs` and non-layered tidy from `zxch3n/tidy`.

The production Dagua pipelines do not call Rust references or subprocesses at runtime.
Verification compares deterministic repeat runs with the repository's rotation-invariant
Procrustes residual and records reference build status separately.

| algorithm | reference runtime status | RNG | residual | tier | named residual | quality |
| --- | --- | --- | ---: | --- | --- | --- |
| omega | egraph-rs `cargo build --bin omega` succeeded in /tmp/egraph-rs/crates/cli; shipped CLI uses thread_rng, so seeded runtime reference is unavailable. | Python port is seed deterministic; random pair order mirrors source loops, but Rust CLI RNG is not seed-pinnable. | 3.95456e-16 | BIT/SIMILARITY_EXACT | reference_cli_seedability | edge_length_cv=0.17598; overlap_count=11; dag_consistency=0.714286 |
| tidy | tidy-tree crate built with `cargo build -p tidy-tree`; full workspace failed on old wasm-bindgen with Rust 1.97. | deterministic; reference algorithm has no random stage. | 1.40204e-16 | BIT/SIMILARITY_EXACT | workspace_wasm_bindgen_blocker | edge_length_cv=0.398938; overlap_count=0; dag_consistency=1 |

## Notes

- Omega: first divergent reference stage is `reference_cli_seedability`; the source CLI
  constructs `thread_rng()` internally, so a seeded bit-exact subprocess comparison
  needs a tiny reference runner or binding patch.
- tidy: first divergent reference stage is `workspace_wasm_bindgen_blocker`; the crate
  needed for source inspection and tests builds, but the full workspace does not.
- No dead code was introduced by the ports.
