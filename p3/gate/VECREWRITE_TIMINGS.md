# VECREWRITE timing and identity evidence

Measured 2026-08-23 with Python 3.11, PyTorch 2.8.0 CPU float64, after module
import and scene ingestion. Each cell is one wall-clock run in seconds. Every
old/new pair asserted the same 64-bit prefix of a canonical SHA-256 record whose
leaves preserve the exact IEEE-754 bytes and container types of every published
facet field. Acceptance never used a tolerance.

The U07 scene has `N/2` parallel, nonintersecting routed edges. This isolates the
quadratic crossing sweep without making the unchanged exact event-density
formula dominate the measurement. The U11 scene is the corresponding proof-bank
drawing: 12 routed crossing edges plus remote spectator nodes, so the node-box
obstacle census grows through the 2,000-node band.

| Size band | Nodes | U07 old | U07 vector | Speedup | U11 old | U11 vector | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `le30` | 24 | 0.013345 | 0.001612 | 8.28x | 0.048264 | 0.056968 | 0.85x |
| `31-100` | 64 | 0.070328 | 0.002368 | 29.69x | 0.073612 | 0.064765 | 1.14x |
| `101-300` | 192 | 0.430945 | 0.275494 | 1.56x | 0.118712 | 0.030458 | 3.90x |
| `301-1000` | 800 | 5.851001 | 2.573678 | 2.27x | 0.241986 | 0.097237 | 2.49x |
| `1001-3000` | 2,000 | 35.807388 | 4.898744 | 7.31x | 0.642071 | 0.169841 | 3.78x |

The obstacle-heavy 50-node U11 stress drawing (seeded tangled tree, 49 routed
edges) measured 65.499 s old versus 13.512 s vectorized, a 4.85x speedup. The
original 26-scene 71f84569 battery measured 6.312 s old versus 1.462 s
vectorized across both facets, a combined 4.32x speedup.

The small-band U11 result is an expected fixed-cost crossover: the flag retains
the scalar obstacle census below 128 boxes and cannot amortize the additional
dispatch on a 24-node drawing. The flag remains default-off pending review.
