# r76 OGDF Runner Binary-vs-Source Parity

Date: 2026-07-04
Worktree: `/home/jtaylor/projects/dagua`
OGDF source: `/home/jtaylor/tools/ogdf-src`
Temporary build dir: `/tmp/runner-parity`

## Verdict

Verdict on the narrow binary/source question: **A, with a caveat**.

The committed `scripts/ogdf_runner` is byte-identical to a fresh build from the
committed `scripts/ogdf_runner.cpp` using the documented r75 rebuild recipe and
the installed OGDF libraries under `/home/jtaylor/tools/ogdf`. The r76 attempt-3
local source-build-vs-binary comparison was therefore not evidence of
committed-binary/source drift.

Important caveat: the additional corpus sanity check below found that selected
`eval_output/benchmark_100seed_r76_refs` tensors for `random_dag_50` do **not**
match either the committed binary or the fresh source build under the matched
variant payloads. That is a reference-provenance problem distinct from
binary/source drift. Do not infer "all refs are fine" from this parity result.

## Build Recipe Recovered

Primary recipe source:

- Commit `0817427473692b9c7d406c2bd9c7bea6f983bc98`
  (`fix(bench): rebuild ogdf_runner from committed source (stale binary ignored
  iteration params)`)
- Script: `scripts/rng_match/build_ogdf_runner.sh`
- README: `scripts/rng_match/OGDF_RUNNER_README.md`

Recipe:

```text
OGDF_TAG=foxglove-202510
OGDF_SRC=/home/jtaylor/tools/ogdf-src
OGDF_BUILD=/home/jtaylor/tools/ogdf-build
OGDF_PREFIX=/home/jtaylor/tools/ogdf

cmake -S "$OGDF_SRC" -B "$OGDF_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$OGDF_PREFIX" \
  -DBUILD_SHARED_LIBS=OFF \
  -DOGDF_INCLUDE_CGAL=OFF \
  -DOGDF_SEPARATE_TESTS=OFF \
  -DDOC_INSTALL=OFF
cmake --build "$OGDF_BUILD" --target install --parallel 8

g++ -std=c++17 -O2 scripts/ogdf_runner.cpp \
  -I/home/jtaylor/tools/ogdf/include \
  -I/home/jtaylor/tools/ogdf/include/ogdf-release \
  -L/home/jtaylor/tools/ogdf/lib \
  -lOGDF -lCOIN -pthread \
  -o scripts/ogdf_runner
```

For this probe I used the same compile command but wrote the binary to
`/tmp/runner-parity/ogdf_runner_fresh` to avoid touching `scripts/ogdf_runner`.

## Binary and Library Evidence

Committed binary:

```text
file scripts/ogdf_runner:
ELF 64-bit LSB shared object, x86-64, dynamically linked, not stripped
BuildID[sha1]=a401fca0b28391b64504a579a86d9630cc5293fc

sha256:
1953eb397296db93af4f3cf399196ee5719f42acc200805e4d4675723e0db335

.comment:
GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
```

Fresh build:

```text
file /tmp/runner-parity/ogdf_runner_fresh:
ELF 64-bit LSB shared object, x86-64, dynamically linked, not stripped
BuildID[sha1]=a401fca0b28391b64504a579a86d9630cc5293fc

sha256:
1953eb397296db93af4f3cf399196ee5719f42acc200805e4d4675723e0db335
```

The committed binary and fresh source build are byte-identical.

OGDF checkout:

```text
git -C /home/jtaylor/tools/ogdf-src status --short
# no output

git -C /home/jtaylor/tools/ogdf-src rev-parse HEAD
5b6795655399b9d8e2921afec9d97bab9107d5ee

git -C /home/jtaylor/tools/ogdf-src describe --tags --always --dirty
foxglove-202510
```

Library drift check:

- `/home/jtaylor/tools/ogdf/lib/libOGDF.a`: `2026-07-01 21:03:37`
- `/home/jtaylor/tools/ogdf-build/libOGDF.a`: `2026-07-01 21:03:37`
- latest sampled OGDF object mtimes were also `2026-07-01` or earlier
- no evidence found of OGDF library modification after `2026-07-02`

## Payload Method

Payloads were generated once in Python using the benchmark graph fixtures and
the OGDF adapter serialization helper:

- Graphs from `dagua.eval.graphs.get_test_graphs()`
- Edge serialization from `dagua.eval.competitors.ogdf_competitor._graph_edges`
- JSON serialization via `json.dumps(payload)`
- Identical payload string fed to `scripts/ogdf_runner` and
  `/tmp/runner-parity/ogdf_runner_fresh`

Variant parameters used:

- `classic_fmmm_steps10` -> `{"fmmmFixedIterations": 10}`
- `classic_fmmm_steps100` -> `{"fmmmFixedIterations": 100}`
- `classic_gem_iters100` -> `{"gemRounds": 100}`
- `classic_gem_iters2000` -> `{"gemRounds": 2000}`

Payloads and stdout captures were written under `/tmp/runner-parity`.

## Parity Table

The prompt called this "6 probes" but listed four graph/variant combinations
across seeds `100` and `101`, so I ran all eight resulting layouts.

| Graph | Variant | Seed | Byte equal | Raw RMSD | Procrustes RMSD | Payload SHA-256 prefix |
|---|---|---:|---|---:|---:|---|
| `random_dag_50` | `fmmm_steps10` | 100 | yes | `0.0e+00` | `3.521e-16` | `5340daa40756` |
| `random_dag_50` | `fmmm_steps10` | 101 | yes | `0.0e+00` | `2.486e-16` | `97275707a554` |
| `random_dag_50` | `gem_iters2000` | 100 | yes | `0.0e+00` | `1.074e-16` | `f1df61f05fd9` |
| `random_dag_50` | `gem_iters2000` | 101 | yes | `0.0e+00` | `2.308e-16` | `260892792762` |
| `deep_chain_20` | `fmmm_steps100` | 100 | yes | `0.0e+00` | `1.616e-16` | `dc2b900fa9f6` |
| `deep_chain_20` | `fmmm_steps100` | 101 | yes | `0.0e+00` | `1.697e-16` | `ee19f009512c` |
| `grid_5x5` | `gem_iters100` | 100 | yes | `0.0e+00` | `1.717e-16` | `9a12273f22a1` |
| `grid_5x5` | `gem_iters100` | 101 | yes | `0.0e+00` | `1.666e-16` | `046c084f84f0` |

All committed-binary and fresh-build stdout payloads had identical SHA-256
hashes per row.

## Reference Corpus Sanity Check

The selected sample tensors exist in
`eval_output/benchmark_100seed_r76_refs/positions` and have mtimes around
`2026-07-03 00:05` to `00:10`.

Comparing those tensors to the fresh-build stdout for the same matched
payloads gave:

| Graph | Variant | Seed | Ref status | Raw RMSD vs ref | Procrustes RMSD vs ref |
|---|---|---:|---|---:|---:|
| `random_dag_50` | `fmmm_steps10` | 100 | ok | `1.414e+02` | `9.712e-01` |
| `random_dag_50` | `fmmm_steps10` | 101 | ok | `1.610e+02` | `1.019e+00` |
| `random_dag_50` | `gem_iters2000` | 100 | ok | `3.463e+02` | `1.069e+00` |
| `random_dag_50` | `gem_iters2000` | 101 | ok | `3.599e+02` | `1.047e+00` |
| `deep_chain_20` | `fmmm_steps100` | 100 | ok | `0.000e+00` | `1.616e-16` |
| `deep_chain_20` | `fmmm_steps100` | 101 | ok | `0.000e+00` | `1.697e-16` |
| `grid_5x5` | `gem_iters100` | 100 | ok | `3.447e-06` | `6.177e-08` |
| `grid_5x5` | `gem_iters100` | 101 | ok | `3.471e-06` | `6.054e-08` |

I also checked `random_dag_50` against runner defaults with only `algorithm` and
`seed` in the payload. Defaults did not match those refs either:

| Graph | Variant label | Seed | Default-param raw RMSD vs ref | Default-param Procrustes RMSD vs ref |
|---|---|---:|---:|---:|
| `random_dag_50` | `fmmm_steps10` | 100 | `1.127519e+02` | `7.426316e-01` |
| `random_dag_50` | `fmmm_steps10` | 101 | `1.707890e+02` | `9.884694e-01` |
| `random_dag_50` | `gem_iters2000` | 100 | `3.011217e+02` | `8.925233e-01` |
| `random_dag_50` | `gem_iters2000` | 101 | `8.539487e+02` | `1.098215e+00` |

Implication: the r76 attempt-3 source-build-vs-stored-ref failure on
`random_dag_50` is reproducible against the current committed binary too. The
root cause is not binary/source drift; it is either reference provenance,
payload construction, graph selection, or another corpus-generation detail not
captured by the current adapter serialization.

## Bisect Status

No B-style bisect was needed because the committed binary and committed source
build are byte-identical. I did not build runner source at `0817427` separately:
the current source-built binary already matches the current committed binary
exactly, including BuildID and SHA-256.

## Recommendation

1. Treat `scripts/ogdf_runner` and committed `scripts/ogdf_runner.cpp` as in
   parity for this worktree and OGDF installation.
2. Do not redo the r76 trace as "source build vs committed binary" debugging;
   that path is closed.
3. Add a runner tripwire before future corpus generation:
   - binary SHA-256
   - source SHA-256
   - OGDF source tag and commit
   - `libOGDF.a` and `libCOIN.a` SHA-256
   - JSON payload SHA-256 per tensor
4. Open a separate reference-provenance probe for `random_dag_50` refs:
   identify the exact generator script/process that wrote
   `benchmark_100seed_r76_refs`, because those selected tensors do not match
   current committed runner output even though binary/source parity is exact.

## Concerns

- The prompt's A implication says the refs are fine if binary/source parity
  holds. The sampled `random_dag_50` tensors contradict that implication in
  this worktree.
- The working tree had unrelated dirty/untracked files before this report,
  including pyc changes and r76 scratch logs. I ignored them and only added
  this markdown file.
- No project tests were run because this was a pure probe with no source-code
  changes.
