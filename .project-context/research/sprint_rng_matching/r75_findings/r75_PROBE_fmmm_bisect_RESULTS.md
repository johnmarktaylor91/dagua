# r75 FMMM First-Divergence Bisection Probe

Date: 2026-07-01
Repo: `/home/jtaylor/projects/dagua`
OGDF source: `/home/jtaylor/tools/ogdf-src`
Mode: research/probe only

## STOP Verdict

STOPPED before bisection. The restore-contract byte reproduction check failed after rebuilding the
original `scripts/ogdf_runner` via `scripts/rng_match/build_ogdf_runner.sh`.

Per the task contract, no dagua-side bisection or seed comparison was run after this failure.

## Commands Run

Baseline capture before touching OGDF:

```bash
python - <<'PY'
import json, subprocess
from dagua.eval.graphs import get_test_graphs
from dagua.eval.competitors.ogdf_competitor import _graph_edges

graph = next(tg.graph for tg in get_test_graphs() if tg.name == 'grid_5x5')
payload = {
    'nodes': graph.num_nodes,
    'edges': _graph_edges(graph),
    'algorithm': 'fmmm',
    'seed': 42,
    'fmmmFixedIterations': 10,
}
with open('/tmp/ogdf_grid5x5_payload.json','w') as f:
    json.dump(payload, f, separators=(',', ':'))
result = subprocess.run(
    ['scripts/ogdf_runner'],
    input=json.dumps(payload),
    text=True,
    capture_output=True,
    check=True,
)
open('/tmp/ogdf_baseline_grid5x5.json','w').write(result.stdout)
print('nodes', graph.num_nodes, 'edges', len(payload['edges']), 'bytes', len(result.stdout))
PY
```

Temporary instrumented build:

```bash
cmake -S /home/jtaylor/tools/ogdf-src -B /tmp/ogdf-build-dump \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/ogdf-dump \
  -DBUILD_SHARED_LIBS=OFF \
  -DOGDF_INCLUDE_CGAL=OFF \
  -DOGDF_SEPARATE_TESTS=OFF \
  -DDOC_INSTALL=OFF
cmake --build /tmp/ogdf-build-dump --target install --parallel 8
g++ -std=c++17 -O2 /home/jtaylor/projects/dagua/scripts/ogdf_runner.cpp \
  -I/tmp/ogdf-dump/include \
  -I/tmp/ogdf-dump/include/ogdf-release \
  -L/tmp/ogdf-dump/lib \
  -lOGDF -lCOIN -pthread \
  -o /tmp/ogdf_runner_dump
/tmp/ogdf_runner_dump --help >/dev/null
```

Restore-contract sequence:

```bash
git -C /home/jtaylor/tools/ogdf-src checkout -- .
git -C /home/jtaylor/tools/ogdf-src status --short
JOBS=8 scripts/rng_match/build_ogdf_runner.sh
python - <<'PY'
import subprocess, sys
payload = open('/tmp/ogdf_grid5x5_payload.json').read()
baseline = open('/tmp/ogdf_baseline_grid5x5.json').read()
result = subprocess.run(
    ['scripts/ogdf_runner'],
    input=payload,
    text=True,
    capture_output=True,
    check=True,
)
open('/tmp/ogdf_rebuilt_grid5x5.json','w').write(result.stdout)
print('rebuilt_bytes', len(result.stdout), 'baseline_bytes', len(baseline),
      'byte_match', result.stdout == baseline)
if result.stdout != baseline:
    sys.exit(2)
PY
```

## Restore-Contract Evidence

`/home/jtaylor/tools/ogdf-src` was restored and is clean:

```text
$ git -C /home/jtaylor/tools/ogdf-src status --short
<no output>
```

The rebuilt original runner did not reproduce the saved baseline:

```text
rebuilt_bytes 248 baseline_bytes 246 byte_match False
```

Baseline output:

```json
{"positions":[[25,214],[67,218],[25,167],[116,222],[71,172],[167,224],[122,176],[214,224],[173,179],[219,181],[29,116],[75,121],[126,125],[177,127],[224,132],[33,66],[79,69],[130,73],[181,76],[227,82],[39,25],[86,25],[136,26],[186,30],[228,35]]}
```

Rebuilt output:

```json
{"positions":[[25,232],[67,228],[34,187],[117,223],[80,182],[168,220],[132,179],[215,217],[183,175],[228,175],[47,138],[93,131],[145,127],[196,124],[242,127],[67,93],[110,82],[159,76],[209,72],[252,76],[97,62],[127,37],[174,29],[221,25],[261,27]]}
```

## Instrumentation Prepared But Not Used

Built separate binary:

```text
/tmp/ogdf_runner_dump
```

The temporary OGDF instrumentation was env-gated by `OGDF_FMMM_DUMP` and was intended to write JSONL
records for `force_init`, each `force_iter`, and `postprocess_done`, including per-record OGDF RNG
draw deltas. It was not used for bisection because the restore-contract check failed.

Prepared scratch dagua probe:

```text
/tmp/r75_fmmm_bisect.py
```

It was not run after the restore failure.

## Iteration-0 Parity Verdict

Not measured. Bisection stopped before any instrumented seed run.

## First-Divergent Iteration

Not measured.

## Divergence Signature

Not measured.

## RNG-Count Comparison

Not measured.

## Boundary Events

Not measured.

## Root-Cause Hypothesis

CONFIRMED: The bisection probe cannot be trusted until the runner provenance mismatch is resolved.
The official rebuild changed the grid_5x5 seed-42 fixed-iteration-10 output relative to the
pre-instrumentation runner.

PLAUSIBLE: The pre-existing `scripts/ogdf_runner` binary was not produced by the current
`build_ogdf_runner.sh` + `/home/jtaylor/tools/ogdf-src` tag/prefix combination, or it was built
against different OGDF options/artifacts. The source checkout itself was restored cleanly.

## Minimal Gated Fix Sketch

1. Establish runner provenance before FMMM bisection:
   - Either recover the exact build inputs that produced the baseline `scripts/ogdf_runner`, or
   - bless the rebuilt runner as the new reference only after rerunning the relevant seeded
     reference rows.
2. Add a cheap byte-level runner smoke test for `grid_5x5` seed 42 fixed_iterations 10 before any
   future OGDF source instrumentation.
3. Once runner provenance is stable, rerun the prepared dump binary workflow and compare
   iteration-0 and main-loop records for seeds 42 and 44.

## Concerns

- `scripts/ogdf_runner` is now modified in the dagua worktree because the official rebuild script
  overwrote it. I did not revert it silently.
- The broader dagua worktree had unrelated dirty/untracked research files and `.pyc` files at the
  end of this probe; they were not modified intentionally for this task.

## Knowledge

- `scripts/rng_match/build_ogdf_runner.sh` checks out `/home/jtaylor/tools/ogdf-src` at
  `foxglove-202510`, installs OGDF into `/home/jtaylor/tools/ogdf`, rebuilds `scripts/ogdf_runner`,
  then runs `check_engine.py` for `classic_gem_iters100`, `classic_gem_iters500`, and
  `classic_fmmm_steps100`.
- The build-script checks currently report divergence for those engines; this did not prevent the
  runner rebuild, but the final byte-match check failed.
