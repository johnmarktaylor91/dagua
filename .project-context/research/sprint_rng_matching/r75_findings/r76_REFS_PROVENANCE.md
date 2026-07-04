# r76 Refs Provenance Probe

Date: 2026-07-04
Repo: `/home/jtaylor/projects/dagua`
HEAD during probe: `445bc6173973bc705f0b89b121d831076f6e79c2`
Reference corpus: `eval_output/benchmark_100seed_r76_refs`

## Verdict

The `random_dag_50` r76 OGDF reference tensors are not provenance-verifiable from
the committed code path. A scratch monkeypatch of the actual benchmark adapter
path shows that `scripts/run_benchmark.py` -> `VariantCompetitor` ->
`dagua/eval/competitors/ogdf_competitor.py` sends the expected runner options:

- `ogdf_fmmm__for__classic_fmmm_steps10` sends `algorithm=fmmm` and
  `fmmmFixedIterations=10`.
- `ogdf_gem__for__classic_gem_iters2000` sends `algorithm=gem` and
  `gemRounds=2000`.
- The runner argv is exactly `["/home/jtaylor/projects/dagua/scripts/ogdf_runner"]`.
- The worker environment contains `DAGUA_COMPETITOR_SEED=<seed>`.

Running the committed runner directly on the dumped adapter payload is
byte-identical to the tensor saved by the scratch adapter run. Both differ from
the stored `benchmark_100seed_r76_refs` tensor for `random_dag_50`.

The divergence is in payload construction, not variant parameter mapping and not
binary/source drift. The root payload hazard is
`dagua/eval/graphs.py:_random_dag`: it stores string edges in a Python `set` and
then calls `DaguaGraph.from_edge_list(list(edges), num_nodes=n_nodes)`. Because
the edge endpoints are strings, `list(edges)` depends on Python hash state. That
changes:

1. the order of edges inserted into `DaguaGraph`;
2. the string-node-to-integer index mapping assigned by `from_edge_list`;
3. the final numeric OGDF JSON edge list.

The topology intended by `_random_dag(50, 70, seed=42)` is fixed, but the numeric
payload actually sent to OGDF is not reproducible unless the exact process hash
state or the exact JSON payload is saved.

## Adapter Payload Evidence

Instrumented path: a scratch Python script imported `scripts/run_benchmark.py`,
monkeypatched `dagua.eval.competitors.ogdf_competitor.subprocess.run`, then
called `_run_single_work_item(...)`. No repository source was edited.

Payload for `random_dag_50 x ogdf_fmmm__for__classic_fmmm_steps10 x seed100`:

```json
{
  "nodes": 97,
  "edges": [[50, 51], [52, 53], "... 68 more ..."],
  "algorithm": "fmmm",
  "seed": 100,
  "fmmmFixedIterations": 10
}
```

Payload for `random_dag_50 x ogdf_gem__for__classic_gem_iters2000 x seed100`:

```json
{
  "nodes": 97,
  "edges": [[50, 51], [52, 53], "... 68 more ..."],
  "algorithm": "gem",
  "seed": 100,
  "gemRounds": 2000
}
```

The surprising `nodes=97` is also part of the current adapter path. It comes
from passing `num_nodes=50` to `DaguaGraph.from_edge_list(...)` while the edge
IDs are strings like `"n17"`; DaguaGraph preallocates integer nodes `0..49` and
then appends string nodes as they first appear.

## Comparison Table

Adapter means the tensor saved by `_run_single_work_item` in `/tmp`.
Direct means committed `scripts/ogdf_runner` rerun on the captured JSON payload.
Stored means the existing tensor under `eval_output/benchmark_100seed_r76_refs`.

| Graph | Engine | Seed | Adapter vs direct byte | Direct vs stored byte | Raw RMSD | Procrustes RMSD | Payload SHA-256 |
|---|---|---:|---|---|---:|---:|---|
| random_dag_50 | ogdf_fmmm__for__classic_fmmm_steps10 | 100 | yes | no | 111.535 | 0.0536 | df0780e05df5 |
| random_dag_50 | ogdf_fmmm__for__classic_fmmm_steps10 | 101 | yes | no | 177.071 | 0.0699 | 4884e84aa244 |
| random_dag_50 | ogdf_fmmm__for__classic_fmmm_steps10 | 102 | yes | no | 134.511 | 0.0611 | 81db99142892 |
| random_dag_50 | ogdf_gem__for__classic_gem_iters2000 | 100 | yes | no | 409.904 | 0.0776 | 3d9295c1400d |
| random_dag_50 | ogdf_gem__for__classic_gem_iters2000 | 101 | yes | no | 364.242 | 0.0792 | 5e61d42d2e5d |
| random_dag_50 | ogdf_gem__for__classic_gem_iters2000 | 102 | yes | no | 385.311 | 0.0772 | 6ec2a7ad21b8 |

One-seed sweep across all six stored `random_dag_50` OGDF variants:

| Graph | Engine | Seed | Stored match | Raw RMSD |
|---|---|---:|---|---:|
| random_dag_50 | ogdf_fmmm__for__classic_fmmm_steps10 | 100 | no | 179.974 |
| random_dag_50 | ogdf_fmmm__for__classic_fmmm_steps100 | 100 | no | 187.067 |
| random_dag_50 | ogdf_fmmm__for__classic_fmmm_steps200 | 100 | no | 185.860 |
| random_dag_50 | ogdf_gem__for__classic_gem_iters100 | 100 | no | 114.696 |
| random_dag_50 | ogdf_gem__for__classic_gem_iters500 | 100 | no | 126.714 |
| random_dag_50 | ogdf_gem__for__classic_gem_iters2000 | 100 | no | 354.793 |

Controls from the same corpus:

| Graph | Engine | Seed | Stored match | Raw RMSD |
|---|---|---:|---|---:|
| deep_chain_20 | ogdf_fmmm__for__classic_fmmm_steps100 | 100 | yes | 0.0 |
| grid_5x5 | ogdf_gem__for__classic_gem_iters100 | 100 | yes | 0.0 |

## Payload Diff vs Manual Reconstruction

The manual reconstruction used by the r76 runner parity probe had the same
top-level fields and the same intended variant parameters, but not the same
numeric graph payload. Example manual payload from `/tmp/runner-parity`:

```text
nodes=97
algorithm=fmmm
seed=100
fmmmFixedIterations=10
first edges: [[50, 51], [52, 53], [54, 55], [56, 57], [58, 59], ...]
```

An adapter-path run in another fresh Python process produced:

```text
nodes=97
algorithm=fmmm
seed=100
fmmmFixedIterations=10
first edges: [[50, 51], [52, 53], [54, 53], [55, 56], [57, 58], ...]
```

A separate `PYTHONHASHSEED` probe confirmed the payload hash and numeric edge
list vary even though `_random_dag(..., seed=42)` is unchanged:

| PYTHONHASHSEED | Payload SHA-256 prefix | First edges |
|---:|---|---|
| 0 | 88dda8f9ebe3 | `[[50, 51], [52, 53], [54, 55], [56, 57]]` |
| 1 | 4ae3ac3b7ba3 | `[[50, 51], [52, 53], [54, 55], [56, 57]]` |
| 42 | 67f6bc1b452b | `[[50, 51], [52, 53], [54, 55], [51, 56]]` |
| 100 | 584d899108b6 | `[[50, 51], [52, 53], [54, 53], [55, 51]]` |
| 101 | d9e764bbf2d | `[[50, 51], [52, 51], [53, 54], [55, 56]]` |
| 123 | 03a41b77b6f1 | `[[50, 51], [52, 53], [54, 55], [56, 57]]` |
| 999 | e73aaff375de | `[[50, 51], [52, 53], [50, 54], [55, 56]]` |

So the manual reconstruction missed a required provenance input: the exact JSON
payload, including numeric node mapping and edge order. This also means MAAR
attempt-3 comparisons that reused the same manual-payload assumption were not a
valid gate against the stored corpus. They compared against a plausible OGDF
payload, not necessarily the stored r76 reference payload.

## Drift Check

The stored corpus manifest says:

```text
git_sha: 515a1ee23c9f5332fbe1f8ed30369f7f5a2fd225
workers: 8
timeout: 1800
seeds: 100
seed_start: 100
seed_refs: [ogdf_fmmm, ogdf_gem]
max_nodes: 300
engines: ogdf_fmmm,ogdf_gem
graphs: None
variants: True
save_positions: True
```

Relevant history since 2026-07-02:

```text
8b43153 2026-07-03 23:41:53 -0400 test(eval): add oracle invariant guardrails
```

Diff from corpus SHA to current HEAD across relevant files:

```text
scripts/run_benchmark.py | 136 +++++++++++++++++++++++++++++++++++++++++++++--
```

That later change adds row-count guardrails and explicit max-node reporting. It
does not change `_run_single_work_item`, `_build_engine_instance`, the OGDF
adapter, variant mappings, `_random_dag`, `DaguaGraph.from_edge_list`, or the
runner. There is no evidence that source drift after 2026-07-03 explains the
random_dag_50 mismatch.

Runner and library hashes during this probe:

```text
scripts/ogdf_runner      1953eb397296db93af4f3cf399196ee5719f42acc200805e4d4675723e0db335
scripts/ogdf_runner.cpp  d82fe7e757457dcb628a755c4fcc4975ed4deb146f4a01ab03536f4fbb4d4f70
libOGDF.a               7c5626e1b2680583c1bd265eb8f4ac29700fe7c4a6efa6e6d66706c579136641
libCOIN.a               1e58001328f27da976977f179687f65d01bb91d1f518aeaa9a44a791220d3f1e
OGDF source             5b6795655399b9d8e2921afec9d97bab9107d5ee (foxglove-202510)
```

## Affected Corpus List

Confirmed affected slice:

```text
eval_output/benchmark_100seed_r76_refs/positions/random_dag_50__ogdf_fmmm__for__classic_fmmm_steps10__seed100.pt ... seed199.pt
eval_output/benchmark_100seed_r76_refs/positions/random_dag_50__ogdf_fmmm__for__classic_fmmm_steps100__seed100.pt ... seed199.pt
eval_output/benchmark_100seed_r76_refs/positions/random_dag_50__ogdf_fmmm__for__classic_fmmm_steps200__seed100.pt ... seed199.pt
eval_output/benchmark_100seed_r76_refs/positions/random_dag_50__ogdf_gem__for__classic_gem_iters100__seed100.pt ... seed199.pt
eval_output/benchmark_100seed_r76_refs/positions/random_dag_50__ogdf_gem__for__classic_gem_iters500__seed100.pt ... seed199.pt
eval_output/benchmark_100seed_r76_refs/positions/random_dag_50__ogdf_gem__for__classic_gem_iters2000__seed100.pt ... seed199.pt
```

Total confirmed affected tensors: 600.

This probe does not prove the other 52,800 r76 tensors are bad. Two controls
matched byte-exactly. But any graph constructed through unordered string-edge
sets should be considered suspect until payload hashes exist.

## Implications

FMMM/GEM/MAAR verdicts scored against r76 refs are not sound for the affected
`random_dag_50` rows as an OGDF-oracle claim. They may be comparisons against a
fixed stored tensor artifact, but that artifact lacks the payload provenance
needed to establish it as the intended OGDF reference.

Required action:

1. Stabilize `random_dag_50` construction before regenerating references. The
   conservative fix is to make `_random_dag` deterministic at the payload level,
   for example by sorting the string edge set before `from_edge_list`, or by
   building a numeric `edge_index` directly with `num_nodes=n_nodes`.
2. Regenerate and rescore the 600 confirmed affected `random_dag_50` OGDF
   reference tensors.
3. Do not use manual reconstruction without payload hashes as a hard oracle
   gate for MAAR, FMMM, or GEM.

## Validator Tripwire Spec

Every saved reference tensor should have a sidecar or manifest entry containing:

- `position_file`: relative tensor path.
- `tensor_sha256`: SHA-256 over the saved tensor bytes.
- `payload_json_sha256`: SHA-256 over the exact JSON string sent to the runner.
- `payload_canonical_sha256`: SHA-256 over `json.dumps(json.loads(payload), sort_keys=True)`.
- `runner_argv`: exact argv list.
- `runner_binary_sha256`: SHA-256 of `scripts/ogdf_runner`.
- `runner_source_sha256`: SHA-256 of `scripts/ogdf_runner.cpp`.
- `ogdf_source_commit`: OGDF source git commit.
- `ogdf_source_describe`: OGDF source tag/dirty string.
- `libOGDF_sha256` and `libCOIN_sha256`.
- `python_version` and `PYTHONHASHSEED`.
- `benchmark_git_sha` and dirty-status flag.
- `graph_name`, `engine_name`, `seed`, `variant_params`, `num_nodes`, `num_edges`.

Validator behavior:

1. Recompute all hashes before comparing tensors.
2. Refuse oracle validation if any runner, source, library, or payload hash is
   missing for a stochastic reference tensor.
3. On mismatch, report whether drift is in payload, runner binary, OGDF library,
   or tensor bytes.

## Commands Used

Read prior finding and code paths:

```bash
sed -n '1,240p' .project-context/research/sprint_rng_matching/r75_findings/r76_RUNNER_PARITY.md
sed -n '1,520p' dagua/eval/competitors/ogdf_competitor.py
sed -n '990,1120p' scripts/run_benchmark.py
sed -n '540,660p' dagua/eval/competitors/classic_competitor.py
sed -n '1020,1110p' dagua/eval/variants.py
```

Inspect corpus and history:

```bash
python - <<'PY'
import json, collections
p='eval_output/benchmark_100seed_r76_refs/results.json'
with open(p) as f:
    data=json.load(f)
rows=[v for v in data.values() if v.get('graph_name')=='random_dag_50' and v.get('engine_name','').startswith('ogdf_')]
print(len(rows))
print(collections.Counter(r['engine_name'] for r in rows))
PY

git log --since='2026-07-02' --date=iso --pretty=format:'%h %ad %s' -- \
  dagua/eval/graphs.py dagua/graph.py dagua/eval/competitors/ogdf_competitor.py \
  dagua/eval/variants.py scripts/run_benchmark.py scripts/ogdf_runner.cpp scripts/ogdf_runner

git diff --stat 515a1ee23c9f5332fbe1f8ed30369f7f5a2fd225..HEAD -- \
  dagua/eval/competitors/ogdf_competitor.py dagua/eval/variants.py \
  dagua/eval/graphs.py scripts/run_benchmark.py scripts/ogdf_runner.cpp scripts/ogdf_runner
```

Hash-seed payload probe:

```bash
for s in 0 1 42 100 101 123 999; do
  PYTHONHASHSEED=$s python - <<'PY'
import json, hashlib
from dagua.eval.graphs import _random_dag
from dagua.eval.competitors.ogdf_competitor import _graph_edges
g=_random_dag(50,70,seed=42)
p={'nodes':g.num_nodes,'edges':_graph_edges(g),'algorithm':'fmmm','seed':100,'fmmmFixedIterations':10}
print(hashlib.sha256(json.dumps(p).encode()).hexdigest(), p['edges'][:4])
PY
done
```

No project tests were run because this was a read-only provenance probe plus
this markdown output file.
