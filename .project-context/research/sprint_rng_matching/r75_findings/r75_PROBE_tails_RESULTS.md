# r75 tails probe results

Date: 2026-07-01
Repo: `/home/jtaylor/projects/dagua`, develop at `89ed3c3`
Scope: research/probe only. Scratch harness was `/tmp/r75_probe.py`; raw JSON/logs were written under `/tmp`.

No production code was modified.

## Commands

```bash
python /tmp/r75_probe.py
timeout 90s python - <<'PY' > /tmp/r75_e1e3.log 2>&1
import json, sys, pathlib
sys.path.insert(0, "/tmp")
import r75_probe
out = {}
for k, fn in [("E1", r75_probe.experiment_e1), ("E3", r75_probe.experiment_e3)]:
    print("RUN", k, flush=True)
    try:
        out[k] = fn()
    except Exception as e:
        out[k] = {"error": repr(e)}
    pathlib.Path("/tmp/r75_probe_E1E3.partial.json").write_text(json.dumps(out, indent=2))
    print("DONE", k, flush=True)
pathlib.Path("/tmp/r75_probe_E1E3.json").write_text(json.dumps(out, indent=2))
PY
timeout 150s python - <<'PY' > /tmp/r75_e2.log 2>&1
import json, sys, pathlib
sys.path.insert(0, "/tmp")
import r75_probe
pathlib.Path("/tmp/r75_probe_E2.json").write_text(
    json.dumps({"E2": r75_probe.experiment_e2()}, indent=2)
)
PY
timeout 240s python - <<'PY' > /tmp/r75_e4_slim.log 2>&1
# Slim E4: seeds 42/43, partial JSON after each row.
PY
```

Note: the first all-in-one run exposed harness bugs and was not used for verdicts. The fixed harness reruns above produced the numbers below.

## E1 - igraph Sugiyama LP objective

Installed runtime traced:

- `python-igraph`: `1.0.0`
- C extension: `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/igraph/_igraph.abi3.so`

I tested three directed DAGs with asymmetric in/out distributions. The first two are not distinguishing because both LPs land on the longest-path layering. The third distinguishes the objectives.

| graph | igraph y-layers | dagua r74 out/in prediction | IN/IN zero-objective prediction | result |
|---|---:|---:|---:|---|
| fork_join_tail | `[0,1,1,1,2,3,3]` | `[0,1,1,1,2,3,3]` | `[0,1,1,1,2,3,3]` | indistinguishable |
| asym_sources_sink | `[0,0,0,1,2,2,3,3]` | `[0,0,0,1,2,2,3,3]` | `[0,0,0,1,2,2,3,3]` | indistinguishable |
| two_hubs_bridge | `[0,0,1,2,2,0,3,4,4]` | `[0,0,1,2,2,2,3,4,4]` | `[0,0,1,2,2,0,3,4,4]` | distinguishes |

Verdict: **dagua objective WRONG** for installed igraph 1.0.0 on constructibles. The installed wheel behavior matches the IN/IN source-bug outcome on the distinguishing graph. Recommended fix: gate a change to the igraph Sugiyama LP objective behind igraph-fidelity/default Sugiyama only, and add this `two_hubs_bridge` regression because it isolates the independent-source layer difference.

## E2 - connected classical_mds eigenspace test

Method: for each connected target graph, compute the same double-centered graph-distance Gram matrix used by dagua; compare dagua's 2D embedding to installed igraph `layout("mds")`; run SciPy `eigh` drivers `evr`, `evx`, and `evd`.

| graph | lambda1 | lambda2 | lambda3 | gaps 1-2 / 2-3 | igraph-in-dagua residual | dagua vs igraph Procrustes | driver effect |
|---|---:|---:|---:|---:|---:|---:|---|
| bipartite_4_3_4 | 2.000000 | 2.000000 | 2.000000 | 0 / 0 | 0.836363 | 1.012993 | evr=evx, evd differs 1.003909 |
| center_port_backedge_hub | 2.000000 | 2.000000 | 2.000000 | 0 / 0 | 0.781031 | 0.944167 | evr=evx, evd differs 0.335794 |
| densenet_block | 3.343466 | 0.500000 | 0.500000 | 2.843466 / 0 | 0.660736 | 0.409284 | evr=evx, evd differs 0.510080 |
| org_chart_1_5_4_8 | 44.449944 | 44.449944 | 44.449944 | 0 / 0 | 0.641322 | 0.760787 | evr=evx, evd differs 0.089901 |
| petersen_10 | 3.500000 | 3.500000 | 3.500000 | 0 / 0 | 0.675406 | 0.768818 | evr=evx, evd differs 0.720279 |
| wide_single_layer_1_50_1 | 2.000000 | 2.000000 | 2.000000 | 0 / 0 | 0.997241 | 1.376248 | evr=evx, evd differs 2.000000 |
| wide_3_50_3 | 2.000000 | 2.000000 | 2.000000 | 0 / 0 | nan | 2.000000 | evr/evx/evd all differ |

Verdicts:

- `bipartite_4_3_4`: **GENUINE numerical floor**. Repeated top eigenspace; installed igraph chooses a different 2D basis than dagua, and `evd` changes SciPy's coordinates.
- `center_port_backedge_hub`: **GENUINE numerical floor**. Same pattern.
- `densenet_block`: **GENUINE numerical floor** for the second dimension. Lambda1 is isolated, but lambda2/lambda3 are exactly tied.
- `org_chart_1_5_4_8`: **GENUINE numerical floor**. Triple top tie; note this graph was already near/equivalent in the critique context.
- `petersen_10`: **GENUINE numerical floor**. Triple top tie in the measured top three; high subspace residual.
- `wide_single_layer_1_50_1`: **GENUINE numerical floor**. Complete top tie and very high residual.
- `wide_3_50_3`: **GENUINE numerical floor / degenerate edge case**. Igraph reference subspace normalization produced `nan` residual, but Procrustes and driver disagreement are maximal, with zero eigengaps.

Disposition: no deterministic basis-selection port is justified from this probe. The reference is not inside dagua's selected 2D subspace on these measurements; this is not a simple rotation/sign issue. Treat these connected rows as eigensolver/basis numerical floors unless a future port vendors the exact installed igraph/LAPACK basis path.

## E3 - maxent `random_dag_50` first divergence

Important source finding: `scripts/ogdf_runner.cpp:330-334` calls `StressMinimization::hasInitialLayout(true)`. The runner also fills every node with `std::rand() % 1000 / 10.0` at `scripts/ogdf_runner.cpp:417-422` before invoking OGDF. Therefore the runner path used by benchmarks **does not use OGDF's internal disconnected `ComponentSplitterLayout(PivotMDS)` warm start** from `StressMinimization.cpp:107-123`.

Probe graph facts for benchmark `random_dag_50`:

- `N=97`, `E=70`
- weak component sizes: 50 singleton components plus one size-45 component and one size-2 component
- OGDF disconnected distance fill used by dagua: `100 * sqrt(97) = 984.8857801796104`
- unreachable matrix entries filled: `7330`

Raw comparisons:

| comparison | value |
|---|---:|
| dagua iter0 vs runner-owned random init max abs | `3.051757815342171e-06` |
| dagua iter1 vs OGDF runner iterations=1 Procrustes | `2.727085936466163e-08` |
| dagua steps200 vs OGDF runner iterations=200 Procrustes | `3.123075155471523e-08` |

The attempted `iterations=0` runner comparison is not a true step-0 readout because the runner only calls `setIterations()` when the supplied value is positive; zero falls back to OGDF's default iteration count.

Verdict: **initialization and iterations match the benchmark runner; killed as a current fix target**. The first-divergence hypothesis based on OGDF internal PivotMDS/component placement does not apply to the installed benchmark runner. Recommended disposition: do not change maxent disconnected initialization or distance fill for `random_dag_50`; investigate scoring/reference artifact drift if the three rows still appear divergent.

## E4 - neato disconnected pack/RNG probe

Method: compare saved `eval_output/benchmark_100seed_seeded_refs/positions/*__graphviz_neato__for__classic_neato__seed*.pt` references against:

- base dagua neato path;
- monkeypatch forcing every component stress solve to reuse the same seed, approximating `component_index=0`;
- `pack=false` dagua variant.

The full 5-seed run timed out. A slim run completed both seeds for `parallel_cycles_4x5` and seed 42 for `random_dag_50`; it timed out during `random_dag_50` seed 43. All completed rows used saved references.

| graph | seed | base Proc | same-seed Proc | pack=false Proc | base stress | same-seed stress | pack=false stress | ref stress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parallel_cycles_4x5 | 42 | 0.789696 | 0.787459 | 0.949742 | 0.00806848 | 0.00806815 | 0.00823687 | 0.00806588 |
| parallel_cycles_4x5 | 43 | 0.849857 | 0.847950 | 0.946621 | 0.00806701 | 0.00806668 | 0.05757746 | 0.00806569 |
| random_dag_50 | 42 | 1.279334 | 1.330612 | 1.250949 | 0.05267782 | 0.05966738 | 0.07427403 | 0.31195889 |

Verdict: **seeding-policy KILLED as primary cause**. The same-seed monkeypatch barely moves `parallel_cycles_4x5` and worsens `random_dag_50` seed 42. `pack=false` also does not explain the saved Graphviz references: it worsens `parallel_cycles_4x5` shape and stress, and only slightly improves `random_dag_50` Procrustes while moving stress farther from dagua's base.

Recommended disposition: do not change shared component seeding based on this probe. If neato remains worth chasing, inspect Graphviz pack placement/CG solver details rather than per-component seed arithmetic.

## Assumptions and blockers

- The requested results markdown is the only repo write. Scratch code and raw data stayed in `/tmp`.
- For E1, I interpret the distinguishing installed-wheel behavior as sufficient runtime tracing because the wheel's C extension path and version are recorded; no unpinned source tree is cited as installed truth.
- E4 did not complete the full 5-seed matrix within the timeout. The three completed rows are enough to kill the seed-policy hypothesis as a primary cause, but not enough for a full statistical statement.

## Knowledge

- Installed igraph 1.0.0 Sugiyama behaves like the IN/IN objective bug on a constructed DAG that distinguishes it from dagua's r74 out/in objective.
- Connected classical MDS tails are dominated by exact eigenspace ties or second-dimension ties; SciPy `evr` and `evx` match each other, while `evd` often selects a materially different basis.
- The benchmark OGDF stress runner bypasses OGDF's internal PivotMDS warm start by always setting `hasInitialLayout(true)` and filling positions itself.
- Neato component seed arithmetic is not the main residual driver on the completed disconnected probes.
