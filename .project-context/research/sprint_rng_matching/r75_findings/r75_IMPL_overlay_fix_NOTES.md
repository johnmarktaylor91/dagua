# r75 Overlay Fix Implementation Notes

## Bug

`scripts/definitive_fidelity_analysis.py::load_results_multi()` used per-record-key overlay
semantics across repeated `--data-dir` roots. Because older benchmark roots contained seed rows
such as 42-99 and newer roots contained seed rows such as 100-199, the merged result set could
combine rows from different code eras for one `(graph, engine)` combo. That made downstream
quality batteries average pre-fix and post-fix layout rows together.

## Fix

`load_results_multi()` now resolves overlay precedence per `(graph, engine)` combo:

- each input directory is grouped by combo;
- a directory is eligible to win a combo only if it has at least one `status == "ok"` row;
- the last eligible directory in `--data-dir` order wins;
- all surviving rows for that combo come only from the winning directory;
- older directories still supply combos absent from newer directories;
- reference engines such as `igraph_mds__for__...` use the same combo rule.

The loader still absolutizes `positions_file` against the source benchmark root and preserves
`source_dir` on each row. It also prints a load-time audit summary. In the 11-directory MDS rescore
chain, the fixed loader printed:

```text
overlay: 8756 combos resolved, 4502 would have era-mixed under union semantics
```

## Test Evidence

Commands run:

```text
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_definitive_fidelity_overlay.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -k "overlay or results or analysis" -x -q
```

Results:

```text
ruff check . --fix
All checks passed!

mypy --follow-imports=silent dagua/cli.py
pyproject.toml: note: unused section(s): module = ['dagua.layout.multilevel']
Success: no issues found in 1 source file

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_definitive_fidelity_overlay.py -q
4 passed, 3 warnings in 0.68s

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -k "overlay or results or analysis" -x -q
35 passed, 3115 deselected, 74 warnings in 440.63s (0:07:20)
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` was needed because the ambient environment auto-loads a broken
external `zarr` pytest plugin and fails during plugin import before project tests are collected.

## Rescore Evidence

Command started from the worktree with `PYTHONPATH=$PWD`, reading benchmark roots from
`/home/jtaylor/projects/dagua/eval_output` and writing `/tmp/r75_mds_rescore_fixed.jsonl`.

The run completed 18 of 30 rows, then stayed flat for over 30 minutes with all four worker
processes alive but near-idle in futex waits. I terminated that verification run rather than
leave worker processes running indefinitely. The decisive sentinel row completed and showed the
expected drop:

```text
parallel_cycles_4x5::classic_classical_mds_igraph_fidelity
before battery_stress_D_mean: 1.0
after  battery_stress_D_mean: 0.011022762500039354
before quality_identical_raw: False
after  quality_identical_raw: True
```

Completed-row false-to-true flip count: 6 of 18 completed rows.

| combo_id | before D | after D | before identical | after identical |
| --- | ---: | ---: | --- | --- |
| bipartite_4_3_4::classic_classical_mds_default | 0.24980721420860652 | 0.22733072392922907 | False | False |
| bipartite_4_3_4::classic_classical_mds_igraph_fidelity | 0.24980721420860652 | 0.2498072142086065 | False | False |
| center_port_backedge_hub::classic_classical_mds_default | 0.17914548919708936 | 0.14832014421028597 | False | False |
| center_port_backedge_hub::classic_classical_mds_igraph_fidelity | 0.17914548919708936 | 0.1791454891970893 | False | False |
| densenet_block::classic_classical_mds_igraph_fidelity | 0.19967600759210852 | 0.19967600759210855 | False | False |
| densenet_block::classic_classical_mds_default | 0.19967600759210852 | 0.19701081153078293 | False | False |
| disconnected_encoder_residual::classic_classical_mds_default | 0.35044014508202215 | 0.024788377200399787 | False | True |
| disconnected_encoder_residual::classic_classical_mds_igraph_fidelity | 0.35044014508202215 | 0.024788377200399787 | False | True |
| disconnected_label_cycle_collage::classic_classical_mds_default | 1.0 | 2.5780523360082355e-06 | False | False |
| disconnected_label_cycle_collage::classic_classical_mds_igraph_fidelity | 1.0 | 2.5780523360082355e-06 | False | False |
| kitchen_sink_platform_graph::classic_classical_mds_igraph_fidelity | 0.1098906778511548 | 0.06133110060915265 | False | True |
| kitchen_sink_platform_graph::classic_classical_mds_default | 0.1098906778511548 | 0.06133110060915265 | False | True |
| org_chart_1_5_4_8::classic_classical_mds_default | 0.14733136540131558 | 0.1597055478121375 | False | False |
| org_chart_1_5_4_8::classic_classical_mds_igraph_fidelity | 0.14733136540131558 | 0.14733136540131558 | False | False |
| parallel_cycles_4x5::classic_classical_mds_default | 1.0 | 0.011022762500039354 | False | True |
| parallel_cycles_4x5::classic_classical_mds_igraph_fidelity | 1.0 | 0.011022762500039354 | False | True |
| petersen_10::classic_classical_mds_default | 0.1681646874525052 | 0.1547985345259336 | False | False |
| petersen_10::classic_classical_mds_igraph_fidelity | 0.1681646874525052 | 0.16816468745250526 | False | False |

Incomplete after the stopped verification run:

```text
multi_component_80::classic_classical_mds_default
multi_component_80::classic_classical_mds_igraph_fidelity
random_bipartite_60::classic_classical_mds_igraph_fidelity
random_bipartite_60::classic_classical_mds_default
random_dag_50::classic_classical_mds_default
random_dag_50::classic_classical_mds_igraph_fidelity
random_dag_200::classic_classical_mds_default
random_dag_200::classic_classical_mds_igraph_fidelity
wide_single_layer_1_50_1::classic_classical_mds_default
wide_single_layer_1_50_1::classic_classical_mds_igraph_fidelity
wide_3_50_3::classic_classical_mds_default
wide_3_50_3::classic_classical_mds_igraph_fidelity
```

## Assumptions

- A winning directory contributes all rows for the combo from that directory, not only the `ok`
  rows. Downstream row resolution already filters usable rows; preserving same-directory non-ok
  diagnostics keeps the loader conservative while preventing cross-era union.
- A combo with no `ok` row in any directory has no winning directory and is omitted from the
  merged loader output. This matches the approved definition that a directory "has" a combo only
  with at least one usable `ok` row.

## Controversial Choices

- The overlay summary counts combos with `ok` rows in more than one source directory as "would
  have era-mixed under union semantics." This is a structural count; it does not try to infer code
  version labels from directory names.

## Concerns

- The full 30-combo rescore did not complete in this environment after the first 18 rows. The
  loader sentinel passed, but the final flip count for all 30 combos remains unverified by this
  run.

## Knowledge

- The 11-directory r75 MDS rescore chain has 4502 combos that would have mixed eras under the old
  per-record-key union semantics.
- `parallel_cycles_4x5::classic_classical_mds_igraph_fidelity` is a good sentinel for this bug:
  its stress D dropped from `1.0` in `r74_phase2_rescore.jsonl` to `0.011022762500039354` with
  per-combo freshest-dir overlay.

## Commit

Commit SHA: 391618211f337de98a57119c77921c5bd6975435
