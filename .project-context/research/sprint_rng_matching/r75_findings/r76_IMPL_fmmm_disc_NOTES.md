# r76-B2 FMMM Disconnected Attempt Notes

Date: 2026-07-03
Branch: `r76/fmmm-disconnected`
Status: resisted on attempt 1; no implementation commit made.

## First Divergence Finding

The first confirmed divergence is not the component-local FMMM solve for the
large disconnected pieces. On `random_dag_50`, seed 100, steps10:

| Scope | RMSD |
|---|---:|
| Whole graph | `0.083950565` |
| 45-node component only | `0.001055270` |
| 2-node component only | `3.35e-16` |

This means the structural error is dominated by disconnected-component packing,
especially placement/order of many singleton rectangles, not by the internal
layout of the nontrivial components.

Source checks:

- OGDF runner creates graph nodes in index order.
- `FMMMLayout::make_simple_loopfree()` creates the reduced graph in original
  node order and only deletes loops/parallel edges.
- `FMMMLayout::call_DIVIDE_ET_IMPERA_step()` computes connected components on
  the reduced graph, lays subgraphs out, then calls `pack_subGraph_drawings()`.
- Contrary to the task prompt, OGDF FMMM default initial placement does not use
  one continuous random stream across components: `Multilevel::create_multilevel_representations()`
  calls `setSeed(rand_seed)`, and `create_initial_placement()` calls
  `setSeed(randSeed())` for `RandomRandIterNr`.

Component shapes from Dagua's split match the probe shape:

| Graph | Components |
|---|---|
| `random_dag_50` | 52 components: 50 singleton, 45-node, 2-node |
| `random_dag_200` | 202 components: 200 singleton, 181-node, 2-node |
| `multi_component_80` | 7 components: 40, 20, 10, 5, 3, 1, 1 |

## Fixes Tried

1. Added OGDF's `stepsForRotatingComponents()==10` pre-packing component
   rotation for disconnected components.
   - Result: mixed/worse. `random_dag_50` remained around `0.095-0.104`;
     `multi_component_80` worsened on sampled seeds; not kept.
2. Ported OGDF `Array::quicksort` tie behavior for MAAR decreasing-height
   presort.
   - Result: improved `random_dag_200` to about `0.036-0.038` in one probe,
     but worsened or failed `random_dag_50`; not sufficient.
3. Added OGDF `PairingHeap` row tie behavior approximation for MAAR row
   selection, choosing the most recently queued shortest row.
   - Result: still failed the gate. `random_dag_50` stayed about `0.084-0.092`,
     `multi_component_80` unchanged, `random_dag_200` about `0.041-0.042`.

All partial code changes were reverted because they did not materially improve
at least 3 of 4 disconnected target graphs.

## Before/After RMSD Evidence

Baseline from r76 probe:

| Graph | steps10 RMSD |
|---|---:|
| `random_dag_50` | `~0.099` |
| `multi_component_80` | `~0.090` |
| `kitchen_sink_platform_graph` | `~0.057` |
| `random_dag_200` | `~0.051` |

Best partial local probes from this attempt:

| Patch | Graph/seeds | RMSD |
|---|---|---:|
| component rotation | `random_dag_50` seeds 100-104 | `0.0946, 0.1036, 0.1002, 0.0995, 0.0951` |
| component rotation | `multi_component_80` seeds 100-102 | `0.1020, 0.0917, 0.0933` |
| component rotation | `random_dag_200` seeds 100-102 | `0.0541, 0.0517, 0.0527` |
| OGDF quicksort presort | `random_dag_200` seeds 100-102 | `0.0375, 0.0364, 0.0369` |
| quicksort + row tie | `random_dag_50` seeds 100-104 | `0.0916, 0.0836, 0.0864, 0.0912, 0.0911` |
| quicksort + row tie | `multi_component_80` seeds 100-102 | `0.0967, 0.0902, 0.0757` |
| quicksort + row tie | `random_dag_200` seeds 100-102 | `0.0410, 0.0420, 0.0412` |

No patch reached `<0.01` on `random_dag_50`, and no patch gave material
improvement on at least 3 of the 4 target disconnected graphs.

## Verification Run

Targeted FMMM tests passed while testing the partial patches:

```text
15 passed, 3 warnings in 8.74s
```

`ruff check . --fix` passed during the attempt.

Final source state after reverting partial fixes:

- No source implementation changes.
- Notes file only.
- No commit.

## Concerns / Next Attempt

The remaining likely root is exact MAAR packing internals, not component-local
FMMM. The next attempt should instrument OGDF `MAARPacking::export_new_rectangle_positions()`
directly and dump, for `random_dag_50` seed 100:

- rectangle order after `presort_rectangles_by_height()`;
- row chosen for every rectangle;
- final `new_dlc_position`, `old_dlc_position`, and `is_tipped_over`;
- exported node coordinates before final `adjust_positions()`.

Then compare those rows with Dagua's `_ogdf_maar_pack_component_transforms()`.
The evidence so far says guessing at quicksort/PQueue behavior is not enough.

## Commit

No commit was created because the implementation gate was not met.

## Attempt 2: instrumented MAAR trace

Date: 2026-07-03
Branch: `r76/fmmm-disconnected`
Status: honest parked; no implementation commit.

### Source-Pinned Tie Rules

OGDF source pin: `/home/jtaylor/tools/ogdf-src` (`foxglove-202510`).

| Path | Source cite | Named rule |
|---|---|---|
| FMMM component packing | `src/ogdf/energybased/FMMMLayout.cpp:746-760` calls `MAARPacking`; `src/ogdf/energybased/fmmm/MAARPacking.cpp:58-104` runs Best-Fit | FMMM does not use `TileToRowsCCPacker`; it uses `MAARPacking` with decreasing-height presort and `TipOver::NoGrowingRow`. |
| MAAR presort | `MAARPacking.cpp:108-115`; `include/ogdf/basic/internal/list_templates.h:85-99`; `include/ogdf/basic/Array.h:766-803` | `List::quicksort` copies to `Array`; equal keys are stable only when `pR - pL < 40`, otherwise partition swaps equal keys. |
| MAAR row choice | `MAARPacking.cpp:140-179`; `include/ogdf/basic/PriorityQueue.h:271-308`; `include/ogdf/basic/heap/PairingHeap.h:245-258` | The row priority queue is a strict-comparator pairing heap. Equal row widths promote the newly pushed node, so row ties choose the most recently pushed equal-width row. |
| GEM component packing | `src/ogdf/energybased/GEMLayout.cpp:214-229`; `src/ogdf/packing/TileToRowsCCPacker.cpp:121-180` | GEM uses `TileToRowsCCPacker`; it quicksorts component indices by decreasing height using the same `Array::quicksort` equal-key behavior. |

Standalone reproduction of OGDF equal-key quicksort order:

| Equal-height item count | OGDF order head | OGDF order tail | Stable order head |
|---:|---|---|---|
| 10 | `0,1,2,3,4,5,6,7,8,9` | `0,1,2,3,4,5,6,7,8,9` | `0,1,2,3,4,5,6,7,8,9` |
| 40 | `0,1,2,3,4,5,6,7,8,9` | `30,31,32,33,34,35,36,37,38,39` | `0,1,2,3,4,5,6,7,8,9` |
| 41 | `40,39,38,37,36,35,34,33,32,31` | `9,8,7,6,5,4,3,2,1,0` | `0,1,2,3,4,5,6,7,8,9` |
| 52 | `51,50,49,48,47,46,45,44,43,42` | `9,8,7,6,5,4,3,2,1,0` | `0,1,2,3,4,5,6,7,8,9` |
| 202 | `150,149,148,147,146,145,144,143,142,141` | `60,59,58,57,56,55,54,53,52,51` | `0,1,2,3,4,5,6,7,8,9` |

First differing placement decision:

- `random_dag_50` has 52 connected components: 50 singleton components, one 45-node component, and one 2-node component.
- For any equal-height singleton block above 40 entries, current Dagua's stable Python sort visits singleton component `0` first; OGDF `Array::quicksort` visits singleton component `51` first for a pure 52-equal-key block.
- For `random_dag_200`, current Dagua visits singleton component `0` first; OGDF `Array::quicksort` visits component `150` first for a pure 202-equal-key block.
- The row-tie rule also diverges: current Dagua's `min(..., key=total_width)` chooses the oldest equal-width row; OGDF `PairingHeap::merge` promotes the newly pushed equal-priority row.

### Port Attempt and Gate Result

Attempted local port:

- FMMM: replaced stable decreasing-height presort in `_ogdf_maar_pack_component_transforms()` with an `Array::quicksortInt` replica and changed equal-width row selection to the newest pushed row.
- GEM: replaced stable decreasing-height `sorted()` in `_ogdf_tile_to_rows_offsets()` with the same `Array::quicksortInt` replica.
- Scope stayed inside OGDF-family paths: `dagua/layout/ops/pipelines/fmmm.py` and `dagua/layout/ops/gem.py`.

The port was reverted because it decisively failed gate 1:

| Graph | Engine | Seeds | RMSD after attempted port |
|---|---|---|---|
| `random_dag_50` | `classic_fmmm_steps10` | 100-104 | `0.875726474, 0.937536197, 0.969028068, 1.00536909, 1.10005342` |
| `random_dag_50` | `classic_fmmm_steps100` | 100-104 | `0.914555166, 0.974438672, 0.860822156, 0.968169109, 0.684440193` |
| `random_dag_50` | `classic_fmmm_steps200` | 100-104 | `0.911767335, 0.914839982, 0.865514939, 0.948250512, 0.71062376` |
| `random_dag_50` | `classic_gem_iters2000` | 100-104 | `0.957918847, 0.993176241, 1.04400096, 1.03882264, 0.900542281` |

Control probe with only the MAAR presort monkeypatched back to stable order still failed:

| Graph | Engine | Seeds | RMSD with stable presort + newest-row tie |
|---|---|---|---|
| `random_dag_50` | `classic_fmmm_steps10` | 100-104 | `0.904670135, 0.943453448, 0.826494919, 0.835699857, 0.935033545` |
| `random_dag_50` | `classic_fmmm_steps100` | 100-104 | `0.920496674, 0.904937817, 0.954985801, 0.812001922, 0.853137827` |
| `random_dag_50` | `classic_fmmm_steps200` | 100-104 | `0.898150653, 0.930598387, 0.964788338, 0.803632296, 0.85844442` |

This isolates the row-tie port as independently incompatible with the saved r76 references, and the GEM probe shows the `TileToRows` quicksort port is also incompatible for the available target reference.

### Reference Availability

The expected r76 reference tensors were present for `random_dag_50` target rows, but not for these expected `random_dag_200` filenames:

- `random_dag_200__ogdf_fmmm__for__classic_fmmm_steps10__seed100.pt`
- `random_dag_200__ogdf_gem__for__classic_gem_iters100__seed100.pt`

`find /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r76_refs/positions -name 'random_dag_200*'` returned no files in this worktree session.

### Verdict

The exact source-level tie rules are portable, but porting them made the available target rows much worse rather than closing the residual. That means the current residual is not fixed by the naked MAAR/TileToRows sort and row-tie semantics, or the saved r76 reference corpus being used by the gate is not the same corpus implied by the task text.

No implementation patch was kept. No commit was created.

### Test Results

Not run after reverting the failed implementation patch; there are no source changes to verify. The failed target probes above are the gate evidence for parking the attempt.
