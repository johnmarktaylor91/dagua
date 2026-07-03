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
