# R36 fdp_tilepack Summary

## Source files read

- `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c`
- `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c`
- `/home/jtaylor/projects/_references/graphviz/lib/pack/pack.c`
- `/home/jtaylor/projects/_references/graphviz/lib/common/utils.c`
- `dagua/layout/ops/pipelines/fmmm.py`
- `tests/test_layout/test_fmmm_fidelity.py`

## Implementation summary

- Added Graphviz-style weak-component decomposition for `layout_fmmm_pipeline` under `reference_mode` / `fidelity_mode` only.
- Added a Python port of Graphviz `pack.c` bbox polyomino packing:
  - `computeStep`
  - `genBox`
  - perimeter-descending component order
  - `placeGraph` spiral search
  - C-compatible rounding and integer division helpers
- Reused the same seed for every independently solved component, matching `fdp_tLayout` reseeding each component with `T_seed`.
- Kept default non-fidelity behavior on the existing single full-graph path.

## Tests added

- `test_fmmm_graphviz_tile_pack_offsets_match_pack_c_golden_vectors`
- `test_fmmm_graphviz_tile_pack_offsets_handle_nonzero_box_origins`
- `test_fmmm_fidelity_mode_packs_disconnected_components_only_in_fidelity`

Golden vectors are hand-captured from the referenced `pack.c` algorithm by following `computeStep`, `genBox`, sort order, and `placeGraph` with representative boxes. Direct C golden capture was not isolated because fdp component packing is invoked through derived graph construction and post-layout `putGraphs`, not a standalone CLI-visible subroutine.

## Verification

- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `ruff check dagua/layout/ops/pipelines/fmmm.py tests/test_layout/test_fmmm_fidelity.py --fix`: passed.
- `pytest tests/test_pipeline_fmmm.py tests/test_layout/test_fmmm_fidelity.py -x --tb=short -q`: passed, 29 passed, 2 warnings.
- `pytest tests/test_layout/test_fmmm_fidelity.py -x --tb=short -q`: passed, 9 passed, 2 warnings.
- `pytest tests/test_graph.py -x --tb=short -q`: passed, 37 passed, 2 warnings.
- `python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp --graphs multi_component_80 --seeds 1 --output-dir eval_output/algo_fidelity/round_36/fdp_tilepack/live_compare`: completed, median RMSD 0.123076.

## Blockers / interface assumptions

- Repository-wide `ruff check . --fix` is blocked by unrelated concurrent R36 files: `dot_rank.py`, `quadtree.py`, `sfdp.py`, and `sugiyama.py`.
- Final Tier 2 `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` is blocked at collection by pre-existing `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` was killed without output after a long run; split coverage for the relevant new layout test and `tests/test_graph.py` passed.
- Staging/commit was not completed because sibling sprint edits are concurrently present in the same target file (`dagua/layout/ops/pipelines/fmmm.py`). Staging the file wholesale would include unrelated fdp ports/recursion changes.

## Integration notes

- The packer currently uses bounding boxes from solved component positions plus optional `node_sizes`; edge splines and node-level polyomino coverage are not available at this pipeline layer.
- The public behavior change is gated by `reference_mode or fidelity_mode` and only runs when more than one weak component is present.
