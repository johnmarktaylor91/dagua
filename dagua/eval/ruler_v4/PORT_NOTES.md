# V4 facet-layer port notes

The production package is independent: it imports no earlier ruler implementation and
has no workspace-path dependency. Cross-facet composition, fitted weights, surrogate
construction, and entry-point integration remain absent by design.

## Shared symbols

| Spike symbol | Production location |
|---|---|
| `Scene` | `scene.Scene` plus the input-owned `GraphSemantics`, `StyleContract`, and `ObservationProfile` records |
| `FacetResult`, `_value`, `_na` | `scene.FacetResult`, `scene.value_result`, `scene.na_result` |
| `_scene_from_record` | `ingestion.ingest_record` |
| `_robust_half_extent`, `_frame` | `frames.robust_projection`, `frames.robust_frame` |
| `_all_pairs_graph_distances`, `_adjacency`, `_components` | `_util.graph_distances`, `_util.adjacency`, `_util.components` |
| `_pava`, `_stress` | `_util.pava`, `_util.isotonic_stress` |
| `smoothstep`, `soft_pos` | `_util.smoothstep`, `_util.soft_pos` |
| `FACET_FUNCTIONS`, `run_scene` | `registry.FACET_FUNCTIONS`, `registry.evaluate_facet` |
| frozen event JSON access | `events.load_event_registry`, `events.evaluate_jump_bound` |

## Facet symbols present in the spike

| Spike symbol | Production module and exact-id function |
|---|---|
| `facet_u01`, `facet_u03`, `facet_u09`, `facet_u22` | `structure.U01`, `structure.U03`, `structure.U09`, `structure.U22` |
| `facet_u07`, `facet_u08`, `facet_u10`, `facet_u11` | `edges.U07`, `edges.U08`, `edges.U10`, `edges.U11` |
| `facet_u17`, `facet_u18`, `facet_u20a`, `facet_u21` | `legibility.U17`, `legibility.U18`, `legibility.U20a`, `legibility.U21` |
| `facet_u25`, `facet_u26`, `facet_u27`, `facet_u28` | `clusters.U25`, `clusters.U26`, `clusters.U27`, `clusters.U28` |
| `facet_u31`, `facet_u32`, `facet_u33` | `directed.U31`, `directed.U32`, `directed.U33` |
| `facet_u35`, `facet_u36` | `weights.U35`, `weights.U36` |
| `facet_u38`, `facet_u41` | `packing.U38`, `packing.U41` |

## Contract-only production families

The remaining 22 ids have no spike facet symbol and are implemented from their frozen
Markdown contracts: U01b, U02, U04a, U04b, U05, U06, U12, U13, U14, U15, U16, U19,
U20b, U23, U24, U29, U30, U34, U37, U39, U40, and U42.

