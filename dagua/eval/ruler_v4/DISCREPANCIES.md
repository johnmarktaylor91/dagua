# V4 facet-port discrepancies

The frozen Markdown contracts and `MANIFEST.json` are authoritative. This file records
both frozen-artifact inconsistencies and deliberate certification-tier limitations in
the production port; none of the entries below overrides a contract.

1. The dispatch claim in the phase brief says the executable spike runs all 45
   contracts. Its `CORE_FACETS` and `FACET_FUNCTIONS` contain 23 ids: U01, U03, U07,
   U08, U09, U10, U11, U17, U18, U20a, U21, U22, U25, U26, U27, U28, U31, U32, U33,
   U35, U36, U38, and U41. Production dispatch is derived from the 45-row manifest.
2. The spike accepts `half_extents` and `u` as scene fields. CC-20 and the frozen
   contracts make extents a function of GraphSemantics plus StyleContract and make
   `u` the median derived primitive diagonal. Production ingestion rejects reported
   extent fields and derives both values.
3. The spike's U21 overflow anchor reduces to an outside-cell count divided by a
   squared grid-step count. U21 section 6c includes mean primitive area and maximum
   primitive diagonal explicitly. Production uses the frozen dimensional formula.
4. `EVENT_REGISTRY.json` records U20a contract SHA
   `625d95ce9d9caef1fc20e174278602efb06bab06c03754d34dacdfd3e1c37dc1`, while the
   frozen contract bytes and manifest record
   `65e179ba53966fecc9fb77c4c219e05086bee5dac536540ab46bab785313b530`.
   Production contract identity follows the manifest and actual bytes; event loading
   preserves the event row independently of its stale provenance field.
5. `EVENT_REGISTRY.json` records U30 contract SHA
   `6e4d76f0d3f37fdb60a7244fab36ee4642038dd42c4f05162b76bcfe238a1d3f`, while the
   frozen contract bytes and manifest record
   `64ce190db377b294922a09b4b0e045224c02d611ff451b33f3fb5d07d1ca69c4`.
   Production contract identity follows the manifest and actual bytes.
6. The manifest contains 94 declared sub-terms, while the phase brief states 91
   scored sub-terms. The manifest marks two U07 rows and one U33 row as unscored, so
   the production scored inventory is 91 and the complete declared inventory is 94.
7. The spike depends on producer-populated component count and flow-axis fields.
   Frozen ingestion makes component structure and declared-axis applicability
   GraphSemantics-owned. Production derives component count from topology and admits
   an axis only from declared direction/rank semantics.
8. `BoxGeometry` currently stores axis-aligned half extents and no primitive
   orientation. U17/U18 and cluster-region consumers therefore evaluate exact AABB
   separation/intersection rather than the contracts' general OBB polygon geometry.
   Ingestion-derived v4.0 rows are axis-aligned; promotion to oriented primitives
   requires a schema-owner extension.
9. U16 route/label overlap and U21 escaped-content mass use exact clipped centerline
   length times declared stroke width. They do not yet include round-cap area or form
   an exact union of overlapping escaped primitives. U21 does apply declared node
   masses, edge masses, and per-edge stroke widths, so the remaining difference is
   union/capsule certification rather than missing input weights.
10. U27 penetration uses the minimum signed distance to the member-offset pieces. This
    is exact outside the union but is a conservative lower magnitude inside overlapping
    pieces; exact distance to the complement of the full union remains a geometry-owner
    certification item. U28's adaptive analytic region integration likewise lacks the
    separate arrangement-proof certificate requested by its contract.
11. The frozen U13 contract fixes a 0.60 smooth-maximum share for sub-term (ii) but does
    not divide the remaining 0.40 between mean and CVaR. Production preserves the
    global 0.65:0.25 ratio for that remainder. U15 does not freeze which shared length
    defines `L_sh` or the tangent-window obstacle construction; production uses the
    shorter route and records those choices as scheduler-owner docket items.
12. The stored-row total order is frozen by `SCENE_RECONSTRUCTION_CONTRACT.md` section
    8d (routes below nodes below labels, then canonical order). It does not assign a
    relative canonical-id grammar between node labels and cluster labels, nor does the
    scene schema expose a container-boundary primitive. U30 isolates a bounded interim
    for those pairs: an overdrawn subject pays exactly one half, never more than the
    contract's stated severity interval. The scheduler owner must freeze the missing
    primitive-id grammar before those branches can be certified.
13. U17's declared-containment exemption has no corresponding immutable containment
    relation in `GraphSemantics`. No v4.0 frozen corpus row declares one, so current
    scores are unaffected; schema-owner work is required before such rows are admitted.
