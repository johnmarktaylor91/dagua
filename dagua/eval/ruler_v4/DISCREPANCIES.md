# V4 facet-port discrepancies

The frozen Markdown contracts and `MANIFEST.json` are authoritative. The production
port follows them in every case below.

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

