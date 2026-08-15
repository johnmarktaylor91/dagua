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
   masses, fixed unit route masses, and per-edge stroke widths, so the remaining
   difference is union/capsule certification rather than missing input weights.
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
    contract's stated severity interval. U27's container relation uses the same isolated
    helper. The scheduler owner must freeze the missing primitive-id grammar before those
    branches can be certified.
13. U17's declared-containment exemption has no corresponding immutable containment
    relation in `GraphSemantics`. No v4.0 frozen corpus row declares one, so current
    scores are unaffected; schema-owner work is required before such rows are admitted.
14. `SCENE_RECONSTRUCTION_CONTRACT.md` section 8d simultaneously states the class order
    "nodes below labels" and that each label is immediately above its own node. Production
    gives the class clause priority for U18 foreign-node pairs. It also uses the assigned
    stored-row order for U17/U18/U27/U30 rather than parsing future producer `z_order`
    tokens, while U42 still consumes its typed channel order. A future scene-bearing
    corpus needs one schema-owner primitive-id grammar before either posture generalizes.
15. U33 specifies `a_perp` without orienting it. Production orients it clockwise from the
    declared top-to-bottom page axis, so positive sibling order is left-to-right. This is
    score-visible and remains a scheduler-owner convention rather than contract text.
16. `MANIFEST.json` declares U09 scale-neutral, while U09 section 3 freezes
    `log(len_e + 1e-12*u)`. Because `u` is unchanged under position-only scaling, the
    formula cannot be exactly scale-neutral on every fixture; production follows the
    facet formula.
17. U13 section 7 does not state whether its three sub-term-(i) blend components are
    saturated separately. Production follows U10 section 7's sibling construction:
    compute raw integrals, saturate each component by `x/(x+0.05)`, then blend.
18. Graphs with multiedges or self-loops require producer-supplied distinct routes in the
    current ingestion schema because a derived straight chord cannot distinguish them.
    They are typed `MISSING_REQUIRED_PRIMITIVE`; admitting default render-truth routes for
    those graph classes requires a route-schema extension.
19. `STYLECONTRACT_FAIRFIELD.md` field 6 places a cluster label at the region top above
    the robust-core center x but does not define the region top when that vertical line
    misses the region entirely (a cluster drawn as separated lumps with the core center
    in the gap). Production evaluates the boundary at the nearest covered x, falling back
    to the region-bounds top; both fallbacks are input-only functions of the derived
    region, so scattered clusters ingest and are scored rather than aborting the scene.
    Measured (P4REVERIFY4): the fallback does NOT punish scattering -- an 8-node cluster
    scores U30 = 0.903 drawn tight, 9.07e-4 as two lumps 100u apart, and exactly 0.0 as
    three lumps (the label lands in empty space, so sub-term (iii) sees no occluder).
    The fallback is docketed for determinism and input-only provenance, not deterrence.
20. U22 section 6 sends "all other declared classes and undeclared graphs" to
    `kappa_class = 1`, while section 13 case (c) types a declared class outside the
    frozen exemption table's domain as `unknown_declared_class` INVALID and pre-bans the
    silent unit fallback as an undocumented score-visible branch. The two clauses
    conflict for unknown class strings. Production follows section 13, the CC-4
    failure-behavior authority: unknown declared classes are INVALID; undeclared graphs
    still score against the unit target. Resolving the section 6 wording is a
    contract-owner item.
21. U03 section 4 weights centers equally within each degree tercile, while section 7
    runs the tercile blend "over the fade-transformed per-center defects ... at
    INPUT-ONLY node-mass weights". The two clauses conflict on mass-declaring graphs.
    Production follows section 4 on both counts: centers are equal-weighted within each
    tercile, terciles combine at equal mass, and components pool by node-count input
    mass (the section 4 / section 7 component-pooling conflict is entry 27). Reconciling
    the section 7 center-weight wording is a contract-owner item.
22. U20a's E2 residual frame requires the declared flow axis for its construction, but
    the exemption's trigger clause is "when GraphSemantics declares ranks/layers" with
    no axis condition. For a ranks-declared, axis-undeclared graph production keeps the
    no-fabrication posture (matching U32's `RANK_AXIS_ABSENT` and the U22/U23 declared-
    axis rule): E2 is not applied, so a correct column of such a graph scores the raw
    quotient. E2-uncomputable-without-axis is a contract gap; the scheduler owner may
    prefer a typed sub-term drop over the raw score.
23. U08's golden-exactness envelope `_ANGULAR_ZERO_ENVELOPE = 1e-12` snaps a pair defect
    at or below 1e-12 to exactly 0.0 in production so the contract's exact-zero golden
    (U08 section 14 golden 1: "defect 0 (exact)") is reachable. The constant appears
    nowhere in U08.md; it sits inside the ecosystem's 5.8 numerical envelope (U08 itself
    tolerates 1e-9 there) and can only ever lower a defect by <= 1e-12, but it is an
    unfrozen score-path constant pending contract-owner adoption.
24. U20a section 6 (iii) reads `D^iii_c = 1 - prod_pairs (1 - m_kl)^(1/n_c)` flat over
    pairs, while sections 4 and 7 collapse feature pairs "onto their owning node/route
    object" and run the global blend over feature-owning objects. Production implements
    the object-level reading; because each pair's loss lands on both owners, the pair
    survival contributes `(1 - m)^(2/n_c)` across the object population rather than the
    flat `(1 - m)^(1/n_c)`. Score-visible on multi-pair scenes; reconciling section 6's
    flat product with the section 4/7 collapse is a contract-owner item.
25. U11 section 6 (v)'s confusability closed form factors into a gap term and a tangent
    angle term. A zero-arc route has no initial tangent, so the angle factor is
    undefined; production charges its supremum (1.0) and keeps the well-defined gap
    factor, so a coincident tangent-less pair is the coincidence limit (golden G10's
    rising branch) while a distant one still earns its separation (G10's falling
    branch). The zero-tangent case itself is uncontracted. (Since r4 BLOCKER-1, pairs
    with PRESENT tangents use the oriented angle -- anti-parallel departures read as
    `pi`, maximal separation -- so the supremum charge here applies only to genuinely
    missing tangents.)
26. Declared ranks without a declared flow axis reach three different branches: U20a's
    E2 residual frame is not applied (entry 22), U22 computes the section 6
    layer-profile target from the ranks but measures in the direction-free frozen
    rotation frame (the orientation-less target is folded to its >= 1 side since the
    rotation-scan aspect is >= 1 by construction), and U23 stays rotation-averaged.
    No direction is ever fabricated from ranks; whether a rank declaration constitutes
    a declared axis needs one contract-owner answer across the package.
27. U03's section 4 weights block pools components "by node-count input mass" while the
    section 7 composition summary says "components by node mass". The clauses conflict
    on mass-declaring graphs. Production follows the dedicated weights block: components
    pool by node count. Centers stay equal-weighted within each tercile (entry 21);
    section 7's node-mass wording is the unadopted side of both conflicts.
28. U30 sub-term (iii) saturates by contract arithmetic on ordinary labelled cluster
    scenes: `a_c` from U27's closed form admits every primitive within ~2.5 node
    diagonals of the label, and the flat unnormalized noisy-or over the admitted
    population pins at 1 (no per-opportunity exponent like U20a's class product). The
    port matches both formulas exactly, so the row is non-discriminative as frozen;
    a scheduler-owner ruling is required before U30 (iii) can carry corpus rows.
29. U20a's E2 exemption is reachable only on an exactly zero residual:
    `_isotropy_quotient` returns the degenerate reading (`1.0`, exempt) iff the residual
    cloud's leading singular value is exactly 0, and any nonzero residual along the
    declared axis is a 1-D cloud whose quotient is 0, so `max(q_raw, q_residual)` grants
    nothing. The exemption set is measure-zero, and `U20a.ii` jumps by its full range
    across it: a declared-layer column with within-rank jitter `eps = 1e-9` scores
    `ii = 1.0` while `eps = 0` scores `0.0` (verified at every `eps` up to 4.0, and at
    the `75959e8d` baseline with `eps = 2.0` -- residue, not regression). Read along
    U20a golden 2's collapse ladder the sub-term therefore improves `1.0 -> 0.0` at the
    instant within-rank pairs become coincident, against the ladder's "no cliff" clause;
    golden 3's "declared ranks whose axis explains the collinearity -> D^ii = 0" is
    satisfied only at the exact fixture. A graded exemption needs a contract-owner
    definition of when the declared axis "explains" a nearly-rank-collapsed cloud (a
    residual-scale threshold or a C^1 blend); production keeps the exemption-only
    `max(q_raw, q_residual)` construction, which cannot re-open the E3 collapse channel,
    and escalates the knife edge rather than freezing an uncontracted tolerance. (The
    composite is shielded at full collapse because sub-terms (i) and (iii) saturate;
    the jump is score-visible in the published, manifest-weighted sub-term.)
30. U22 section 6's class-exemption table states its kappa_class constants in the
    direction-free elongation convention (`A_obs = h_max/h_min >= 1`: a path is a tree
    with `b = 1, d = n`, so `path -> 8` and `tree -> max(1, b/d) = 1` cannot share a
    signed ratio), while section 5's declared-axis bullet makes `A_obs` signed
    (breadth over depth) and never contemplates a declared axis plus a declared class
    with no ranks (the mirror of entry 26's ranks-without-axis case). Production maps
    each constant through its class's elongation direction: `path`/`chain` elongate
    along the flow axis, so kappa 8 enters the signed frame as a breadth/depth target
    of 1/8; `tree`'s `max(1, b/d)` and `lattice`/`grid`'s declared `width/height` are
    already breadth-over-depth quantities and enter unchanged; in the direction-free
    branch every target folds onto the `>= 1` side of unity (entry 26's fold, which
    for the class table only ever moves a sub-unit grid aspect). Residue kept as
    frozen: for a deep tree (`b < d`) the table's `max(1, .)` fold pins the target at
    square in both frames, so the signed frame cannot express "should draw deep";
    granting a depth-side tree exemption is a contract-owner item.
