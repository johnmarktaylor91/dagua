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
    flat product with the section 4/7 collapse is a contract-owner item. Interim row
    status: U20a (iii) carries corpus rows under the adopted object-level reading --
    every candidate drawing of a fixture is scored under the same exponent, so rows
    stay comparable while the wording conflict is open.
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
    the jump is score-visible in the published, manifest-weighted sub-term.) Interim
    row status: U20a (ii) carries corpus rows -- the jump is confined to the
    measure-zero exact-collapse manifold, every off-manifold fixture scores the
    continuous raw quotient, and a fixture landing exactly on the manifold scores the
    golden-3 exemption side; the contract-owner threshold decision is needed before
    within-rank near-collapse ladders (golden 2's regime) can be certified, not
    before ordinary rows ingest.
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
    granting a depth-side tree exemption is a contract-owner item. Two further silent
    section 6 gaps live in this seam. First, the three `A_target` cases carry no
    stated precedence, and production prefers the layer profile when a graph declares
    both ranks and a known class (the class multiplier is gated on absent ranks) --
    score-visible on `path`/`chain` (a 40-node declared path drawn along its axis
    scores 0.522154 class-only vs 0.0 with ranks added), adopted as the more specific
    input-only statistic per entry 20's section 13 / CC-4 tie-break; the published
    `exemption` key records which branch fired. Second, a path graph is also the
    `1 x N` lattice, so two truthful declarations of one graph exist and disagree
    outside the `8 x 3 = 24` plateau (0.010723 as `path` vs 0.0 as `grid (1, N)` on a
    3.0u strip) -- a `V4_SPEC` 5.3 #17 exposure confined to beyond-plateau strips,
    reachable only by a coordinated `class` + `dims` mutation (no single-field
    mutation improves), and contract-internal (section 6 mixes a coarse constant
    with an exact declared aspect over overlapping families); kept as frozen.
31. `_UNIT_DUST = 1e-12` (`_util.py: snap_unit`) clamps accumulated float rounding off
    producers whose closed form is analytically in `[0, 1]` -- signed log sums
    (Jensen-Shannon divergences), renormalized convex combinations, log-sum-exp and
    sigmoid means -- at the producer call sites feeding the `[0, 1]` guards. The
    constant appears in no contract; entry 23's reasoning covers it verbatim: it sits
    inside the ecosystem's 5.8 numerical envelope, is one-sided into range, can only
    ever move a value by <= 1e-12, and excess beyond the envelope passes through
    unchanged so `value_result` and the blend domain checks still raise on any real
    range violation (measured value-inert on the round-4 540-cell and round-5
    2295-cell batteries). Like entry 23 it is an unfrozen score-path constant pending
    contract-owner adoption; the two constants should be adopted or replaced
    together.
32. V4_SPEC_r4 R3-DR makes the one-parameter loss-space p-mean the v4.0 default and
    names the optional nesting family only by `(q, beta, allowances, lambda)`, while the
    phase-2 implementation brief explicitly requests mean-plus-bottleneck machinery and
    the only published closed form (IDEAS_SOL section 3.3) uses a nondifferentiable
    `max(0, L_tail)`. Production exposes both R3-DR families and replaces that bare
    hinge in the optional bottleneck family with a SUM of per-group C1 positive-onset
    excess debts `phi(loss_g - allowance_g)` with `phi(x) = x^2/(x+tau)` for `x > 0`,
    zero otherwise (the P2 review's normalized log-sum-exp tail was count-dependent --
    `-tau*ln(n)` of relief per applicable group, against 3.3's universal-mass floor --
    and is replaced: a group at or under its allowance contributes exactly zero, so
    the arm is invariant to applicable-group count and a catastrophic group is visible
    at any mass). All parameters remain explicit. This preserves the requested
    non-compensation and CC-1 smooth-onset properties without claiming the optional
    family has passed R3-DR. Two open boundaries are disclosed rather than resolved:
    the implemented pair are DISJOINT families (`beta` is constrained to `(0, 1]` and
    the mean arm is arithmetic, so `P_MEAN` is not the `beta = 0` boundary case of
    `MEAN_SOFT_BOTTLENECK` and R3-DR's freeze-at-prior fallback cannot be executed
    inside one family), and the exact nesting formula plus shipped-family selection
    remain P5/freeze inputs rather than phase-2 weight values. The SUM form also
    changes the arm's RANGE: `P_MEAN` stays in [0, 1] but `MEAN_SOFT_BOTTLENECK`
    `l_total` grows without bound in the applicable-group count under simultaneous
    catastrophe (measured 0.941 -> 6.057 from 1 to 16 groups, all at defect 1.0
    against allowance 0.1, tau 0.05, beta 0.4), so `HeadlineProfile.loss_scale` is
    family-specific and `l_total` values are NOT comparable across families; the
    event-margin seam stays conservative because per-group sensitivity
    `(1-beta)*nm_g + beta*onset_slope` remains below 1.
33. V4_SPEC_r4 3.3 resolves the weight-bearing manifest structure as SOL's generated
    semantic-slot budget ledger with FABLE's 8 groups persisting only as a slot->group
    REPORTING rollup carrying no weight semantics, but the frozen MANIFEST.json carries
    no group/slot field on any facet row, so no build-time source of truth exists for
    either partition. Production therefore makes the shipped P_MEAN family compose over
    the frozen scored sub-term inventory directly (the finest partition already frozen
    in CONTRACTS), which is score-inert to every relabelling of `SubtermWeight.group`
    by construction; the group field feeds attribution rollups only. The optional
    MEAN_SOFT_BOTTLENECK family does consume the group partition through its explicit
    per-group allowances: that partition, like the family selection itself (entry 32),
    is a P5/freeze input to be generated with the slot ledger under A18 -- a table
    shipping that family without a frozen partition artifact has no defense against
    partition drift, and phase-5 tooling must validate it before the family can pass
    R3-DR.
34. V4_SPEC_r4 CC-1's frozen ordering rule makes a strict verdict require BOTH the
    decision margin to exceed the summed nearby jump bounds AND SE_pair to sit below
    half the smallest score-visible single-event jump bound, and 4.1/4.3 additionally
    require the JND/posterior arm (p_win over a calibrated latent advantage) before a
    WIN is published. Phase 2 implements the jump-bound-sum arm always and the SE_pair
    arm whenever the caller supplies the uncertainty-ledger inputs
    (`se_pair`/`smallest_visible_jump_bound`); the JND/posterior arm is V4-POLICY
    fit-time machinery that cannot exist before P5 calibration, so
    `compare_with_event_margin` publishes deliberately scoped `MARGIN_RULE_*` verdicts
    (never WIN/TIE semantics from 4.3) and MODULARITY.md directs phase 3 to wrap --
    not rename -- this result with the ledger and JND machinery. Sub-JND margins
    therefore remain publishable as MARGIN_RULE_* by this seam alone; no consumer may
    treat them as certified strict wins. The SE_pair arm is consequently OPT-IN at
    this seam where CC-1 states it as a conjunct: a caller supplying no uncertainty
    ledger receives a one-arm MARGIN_RULE_* verdict with `se_gate_passed = None`
    rather than being forced to declare an explicit `se_unavailable`
    acknowledgement; the phase-3 wrapper that owns the ledger owns forcing that
    declaration (P2 round-2 finding 5).
35. V4_SPEC_r4 CC-13/6.2b make every unobserved or under-sampled facet enter a tier's
    composite as its FULL FEASIBLE INTERVAL, while 3.6's PM-1 denominator renormalizes
    INAPPLICABLE terms away; phase 2's point-valued `compose` originally collapsed both
    absence classes into renormalization, so absence of observation improved the score
    (a two-row 0.2/0.6 table scored 0.4 observed but 0.2 with the bad row unobserved).
    The two classes are now separated at the composition seam: NA renormalization is
    the inapplicable branch only (every phase-1 NA source is input-side, keeping the
    denominator out of the drawing's control, CC-2), and any absence reason in the
    `UNOBSERVED` class on HEADLINE-BEARING mass -- a non-diagnostic, positive-weight
    row -- refuses point composition outright: the honest point-value handling of a
    width-maximal interval row is escalation, not a number. The refusal is scoped to
    headline-bearing mass because CC-13 concerns a facet's entry into a tier's
    COMPOSITE: a weight-0 diagnostic row carries no mass renormalization could hide,
    and diagnostics are exactly the rows most likely to go unobserved under a cheap
    tier, so an unscoped refusal would abort otherwise fully observed headlines for
    a score-inert reason; the diagnostic row publishes its unobserved absence
    instead (P2 round-2 finding 4). Certified per-term interval propagation through
    `compose` (6.2) remains the phase-3 item MODULARITY.md already names; this seam
    guarantees no cheap tier can ever route unobserved mass through the
    renormalizing branch in the meantime.
36. Three P2-review accounting seams are recorded as P5 obligations rather than
    phase-2 code. (a) `NearbyEvent` carries no facet id, so a manifold owned by a
    DIAG (weight-0) or currently-NA facet still enters the CC-1 strict-win budget;
    the error direction is strictly conservative (more EVENT_MARGIN_LIMITED, never a
    strict win), all eight contact-type registry entries carry bound 0.0 today, and
    filtering to score-visible facets requires the fitted weight table, so the
    score-visibility filter belongs to the P5 comparison tooling that owns that
    table. (b) The 3.6 rule that a k-row bundle sharing one fitted identity counts
    one dof ONLY when its internal ratios were fixed a priori is enforceable only
    against the manifest's `fitted_dof_declaration`/`provenance_class` records at
    fit time; phase 2 ships the `provenance_class` field with consistency
    validation, and the manifest cross-check is the A18-side gate. (c) A caller can
    still misclassify a fitted profile scalar as `preregistered_prior` in
    `ScoringProfiles.parameter_provenance`; the classification is fail-closed on
    omission and on unledgered fitted identities, and provenance truthfulness is
    audited against A18/A20 at freeze, not computable at build time. The same
    fail-closed-on-omission rule now covers the mass surface: `validate_for_contracts`
    refuses any positive-mass or fitted-identity `SubtermWeight` whose
    `provenance_class` is undeclared (P2 round-2 finding 1), so misclassification --
    not omission -- is the only residual on either surface.
37. V4_SPEC_r4 5.5 r3 adds a combined crossing+face K12 metamorph ("adding a
    crossing on a high-face-debt near-planar row must strictly worsen the
    composite") to catch the exchange-rate disease -- face relief paying for a
    priced crossing. The shipped property test
    (`test_combined_crossing_and_face_metamorph_strictly_worsens_composite`) runs
    the combined composite on a certified-planar bowtie fixture where BOTH facets
    degrade together (U07 0.0 -> 0.590, U41 0.081 -> 0.307), so it pins combined
    applicability and per-facet monotonicity but cannot detect an exchange rate
    that lets face relief buy a crossing: the disease needs the face term to move
    the OTHER way. A fixture where the added crossing genuinely reduces face debt
    (so the composite must still worsen against the relief) is the missing
    metamorph, docketed for the property family rather than shipped as a
    misleadingly-named test (P2 round-2 finding 6).
38. [REWRITTEN after P3REVIEW OPUS5 BLOCKER-2 / FABLE B1; the original entry's
    facet-contract claim was FALSE.] V4_SPEC_r4 6.5's deliverable is the
    analytic soft relaxation itself: `score_v4_soft` evaluated for gradient
    alignment against the exact facets, with positions as the differentiation
    variable. The frozen facet contracts DO declare position-level smoothing
    classes with pinned temperatures, inside the scored closed forms
    themselves: U34.md:31 (the scored smooth replacement
    `sp_tau(x) = tau*log1p(exp(x/tau))` with `tau = 0.01*mean_segment_length`);
    U11.md:315-326 (capped baseline through `softmin_{t_soft}`, an LSE with
    `t_soft = 0.1*chord_e` per U11.md:204) and U11.md:179-181 (`hz_tau` hinge,
    exactly zero below 0 and the softplus asymptote above, `tau = 0.05`);
    U03.md:76-77 (sigmoid credit, `tau = 0.25` PREREG-PRIOR); U06.md:109
    (smoothed max = LSE at the global temperature); U02.md:146 (frozen
    dimensionless temperature `tau_r`, log-domain); and the ONE GLOBAL 3.7
    blend's smoothed-max component (U26.md:240, U30.md:227), whose temperature
    3.6 budgets as a fitted constant (V4_SPEC_r4:872-873). CC-1 already
    requires every facet to be continuous in positions off declared event
    manifolds, so 6.5 needs no separate smoothing catalogue: the surrogate is
    the frozen closed forms evaluated differentiably. Production therefore
    ships a traced execution path (`surrogate.traced`) in which position
    tensors flow through the SAME facet implementations to `l_total`;
    the contract-named smoothing classes above are part of those closed
    forms; and constructs that remain a.e.-flat in positions with NO
    contract-named smoothing (hard counts, isotonic/order fits, trims, event
    indicators) carry documented zero-gradient exemptions citing their
    contracts, never an invented softener (finite-differencing,
    straight-through estimation, and ad-hoc soft thresholds remain banned).
    The genuine residual gap is machine-readability: the generated
    `MANIFEST.json` carries contract identity, scored subterms, constants,
    frame class, and event metadata, but NO machine-readable smoothing-class
    field -- the declared classes live in contract prose. OWNER: the A18
    manifest/freeze-allowlist generator (P5 freeze tooling) must emit a
    generated `smoothing_class` field per scored subterm before any smoothing
    rule beyond the contract-named set can be admitted.
    `SurrogateTermTrace.smoothing` is typed `Optional[str]` so the
    contract-named class is recordable today.
39. V4_SPEC_r4 6.3's conformance gates are implemented at pilot tier, not
    P5 tier, and the gap is deliberate scope (P3REVIEW OPUS5 MAJOR-1/-7,
    FABLE M1). Shipped now: rank-fidelity machinery (`certify_rank_fidelity`,
    fail-closed on zero comparable pairs), a deterministic stratified PILOT
    bank (`tests/eval/ruler_v4/scene_bank.py`: 4 structural classes x 3 size
    bands x 6 graded drawings, seeded, digest-frozen per run), per-cell
    soft-path tau gates on that bank, and 6.2a's two pre-P3 conformance
    numbers (L_mean-vs-L_total ordered top-2 disagreement per cell;
    per-tier pruning power under paired, paired-CRN, and marginal
    certificates with published pilot policy: margin 0.01, tier half-widths
    {0, 0.02, 0.1, 0.3}). NOT shipped, P5-owned because they require the
    frozen production bank, the fitted weight table, the JND artifact, and
    battery-#6 adversarial geometry, none of which exist before calibration:
    pairwise decision agreement outside the full-ruler JND region;
    disagreement concentration on honest interval overlap; the adversarial
    sacrificial-tail winner-retention/zero-confident-elimination gate; the
    escalation-cost histogram; the 6.2d bias diagnostic and
    second-order-corrected tier estimate. OWNER: P5 calibration campaign
    (surrogate-certification stage) -- these gates run once on the frozen
    bank against the fitted table before any tier is licensed to prune in
    production.
    P3FIX5 (P3REVIEW2 OPUS5 MAJOR-4): the pilot-bank tau floor is satisfied
    VACUOUSLY and is published as such, never as passing evidence. Measured
    reason: soft-vs-exact `l_total` deviation is 1-3 ULP (max 3.331e-16
    across all 72 bank scenes) while the smallest inter-scene loss gap in
    any cell is 2.809e-4 -- the deviation is 2.5e12 to 5.3e13 times smaller
    than the gap it would have to bridge to flip one pair, so bank tau
    cannot fall below 1.0 for any surrogate built as the frozen closed
    forms executed differentiably. The OPERATIVE 6.5 gates at pilot tier
    are gradient alignment (the descent/ascent battery plus the bank
    descent evidence) and gradient liveness (72/72), not tau. Falsifiable
    rank evidence ships in
    `tests/eval/ruler_v4/test_certification_discriminating.py`: a graded
    loss-gap ladder whose bottom rungs sit 1-4 ULP of `l_total` apart --
    16 scenes, 120 comparable pairs of which 11 have gaps within 4x the
    measured deviation (pairs a deviation-scale error CAN flip; a
    discriminating-population precondition asserts they exist), tau 1.0
    with zero discordant pairs (the traced deviation is a systematic
    per-graph accumulation offset, so it cancels in comparisons) -- plus a
    drift control proving the same machinery REFUSES a quantized
    distillation-class surrogate (tau 0.354, uncertified) on the same
    population. The former certification smoke's `tau == 1.0` assertion is
    restated as the float-determinism check it is (the identity surrogate
    can never produce another value). The 6.3 per-cell tau floor is
    re-scoped at P5 to what it can still catch -- surrogate DRIFT
    (distilled, partially detached, or re-implemented surrogates whose
    forward value no longer tracks exact) -- not the delivered traced
    surrogate's fidelity, which is pinned by construction and measured at
    the ULP scale.
40. V4_SPEC_r4 6.2(b)'s anytime-valid confidence machinery is not
    constructed anywhere in phase 3, and racing is single-round (P3REVIEW
    OPUS5 MAJOR-3/-7, FABLE M2). What exists: `confidence_id` is a required
    allocation label enforced for CONSISTENCY across every candidate
    interval and paired certificate in a race (one race, one allocation),
    and budget exhaustion with an eliminated incumbent fails closed to a
    typed inconclusive result. What does NOT exist: construction or
    validation of simultaneous/anytime-valid confidence sequences, a
    preregistered familywise error allocation across candidates, facets,
    and escalation rounds, a multi-round escalation ladder
    (tier -> wider-budget tier -> full), budget concentration on near-ties,
    or coverage validation under the actual adaptive racing policy. Callers
    supplying ordinary fixed-sample z-intervals under a shared label WILL be
    treated as certified: the label is caller-asserted trust, not a checked
    property. OWNER: P5 calibration campaign (racing-activation stage) --
    the confidence-sequence construction and the multi-round ladder are
    licensed only with the fitted tier definitions, and no production
    selection may rely on `race_candidates` pruning before that lands;
    until then every published score stays full-tier (6.1's "only B2/full
    enters tallies" already guarantees tally integrity independently).
41. U33 conformance fix (P3FIX2): the P4 port returned INVALID with reason
    TREE_SEMANTICS_ABSENT both when tree semantics are absent and when a
    declared tree is trivial (no root or no child anywhere). The frozen
    contract is explicit the other way: "U33 is applicable to a nontrivial
    declared rooted tree/forest. Absence is `NA:TREE_SEMANTICS_ABSENT`;
    malformed parent/depth/order data are invalid" (U33.md, "Input schema
    and applicability"). Every sibling facet already used NA for absent
    semantics (U31 DIRECTION_OR_AXIS_ABSENT, U34 FLOW_SEMANTICS_ABSENT,
    U39 PORTS_ABSENT, U35/U36 WEIGHTS_ABSENT). Fixed to
    `na_result("TREE_SEMANTICS_ABSENT")` for both absence arms; the
    malformed arms (bad layout token, wrong lengths, bad parent index,
    depth mismatch) stay INVALID. Surfaced by the 6.3 pilot bank's
    clustered class, whose scenes declare no tree semantics and were
    unscorable end to end under the port's refusal.
    P3FIX3 refinement: NA is reserved for FULL absence of the tree block
    (no parents, no depths, no layout). A partially declared tree is not
    absence -- "Missing required tree fields is invalid, not NA per
    drawing" (U33.md, "Failure and envelope") -- so declaring some tree
    fields while omitting others returns
    `invalid_result("missing_required_tree_fields")`. Both dispositions
    are banked with their citations in
    `tests/eval/ruler_v4/test_review_repros.py`.
    P3FIX5 refinement (P3REVIEW2 OPUS5 m3): the fully-declared-but-TRIVIAL
    arm (no root or no child anywhere; the "nontrivial" clause of the
    applicability sentence) publishes its own reason token,
    `NA:TREE_SEMANTICS_TRIVIAL`, so 7.2b's per-store applicability rates
    can separate a declared-trivial tree block from true absence.
42. V4_SPEC_r4 6.5's second named evaluation -- "it is evaluated for
    gradient alignment and EXPLOITABILITY" (the clause has survived every
    spec revision, r1 through r4) -- was delivered for gradient alignment
    only; the exploitability half had no implementation, no owner, and no
    deferral until P3FIX5 (P3REVIEW2 OPUS5 MAJOR-3). Structural finding,
    now recorded and executed rather than asserted: the surrogate has NO
    independent exploit surface, because `score_v4_soft`'s forward value is
    the frozen closed forms evaluated on tensors and agrees with the exact
    score to 1-3 ULP at EVERY drawing, not merely at bank drawings. Any
    gradient-followable exploit of the surrogate is therefore an exploit of
    the frozen ruler itself, and ruler exploitation is already owned:
    6.2c routes exploitation discovery to "fresh preregistered blind
    audits" producing V4.x, never a silent change (P5 calibration
    campaign, blind-audit stage). What remains surrogate-specific is
    optimizer DYNAMICS -- detached channels (non-chord routes, edge/cluster
    label boxes, discrete decision layers) steer where descent goes -- a
    descent-quality property with no scoring-integrity consequence, since
    every visited drawing is re-scored by the exact forms. Delivered pilot
    probe: `tests/eval/ruler_v4/test_surrogate_exploitability.py` runs pure
    surrogate-gradient descent trajectories from noisy bank scenes of all
    four structural classes (including the clustered Goodhart-attractor
    family), re-ingests and re-scores BOTH paths at every visited drawing,
    and fails if any trajectory opens |exact - soft| beyond 1e-12, fails to
    reduce the surrogate loss, or fails to transfer the reduction to the
    exact ruler. Measured at pilot: max trajectory gap 1.7e-16, exact loss
    strictly reduced on every class.
43. KNOWING DEVIATION from V4_SPEC_r4 6.2a's cancellation sentence, plus
    the conclusion 6.2a orders from its own measurement (P3REVIEW OPUS5
    BLOCKER-3 fix, docketed per P3REVIEW2 OPUS5 MAJOR-1).
    (i) The deviation. The spec's PRIMARY paired certificate says "shared
    unobserved mass has `D_f` certified at exactly 0, SO IT CANCELS"
    (V4_SPEC_r4, 6.2a). That cancellation claim is FALSE for every
    nonlinear composition family the spec admits (`P_MEAN` with p > 1,
    `MEAN_SOFT_BOTTLENECK`) and exact only for a linear composition: a
    certified-zero paired difference removes the DIRECT channel only,
    while the row's shared level still moves the exchange rate `dL/dd_i`
    at `c + D` versus `c` for every other certified difference.
    `certify_paired_difference` therefore charges every active row the
    full oscillation bound `Lambda_f * (2*level_radius +
    difference_radius)` with no linearity skip for `D == [0, 0]` rows.
    Derivation: with `c_B,i` in the level interval and
    `c_A,i = c_B,i + D_i`, `|c_B,i - c_hat_B,i| <= r_level,i` and
    `|c_A,i - c_hat_A,i| <= r_level,i + r_diff,i`, so the triangle
    inequality on `|L(c_A) - L(c_B) - (L(c_hat_A) - L(c_hat_B))|` with
    per-argument sensitivity bounds `Lambda_i = sup |dL/dd_i|` gives
    exactly `sum_i Lambda_i (2*r_level,i + r_diff,i)`. Single-counted
    level oscillation: still strictly tighter than the marginal rule's
    double-counted levels, but weaker than the spec's stated (unsound)
    cancellation. SPEC AMENDMENT OWED (V4.x): replace "so it cancels" with
    "so its direct channel cancels; the shared level keeps its
    single-counted oscillation charge under any nonlinear family".
    (ii) The measured consequence, which 6.2a itself orders published:
    "if the marginal rule's pruning power is materially lower, the paired
    rule is the difference between a surrogate and a no-op, and the freeze
    report says so". Measured on the pilot bank across all 48 (cell, tier)
    rows (12 cells x 4 tier half-widths): marginal, paired, and paired-CRN
    certificates eliminate IDENTICAL candidate sets everywhere -- pooled
    elimination fractions 0.683 / 0.550 / 0.250 / 0.000 under all three
    families. The power gap is ZERO at pilot scale: with the corrected
    oscillation charge, the PRIMARY paired certificate currently buys
    nothing over the rule 6.2a retains as merely "sufficient", and the
    freeze report says so (GRIND_SUMMARY pruning-power section). The
    r3 rationale for promoting the paired rule (OPUS5 F3's double-counting
    grievance) is voided by the corrected bound at these interval widths.
    (iii) The sound tighter alternative that would restore the paired
    rule's advantage: a curvature bound
    `sup |dL/dd_i(c + D) - dL/dd_i(c)| <= K_i * |D|`, replacing the
    row-width-driven oscillation charge with a difference-width-driven
    one; 6.2d already asserts curvature is analytic "for every RIVALRY-3
    family", so `K_i` is computable at the same cost class as `Lambda_i`.
    OWNER: P5 calibration campaign (racing-activation stage, with entry
    40's confidence-sequence construction) -- adopt the curvature bound or
    demote the paired certificate from PRIMARY before any tier is licensed
    to prune in production.
44. STILL OPEN, recorded so it cannot silently expire (r1 review m3,
    re-raised as P3REVIEW2 OPUS5 m4):
    `RulerRegistration.required_arguments` is advertised metadata that
    `score_with_registered_ruler` never checks
    (`dagua/eval/ruler_registry.py`), so a V3-shaped call routed at
    `{"ruler": "ruler_v4"}` reaches the v4 scorer and dies with a raw
    arity `TypeError` instead of a typed refusal at the registry seam.
    Correctly outside both fix rounds' smallest-sufficient sets (no
    production caller uses registered-ruler dispatch for v4 yet). OWNER:
    registry-seam hardening, to land before any external caller is
    pointed at `ruler="ruler_v4"` (pre-P5 wiring).
45. CLOSED by ADDENDUM-27 FIT-ORD. `PairwiseObjective` consumes the graded
    seven-point response with unit-variance ordered probit, symmetric
    `1:2:3` JND cutpoints, and a seven-category uniform lapse. Verdict magnitude
    drives likelihood selection; the A/tie/B collapse survives only as a
    reporting projection, and confidence remains diagnostic-only.
46. CLOSED by ADDENDUM-27 W-13-EST for the estimator and uncertainty procedure.
    The harness estimates integrated `tau_class` / `tau_band` components,
    freezes the 25-pair and `Q_band=67` gates, publishes supported and
    unestimated cells, profile intervals, separate 2,000-resample graph-cluster
    and generator-family spread intervals with drop rates, and validates true
    cross-session repetitions of one drawing pair. It does not require a
    displayed-order reversal inside every replicate group: the realized campaign
    has no such non-control groups, and W-13-EST requires cross-session
    replication rather than that stricter invariant. Real activation remains
    fail-closed only for the graph-half assignment gap in entry 57.
47. CLOSED by ADDENDUM-27 LOOK-LEDGER (a)-(d). A15 section 5.7's calibration
    reads now route through separate within/cross four-look schedules with frozen
    occasion order, the 8,520 information denominator, cumulative and incremental
    O'Brien-Fleming alpha, content digests, and named decisions. Non-train labels
    remain opaque after loading; a whitelist-only metadata accessor serves the
    unlimited half, W-08 has a separate no-alpha schedule, and reusable holdout
    and adversarial label reads are uncapped but recorded. LOOK-LEDGER
    (e) also requires the ledger root to remain the injected absolute campaign
    home, never checkout-derived; (f) requires every released row and graph set
    to remain a subset of the census sourced only from the verified A15 role hash
    and frozen A16 schedule digest. Whole-partition completeness is enforced only
    for the two sealed `n <= 1` budgets. The `n <= 4` calibration paths and the
    uncapped reusable/adversarial paths may release partial in-census row sets;
    each calibration release still consumes its next ordered look, and each
    uncapped release is still recorded. The shipped seal pins (f)'s A16 digest
    in-package and reconciles the superseded 341-session plan with the delivered
    370-session campaign in stable `base_pair_id` space. The content-bound
    (a)-(d) access path is implemented before any calibration use for stopping or
    capacity selection.
48. CLOSED by ADDENDUM-27 W-13-EST. Outer weights now carry 95% intervals from
    profile likelihood on the fitted objective plus observed-information rank
    and condition number; the graph-half gate freezes failing weights at their
    priors. The JND result type requires cell/component intervals, two separate
    spread intervals, bootstrap drop rates, and the one-shot branch inputs.
49. `era_robustness` is a reporting helper, not authority to ingest pilot data.
    The authorized main bank currently contains CF@4 only, so CF@1-vs-CF@4
    deltas remain absent. `load_bank` now denies `bank/pilot` and `bank/sealed`
    even through recursive public-API inputs. OWNER: populate an authorized,
    purpose-correct multi-era diagnostic corpus before publishing era deltas.
50. P5FIX reconciles every fitted subterm, owning facet, diagnostic bit, fitted
    identity, A18 bucket, and prior floor against the complete `WeightTable`.
    It does not parse an external campaign `MANIFEST.json` because no manifest
    path or frozen `fitted_dof_declaration` schema is part of the harness API.
    OWNER: campaign wiring must pass the frozen declaration and verify its digest
    before the real fit gate is opened.
51. `V4_SPEC_r4.md` section 7.9 names U12/U13/U34 and "U11-route" for prior
    floors, while the frozen `REQUIRED_PRIOR_FLOOR_FACETS` set and validated
    `WeightTable` schema name only U12/U13/U34. P5FIX enforces the frozen table
    exactly and does not silently invent a U11 subterm-level floor. OWNER:
    contract/table owner must define the U11-route floor identity and value.
52. The P5 JND bridge can prove cross-session repetition and displayed-order
    reversal from opaque ids, but cannot prove engine blindness without the
    quarantined blind-id truth map. The JND model is fail-closed independently.
    OWNER: authorized freeze-fit orchestration must attest blind-map separation
    without exposing engine identities to the fitting module.
53. CLOSED by ADDENDUM-27 FIT-ORD. `FitPair` has no synthetic JND, lapse, or
    stratum defaults; real bridge construction supplies every field explicitly,
    and `synthetic_fit_pair` is the separate typed fixture factory.
54. CLOSED by ADDENDUM-28 RIDER-1. C-06 compares the JND block's fitted-parameter
    count `N_JND`, under the same conservative counting rule as the ledgered +2,
    against the granted allocation. The aggregate `effective_dof` scalar required
    by (f) remains published; each component trace is additionally published as a
    dimensioned `(ED_k, n_k_realized)` pair, and no trace is compared with +2 or
    any other constant. The implementation audits the realized free-coordinate
    vector, preserves conformant heterogeneous fits, freezes only extra coordinates
    not named by `(mu, tau_class, tau_band)`, and publishes an explicit PARTIAL
    declaration when a named offender cannot be frozen at its null prior. `N_JND`
    and upper-bound disclosures cover the same active JND-block membership.
55. The frozen A16 schedule's 341-session presentation-id space is disjoint from
    the delivered 370-session campaign after replanning re-minted presentation
    and session ids. The sealed roles reconcile exactly in the stable,
    LOOK-LEDGER(a-bis)-whitelisted `base_pair_id` space: 479 cross-family and
    622 within-family pairs, with zero extras in either direction. The TEST seal
    therefore enforces content subset and completeness in base-pair space while
    retaining the independent graph census check. OWNER: the next protocol
    addendum should record this already-enforced reconciliation explicitly.
56. ADDENDUM-27 FIT-ORD(e) says the pilot's coarser per-stratum lapse estimate
    enters as a prior, never a plug-in, but freezes neither a prior family nor
    its strength/effective sample size. Those choices change the fitted lapse
    and outer weights. The seven-category likelihood is implemented, but the
    low-level optimizer refuses every real row rather than silently treating a
    caller-supplied lapse as fixed or inventing a prior. OWNER: freeze the lapse
    prior family and strength before the first real profiled FIT-ORD fit; synthetic
    discriminating fixtures remain executable.
57. ADDENDUM-27 W-13-EST(e) requires graph-disjoint halves "under the frozen A15
    salt", but A15 freezes two salts for role assignment and no artifact freezes
    a graph-to-half algorithm (hash bit, alternating order, or balancing rule).
    Those choices produce different half estimates and can change which
    components/weights freeze. The complete estimator is executable on typed
    synthetic fixtures with an explicitly fixture-only salted split, while every
    real W-13 call fails closed. OWNER: freeze the graph-half assignment algorithm
    before the first real fit.
58. CLOSED by ADDENDUM-28 RIDER-2, P5FIX7, and P5FIX8. LOOK-LEDGER supports only
    append-only ANNUL entries naming the target line's exact-byte digest, reason,
    exact `(addendum, ledger_key, slot_index)` license, ISO calendar date, and
    no-release scope evidence. The production license registry is empty because no
    addendum licenses a specific correction today. A test-injected license proves
    valid annulments restore budget through generation-suffixed locks without
    removing history; void entries are disclosed. This ledger design can currently
    evidence only the `synthetic_only` manifest basis. The reservation-without-reveal
    and crash-before-publication bases require a future two-phase writer schema and
    are not advertised as live bases. `state == "RELEASED"` is positively refused
    even if a synthetic flag is present. The reviewed synthetic leak never reached
    the campaign ledger, so there was nothing to annul. `run_freeze1_fit` refuses the
    campaign root and every descendant; `evaluate_h_jnd_branch` requires an explicit
    ledger, so neither surface can silently construct the campaign ledger.
