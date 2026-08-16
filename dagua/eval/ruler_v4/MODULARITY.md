# RULER V4 module boundaries

`scene.py` and `ingestion.py` own immutable validated inputs, `registry.py` dispatches the 45 independent contract implementations, `weight_table.py` carries explicit P5-supplied sub-term mass and provenance plus dof, allocation-bucket, and PM-1 accounting (`weights.py` is the U35-U37 declared-edge-weight facet module and nothing else), `composition.py` turns VALUE/NA facet rows into the exact monotone pre-map loss with published per-group partial derivatives and applies the CC-1 margin rule's jump-bound arms, `headline.py` provides only the RENAME-compliant ordinal map, and `score.py` is the deterministic no-I/O orchestration seam that publishes separated TYPE-R context and TYPE-M measurements. Phase 3 can therefore compile each facet and composition operation behind the same `FacetResult` and `CompositionResult` shapes, propagate certified per-term intervals through `compose`, race candidates on `L_total`, and register the finalized scorer without changing ingestion, attribution, weight provenance, or headline semantics. The event-margin comparison is deliberately NOT frozen for phase 3: `compare_with_event_margin` certifies only the jump-bound-sum arm plus the optional paired-SE gate, and its `MARGIN_RULE_*` verdicts must be WRAPPED by the 4.1/4.3 uncertainty-ledger and JND/posterior machinery before any consumer publishes a strict win (DISCREPANCIES entry 34).

## Phase-3 seams (the 6.2/6.5 machinery)

**`_tracing.py` owns the traced-execution seam, and that seam is an ambient,
dynamically scoped side channel -- recorded here precisely because it is invisible in
every call signature.** `_TRACE_BUFFER` is a `ContextVar`; inside a
`trace_subterms()` context, `keep()` returns live tensors instead of casting and
`value_result` (scene.py) records each tensor-valued subterm into the active buffer
while still publishing detached floats behind the frozen `FacetResult` shape. The
type-polymorphic scalar helpers (`p_sqrt`, `p_exp`, `p_log`, `p_log1p`, `p_abs`,
`p_min`, `p_max`, `p_fsum`, `p_sum`, `keep`, `as_float`) make the return TYPE of
score-visible facet arithmetic depend on that ambient context across the eight traced
facet modules. Their contract: **float branches execute byte-for-byte the historical
operations (the exact path is bit-identical to the pre-surrogate implementation);
tensor branches are the autograd equivalents; traced forwards may differ from exact by
accumulation order only, and that gap is measured (ULP scale), never assumed zero.**
`as_float` is the one-way read for control flow and raw statistics: it detaches, so
branch decisions never carry gradient. No smoothing is introduced at this seam;
relaxations live in the facet closed forms where the contracts pin them
(DISCREPANCIES entry 38). The package uses no threads/async, so a single ContextVar
cannot be silently invisible to a worker; any future parallel evaluation must
propagate the context or lose tracing loudly (buffer absent -> exact behavior).

**`surrogate/` owns the compiled surrogate.** `manifest.py` compiles the frozen
contract graph into per-subterm traces with a source digest; `scorer.py`'s
`score_v4_soft` binds caller-supplied differentiable defect coordinates to ACTIVE
exact rows only (applicability always comes from the exact composition) and
recomposes the frozen family differentiably, including the p-mean origin
linearization (all bound terms exactly zero) and the scale-factored underflow branch;
`traced.py` orchestrates end-to-end position gradients: `build_traced_scene` re-links
derived geometry to a position leaf (node boxes ride owners; node-label boxes keep
constant offsets; chord-identical routes follow endpoints; non-chord routes,
edge-label and cluster-label boxes stay input-owned constants), and
`score_scene_soft` runs the facet stack once exactly (applicability + reference) and
once inside a trace (the buffer), then binds buffer tensors into `score_v4_soft`.
Rows without traced tensors are published as exact-value constants
(`constant_subterms`), never silently. The per-row gradient-channel status lives in
`surrogate/CLASSIFICATION.md` and is conditional on the straight-route model.

**`certification.py` owns evidence, not selection:** certified intervals propagated
through the exact `compose` at hyperrectangle endpoints (coordinatewise monotonicity
makes endpoint evaluation a valid bound), fail-closed Kendall rank fidelity
(zero comparable pairs never certifies), and scalar gradient-sanity probes. The
pilot-bank tau floor is vacuous by construction; the falsifiable population and the
drift control live in `tests/eval/ruler_v4/test_certification_discriminating.py`
(entry 39).

**`racing.py` owns single-round candidate racing:** the marginal-bound sufficient
rule and the paired-difference certificate with the single-counted oscillation charge
(the 6.2a cancellation deviation, its zero measured power gap at pilot scale, and the
curvature-bound alternative are entry 43). No anytime-valid confidence machinery
exists in phase 3; `confidence_id` is a consistency label, not a checked property,
and no production selection may rely on `race_candidates` before P5's
racing-activation stage (entry 40).
