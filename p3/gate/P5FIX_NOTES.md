# P5FIX notes

Base: `90236ec0`. Reviews: Fable and the storm-delayed Opus review, both read
from the shared research gate because neither review file was copied into this
worktree. Recovery ruling: noise floor; objective recovery math was not changed.

## Fable disposition

- F1 BLOCKER: fixed. TEST labels are absent from loaded objects, opaque through
  partition/pickle surfaces, re-read only after content-bound one-shot
  reservation, and fitting refuses non-FIT/replication rows.
- F2-F3: fixed as one ledger-boundary change. Floors are mandatory for fitted
  traceability facets and apply to effective coefficient mass; subterm/facet,
  DIAG, fitted identity, bucket, and frozen-table declarations are reconciled.
- F4: finding accepted; silent collapse is removed. Verdict magnitude and
  confidence are retained and real graded rows fail closed. Docket entry 45 is
  required because `V4_SPEC_r4.md` section 7.6 says the seven-point scale feeds
  ordered probit, while section 4.2 still says the latent mechanism is "P2
  synthesis pending". Selecting thresholds/link here would be an unauthorized
  methodology ruling.
- F5: finding accepted; the nonconforming fixed-L2 estimator was removed. Docket
  entry 46 cites `PREREG_V4_CALIBRATION.md` W-13 and
  `A18_INSERT_W13.md` sections 1-3: fitted variance components, split-half gate,
  cell CIs, bootstrap spread CI, `K_JND=3.0`, and a numerically unspecified
  "preregistered minimum" are all mandatory. Inputs are validated, then the
  incomplete model fails closed.
- F6-F8: fixed (implicit abstain/tie consistency, profile fence, and replication
  exclusion from weight fitting).
- F9: docket 47 (calibration four-look ledger).
- F10: fixed (`FitResult.at_bounds`).
- F11: global RNG/determinism mutation fixed; synthetic defaults docketed in 53.
- F12: fixed. The live home-directory test was removed and recovery now pins the
  fixed sample MLE, not population truth at a sub-SE tolerance.

## Opus fold

- B1-B4: fixed: evidence-scaled priors, pilot/sealed subtree denial, nonempty
  content-bound holdout consumption, and actual cross-session side-swap checks.
- B5: **disputed in part, neutralized in full**. The old point-effects optimizer
  is removed. Opus counts every latent class/band effect as an independent dof,
  but the frozen allocation says otherwise: `PREREG_V4_CALIBRATION.md` section 7
  assigns `N_g=4` exactly to `mu`, `tau_class`, `tau_band`, and lapse, and
  `A18_INSERT_W13.md` section 3 guard 5 says the two heterogeneity variance
  components are the `+2` dof and shipping cell bands adds none. P5FIX follows
  that frozen accounting and will not invent a per-level ledger.
- M1-M5: handled: graded-model fail-close, purpose-safe fit access, effective
  prior floors, frozen ownership checks, exact scale-identifiability refusal,
  and bound flags.
- M6: accepted and docketed in 48; real fit paths remain closed until uncertainty
  machinery exists. Calibration looks are docket 47.
- M7: accepted as an activation limitation, not permission to read pilot data;
  docket 49. No current result claims a CF@1/CF@4 delta.
- M8: frozen `WeightTable` fitted-declaration cross-check fixed; external manifest
  plumbing remains docket 50 because the current public API has no manifest
  input/schema.
- Mechanical minors fixed: sample-MLE recovery pin; host RNG preservation;
  post-update JND loss-path defect removed with the invalid optimizer;
  `dataclasses.replace` calibration; side-bit semantics comment and distinct
  validation error; strict schedule schema; tie/verdict consistency; opaque
  holdout repr/pickle surface; mutable external campaign test removed.
- Remaining mechanical-else-docket items: four-look enforcement (47), outer
  uncertainty (48), U11-route floor authority mismatch (51), blind-map
  attestation (52), and synthetic-only defaults (53).

## Review repro coverage

`tests/eval/ruler_v4/test_fit_harness.py` permanently covers the four holdout
defeats, empty-partition burn, purpose/replication/profile leakage, prior scaling,
floor dilution/omission, DIAG/facet/identity smuggling, graded collapse, pilot and
sealed traversal, implicit abstain and contradictory labels, schedule truncation,
real replication provenance, unidentified P-mean scale, bound flags, global RNG
state, calibration replacement, and fixed-sample recovery behavior.

## Assumptions

- "tests/eval scope only" was interpreted as the P5 harness under
  `dagua/eval/ruler_v4/fit`, its `tests/eval` regressions, and the explicitly
  requested `dagua/eval/ruler_v4/DISCREPANCIES.md`/`p3/gate` handoff artifacts.
- Fable's 12 findings remain the marker's primary blocker/major count; Opus is
  folded and itemized separately because it arrived during the pass.
