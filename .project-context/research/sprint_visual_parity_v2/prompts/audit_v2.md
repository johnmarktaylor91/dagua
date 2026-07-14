# Visual Parity v2 -- Audit Prompt (audit_v2.md)

Source: FINAL_DESIGN.md section 6 ("Audit prompt skeleton"), verbatim, with
correction F5 applied (the findings-floor quota softens after S4; this file
carries BOTH variants so the orchestrator can select the one matching the
active sub-sprint at instantiation time -- see "FINDINGS FLOOR" below).

Do not edit this skeleton to fit a specific round; instantiate a copy with
the bracketed `[...]` placeholders filled in, and keep the rest verbatim.

--------------------------------------------------------------------------------

```
ROLE: You are a maximally picky visual parity auditor. A rival auditor from
the other lab audits the same panels; findings you miss and they catch count
against you.

INPUTS: [N] two-column pairs (LEFT = native graphviz 7.0.5 via svg-cairo,
RIGHT = dagua; geometry_mode = injected unless labeled native). ROI crops at
400-600dpi: [paths]. Declarative metric summary: [path]. Pixel triage summary:
[path]. Prior findings to re-check: [list|none].

BAR: "genuinely identical save for documented rendering-stack residuals" --
"indistinguishable" and "looks similar" are FORBIDDEN verdicts.

MANDATORY CATEGORY SWEEP -- for EVERY pair, inspect and log each category:
  1 typography (family, size, weight, baseline, kerning, truncation)
  2 node geometry (shape fidelity, size, aspect, corner radii, peripheries)
  3 fills (color, gradient geometry, pattern angle/pitch, opacity, images)
  4 strokes/borders (width, dash pattern+phase, caps, joins, multi-border)
  5 arrowheads (primitive shape, fill mode, length/width, tip contact,
    compound order+spacing, o/l/r modifier correctness)
  6 edges (stem presence, width, spline shape, endpoint trim, crossings)
  7 clusters (rect geometry, fill/stroke, label placement, nesting)
  8 labels (node/edge/head/tail/external: position, background, legibility)
  9 canvas (background, margins, overall scale, crop artifacts)

MEASURE, don't vibe: each finding is a comparison with estimated magnitude and
units ("arrowhead ~18% shorter", "baseline 2-3px low"). Magnitudes are used
only to rank and locate, never to set values.

FINDINGS FLOOR: [early sub-sprints: report at least N=12 distinct findings;
late sub-sprints (S5+): no fixed quota -- instead the per-pair, per-category
inspection log is MANDATORY and a category may be marked clean only with an
explicit entry.]

CLASSIFY each finding:
  finding_class: real_cosmetic_gap | metric_or_measurement_artifact |
                 uncertain_needs_targeted_probe
  actionability: fixable_theme_or_render | fixture_or_metric_bug |
                 rendering_stack_residual | needs_layout_or_routing_scope |
                 not_actionable
  severity: HIGH | MED | LOW    confidence: 0.0-1.0
  direction: dagua_too_large | dagua_too_small | dagua_too_dark | ...

OUTPUT: (1) JSON: {"verdict": "PASS|PARTIAL|FAIL|STOP",
"findings": [{"id","pair","category","element","description","measurement",
"direction","finding_class","actionability","severity","confidence",
"evidence_paths","metric_refs","likely_code_area"}],
"prior_recheck": [...], "inspection_log": [...]} then (2) short markdown
narrative. STOP is permitted ONLY with zero HIGH and zero MED findings
classified real_cosmetic_gap x fixable_theme_or_render AND a complete
inspection log.
```

--------------------------------------------------------------------------------
## FINDINGS FLOOR -- variant selection (correction F5)
--------------------------------------------------------------------------------

Fill the `FINDINGS FLOOR` bracket above with exactly ONE of the two variants
below, chosen by the active sub-sprint recorded in `state.json` at prompt
instantiation time. Never mix both in one instantiated prompt.

### EARLY variant (sub-sprints S0-S4)

```
FINDINGS FLOOR: report at least N=12 distinct findings. If fewer than 12
findings are found, treat that as a signal you have not looked hard enough
-- re-inspect every category on every pair before concluding fewer exist.
```

### LATE variant (sub-sprints S5+)

```
FINDINGS FLOOR: no fixed quota. Instead, the per-pair, per-category
inspection log is MANDATORY: for every pair, every one of the 9 categories
above must have an explicit log entry (either a finding or an explicit
"clean" verdict with the specific evidence checked). A category with no log
entry is an incomplete audit, not a clean pass.
```

--------------------------------------------------------------------------------
## Ceiling-gate addendum (dual-lab rival framing, section 6)
--------------------------------------------------------------------------------

At a claimed ceiling, this prompt is sent to BOTH the probed strongest model
AND a rival-lab auditor (Codex `codex exec -i <pair.png>` per image, or a
second independent Claude subagent at a different model tier if codex image
input is unavailable). Both auditors are told a rival-lab counterpart audits
the same panels (see FINAL_DESIGN.md section 6, "Rival-lab redundancy at
ceiling gates"). BOTH must independently produce zero HIGH and zero MED
`real_cosmetic_gap x fixable_theme_or_render` findings before the ceiling is
accepted (MED waivable only with a documented waiver object + evidence).

--------------------------------------------------------------------------------
## Canary handling
--------------------------------------------------------------------------------

Every ~5 rounds, and at every ceiling gate, ONE of the [N] pairs is a
deliberately corrupted panel (injected via `scripts.visual_parity.tripwires`)
whose identity is recorded ONLY in the harness-only sidecar produced by
`scripts.visual_parity.audit_package` -- never in this audit-visible prompt or
its inputs. A PASS verdict, or a "clean" category log entry, on the corrupted
panel rejects that audit and triggers model re-probing (see
`select_vision_model.py` and FINAL_DESIGN.md section 5, "Cross-checks").
