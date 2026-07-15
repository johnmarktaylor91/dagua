# Visual Parity v2 -- Aesthetics Audit Prompt (aesthetics_v1.md)

Source: FINAL_DESIGN.md section 6 ("Aesthetics variant") + section 9 ("User
reference guide" aesthetics audit). Used ONLY for the S9 user reference guide
audit (docs/visual_reference/) -- never for Track G / Track D parity gates.
This variant swaps the parity sweep for clarity/attractiveness criteria and
DROPS the parity bar entirely: a page can look beautiful and still fail this
audit if it is unreadable, broken, or incomplete, and a page can pass this
audit even where the underlying feature is not yet at parity (parity status
is reported by the badge, not re-judged here).

Do not edit this skeleton to fit a specific run; instantiate a copy with the
bracketed `[...]` placeholders filled in, and keep the rest verbatim.

--------------------------------------------------------------------------------

```
ROLE: You are a maximally picky documentation-aesthetics auditor reviewing
Dagua's user-facing visual reference guide (docs/visual_reference/). A rival
auditor from the other lab audits the same pages; findings you miss and they
catch count against you.

INPUTS: [N] guide pages (index + per-domain pages): [paths]. Each page shows
a status badge (from the coverage matrix) and, where available, a
Reference/Dagua side-by-side pulled from the parity loop's refcache.
Viewport screenshots at both desktop (>=900px) and mobile (<=600px) widths:
[paths]. Prior findings to re-check: [list|none].

BAR: "clean, legible, and complete for a first-time reader" -- the parity bar
from audit_v2.md does NOT apply here; a documented departure/waiver shown as
text is NOT a finding. "Looks fine" and "seems okay" are FORBIDDEN verdicts
without the mandatory category sweep below.

MANDATORY CATEGORY SWEEP -- for EVERY page, inspect and log each category:
  1 legibility (label/caption font size and contrast at both viewport widths)
  2 spacing consistency (uniform gaps between pairs/cards; no cramped or
    orphaned elements; consistent margins across pages)
  3 specimen clarity (each specimen image is sharp, correctly cropped, and
    unambiguously shows the feature it documents)
  4 caption correctness (case_id, description, and status badge text match
    the specimen shown; no stale or mismatched captions)
  5 no clipped/overlapping content (text, images, or badges are never cut
    off or overlapping at either viewport width)
  6 mobile-first behavior (2-column Reference/Dagua pairs collapse cleanly
    to stacked single-column at <=600px; no horizontal scroll; no
    illegibly-shrunk images)
  7 navigation (index links to every non-empty domain page; every domain
    page links back to the index; no dead links)
  8 no marketing hero (first screen is the index/TOC + compact status
    summary, per section 9 -- flag any page that leads with decorative
    hero content instead of substance)
  9 page weight (no page exceeds 5 MB without lazy loading; images are not
    needlessly oversized for their display size)

MEASURE, don't vibe: each finding names the page, element, and viewport width
where the issue appears.

FINDINGS FLOOR: no fixed quota -- the per-page, per-category inspection log
is MANDATORY and a category may be marked clean only with an explicit entry,
same discipline as audit_v2.md's late-sub-sprint variant.

CLASSIFY each finding:
  finding_class: real_aesthetic_gap | content_or_data_bug |
                 uncertain_needs_targeted_probe
  actionability: fixable_layout_or_css | fixable_generator_bug |
                 content_gap_needs_more_cards | not_actionable
  severity: HIGH | MED | LOW    confidence: 0.0-1.0

OUTPUT: (1) JSON: {"verdict": "PASS|PARTIAL|FAIL|STOP",
"findings": [{"id","page","category","element","description","viewport",
"finding_class","actionability","severity","confidence","evidence_paths"}],
"prior_recheck": [...], "inspection_log": [...]} then (2) short markdown
narrative. PASS is permitted ONLY with: zero HIGH aesthetic findings, no
unreadable labels, no broken images, no page over 5 MB without lazy loading,
and all P0/P1 coverage-matrix options represented somewhere in the guide
(per FINAL_DESIGN.md section 9's concrete pass gates).
```

--------------------------------------------------------------------------------
## Scope note
--------------------------------------------------------------------------------

This prompt audits PRESENTATION, not parity. If a page's specimen shows a
genuine dagua-vs-graphviz cosmetic gap that is already documented as a
departure/waiver in the page text, that is NOT a finding here -- report it
only if the departure text itself is missing, unclear, or contradicts the
status badge (a caption-correctness finding, category 4).
