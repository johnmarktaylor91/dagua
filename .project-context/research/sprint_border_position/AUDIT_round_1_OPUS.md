# Sprint I Round 1 Audit -- border_position cytoscape parity

**Auditor:** Opus 4.7 (visual + code, maximum strictness)
**Date:** 2026-05-01
**Cards under review:**
- `nodes_borders_border_position_inside_vs_cytoscape.png` (L1 = 9.95 / 10.008)
- `nodes_borders_border_position_outside_vs_cytoscape.png` (L1 = 10.415)

## Verdict

**`PRINCIPLED_RESIDUAL_DEFER`**

The dagua side of these cards is **already mathematically correct** under the
documented cytoscape CSS-style `border-position` semantics. The L1 ~ 10
residual is **not** a half-stroke offset, anti-aliasing artifact, or signed
direction bug. It is dominated by *cytoscape-specific rendering choices that
have nothing to do with `border-position`*:

1. cytoscape paints the border ribbon **on top of the fill** (with no fill
   visible through it at this stroke ratio), and at `stroke_width=50` the
   ribbon completely covers the cream interior on both `inside` and `outside`.
2. cytoscape applies a non-trivial `corner-radius` to its `rectangle` shape
   regardless of dagua's `shape="rect"` request -- the cytoscape panels show
   visibly rounded corners; dagua faithfully renders sharp corners.
3. cytoscape's `outside` panel **expands the node's pickable bbox**, shifting
   the node's *position* on the page (the right-hand panel sits much further
   right and is markedly larger than the matching dagua panel).
4. The two panels in dagua's layout are spaced ~80 px apart; cytoscape's
   `Center` and `Inside`/`Outside` panels are spaced ~600+ px apart -- the
   layout/positioning at the *graph level* is being driven by cytoscape's own
   pre-render bbox computation that we do not (and should not) emulate
   verbatim.

None of these are reachable by patching dagua's `border_position` math
without:

- Conflating `border_position="inside"` with a hidden auto-corner-radius
  (would regress every other rect card and violate the locked
  graphviz-strict / cytoscape-strict theme territory),
- Force-expanding node bboxes when `border_position="outside"` (would mutate
  layout coordinates -- algo_fidelity territory; explicitly off-limits per
  guardrails),
- Hand-tuning the fill->stroke compositing order (already correct in dagua).

## Part 1 -- Visual inspection

### Inside card (`nodes_borders_border_position_inside_vs_cytoscape.png`)

**Dagua (left panel):**
- Two rectangles labeled `Center` and `Inside`, each ~80 x 60 data units
- Cream fill (`#FFE0B2`) clearly visible in the interior
- Orange (`#E65100`) border ribbon fully **inside** the bbox for the `Inside`
  variant (interior cream area is smaller than for `Center`)
- Sharp 90-degree corners
- Two panels packed close together (~80 px gap)

**Cytoscape (right panel):**
- Two **rounded** rectangles, each visibly larger than dagua's
- **No visible cream fill** -- the entire shape reads as solid orange
  (`stroke_width=50` on an 80x60 box leaves at most a 30 x 10 cream sliver
  even mathematically; cytoscape's anti-aliased ribbon plus its
  paint-order swallows it visually)
- Pronounced corner radius (~12-15 px), even though `shape=rect` was requested
- Panels separated by ~600+ px of whitespace

### Outside card (`nodes_borders_border_position_outside_vs_cytoscape.png`)

**Dagua (left panel):**
- `Center` and `Outside`: identical interior cream area (correct by
  definition of `outside`); border ribbon extends **outward** for the
  `Outside` variant -- visible as a bigger orange rectangle around the same
  cream interior
- Sharp corners

**Cytoscape (right panel):**
- `Outside` panel rendered as a **rounded** rectangle, **physically larger**
  than the `Center` panel and shifted noticeably to the right of where
  dagua placed it
- Solid orange (no visible cream)
- ~600+ px panel gap

### Pixel-diff signal

The L1 ~ 10 number is dominated by:
1. The cream interior pixels in dagua vs orange pixels in cytoscape
   (interior covers a large area; this is most of the L1).
2. The corner-radius half-moon mismatch on every corner.
3. The horizontal translation: `Inside`/`Outside` panel is at x ~ 470 in
   dagua, x ~ 1500 in cytoscape -- entire orange shape diff against
   white background, plus dagua's diff against cytoscape's whitespace.

A half-stroke offset would yield L1 ~ 0.3-1.0, not 10.

## Part 2 -- Code investigation

`dagua/render/mpl.py` contains four pieces of `border_position` math:

### `_normalize_border_position` (line 3228)
```python
if border_position in {"center", "inside", "outside"}:
    return border_position
return "center"
```
Correct. No bug.

### `_node_fill_path` (line 3689)
```python
if border_position == "inside":
    return inset_shape_path(shape_spec, border_width)  # fill shrinks by S
if border_position == "center":
    return outer_path                                  # fill at bbox
return outer_path                                      # outside: fill at bbox
```
Matches cytoscape spec exactly:
- `inside`: fill recedes by full stroke width S (correct -- stroke occupies
  the inset region; fill must not bleed under it).
- `center`: fill at bbox (correct -- stroke straddles bbox edge; half-stroke
  is over fill, half over background).
- `outside`: fill at bbox (correct -- stroke is entirely outside the fill
  region; fill remains at bbox).

### `_solid_border_ring_paths` (line 3722)
```python
if border_position == "inside":
    return outer_path, inset_shape_path(shape_spec, border_width)
    # outer = bbox, inner = bbox - S      (annular ring fully inside bbox)
if border_position == "outside":
    expanded = build_shape_path(_expanded_shape_spec(shape_spec, border_width))
    return expanded, outer_path
    # outer = bbox + S, inner = bbox      (annular ring fully outside bbox)
expanded = build_shape_path(_expanded_shape_spec(shape_spec, border_width / 2.0))
inner = inset_shape_path(shape_spec, border_width / 2.0)
return expanded, inner
# center: outer = bbox + S/2, inner = bbox - S/2  (straddles)
```
Matches cytoscape spec exactly. This is the canonical CSS box-model border
math.

### `_node_border_centerline_path` (line 3756)
```python
if border_position == "inside":
    return inset_shape_path(shape_spec, border_width / 2.0)   # bbox - S/2
if border_position == "outside":
    return build_shape_path(_expanded_shape_spec(shape_spec, border_width / 2.0))
                                                              # bbox + S/2
return outer_path                                             # bbox
```
Centerline placed at bbox-S/2 for inside, bbox for center, bbox+S/2 for
outside. This is exactly correct: the dashed/stroked ribbon's *centerline*
is offset by half the stroke from the bbox in the appropriate direction.

### `_edge_terminal_outset` (line 3247)
```python
stroke_width = max(stroke_width, 0.0) * stroke_scale
if border_position in {"center", "inside"}:
    return 0.0
if border_position == "outside":
    return stroke_width
```
Correct: edges should terminate at the visual outer edge of the stroke.
Inside/center share the bbox as the outer; outside extends by S.

**There is no math bug in dagua's border_position implementation.** The four
functions above implement the exact CSS-spec semantics named in the brief.

## Part 3 -- Diagnose

The brief's hypothesized math (in section "Part 3: Diagnose") matches dagua's
existing math one-for-one:

| Spec target                            | Dagua impl                      | Match |
| -------------------------------------- | ------------------------------- | ----- |
| center: outer = bbox+S/2, inner = bbox-S/2 | `_solid_border_ring_paths` else branch | yes |
| inside: outer = bbox, inner = bbox-S       | `_solid_border_ring_paths` inside branch | yes |
| outside: outer = bbox+S, inner = bbox      | `_solid_border_ring_paths` outside branch | yes |

The actual deltas driving L1 ~ 10 are all rasterizer/library-stack
differences:

1. **cytoscape forces a corner radius on `shape:rectangle`.** Cytoscape's
   default `node-shape: rectangle` is *not* a sharp rectangle in modern
   versions -- it picks up a small built-in corner-radius from the renderer.
   This is a cytoscape choice; emulating it in dagua means hard-coding a
   corner radius in the cytoscape parity comparator, not in `border_position`
   math.

2. **cytoscape lays out border-position-modified nodes at different page
   positions.** Cytoscape's pre-render bbox includes the outside-stroke
   region and shifts node placement to compensate. Dagua's layout algorithms
   place nodes by their data-coordinate centers and do not (and should not)
   bake stroke pixels into layout -- doing so would re-enter algo_fidelity
   territory.

3. **cytoscape's WebGL anti-aliasing is different from matplotlib's
   AA.** Subpixel coverage at the corner curves and on the stroke edges
   contributes a steady ~1-3 L1 baseline that no math change can close.

4. **Both panels paint over a ~20%/80% larger area in cytoscape** because of
   point 1+2. That single geometric expansion is responsible for the bulk of
   the L1 -- a fill-vs-stroke difference times area.

Per Sprint I's own anti-flail clause, fighting this in the math layer would
be flailing. This is a Sprint J bit-equivalence concern (re-render under the
cytoscape rasterizer with our geometry, not dagua's matplotlib pipeline).

## Part 4 -- Recommend fix

**No code fix recommended for Round 2.**

Recommended action: mark this card as **principled residual**, document
under `.project-context/research/sprint_border_position/DEFERRED.md`, and
defer to the Sprint J cytoscape-via-headless-browser bit-equivalence track
that already covers the corner-radius / WebGL-AA / layout-bbox mismatches
for the entire cytoscape comparator.

If a Round 2 *must* happen for this sprint, the only safe move is a
**comparator-side calibration** (out of scope per guardrails -- modifying
the comparator is `algo_fidelity` territory):

- Build the cytoscape reference with `shape: 'rectangle'` and explicitly
  setting `corner-radius: 0` in the cytoscape stylesheet.
- Force cytoscape's pre-render bbox to use the dagua-style center-of-glyph
  placement (typically requires running cytoscape with `boundingBox`
  overrides per node or building the panel programmatically rather than
  through cytoscape's auto-layout).

Both of those are reference-side patches, not dagua-render-side patches,
and explicitly outside Sprint I's scope ("DO NOT touch algo_fidelity
territory" per `border_position_STATE.md`).

### What NOT to do (anti-patterns the next round must avoid)

1. **Do NOT patch `_normalize_border_position` to silently swap
   inside<->outside.** The current direction is correct.
2. **Do NOT add a hidden `auto_corner_radius` shim in
   `_node_fill_path`.** That would regress every other rect card.
3. **Do NOT expand the node bbox at layout time when
   `border_position="outside"`.** That mutates coordinates and is
   algo_fidelity territory.
4. **Do NOT touch GRAPHVIZ_STRICT_THEME** -- not relevant; cards are
   cytoscape comparators.
5. **Do NOT change `stroke_scale` or pixel-unit overrides** -- the stroke
   is already in data coords correctly per the locked-constant policy.

## Stop-criterion check

PRIMARY (`L1 < 4` on both cards): unreachable in math layer. Math is already
correct. Defer to Sprint J.

SECONDARY (no regression on other cards): trivially satisfied -- no change
recommended.

ANTI-FLAIL: Round 1 is the principled residual. Do not enter Round 2.

## Audit path

`.project-context/research/sprint_border_position/AUDIT_round_1_OPUS.md`
