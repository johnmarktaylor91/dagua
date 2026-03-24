# Gallery Image Sizing For Claude Visual Critics

Research date: 2026-03-23
Sources:
- Anthropic vision docs: https://platform.claude.com/docs/en/build-with-claude/vision
- Local code inspection: `scripts/generate_comprehensive_gallery.py`
- Local code inspection: `scripts/generate_cosmetic_album.py`

---

## Executive Summary

1. Anthropic's `2000px` rule is a **per-image width/height limit**, not a total-pixels-across-the-request limit.
   - Normal API limit: each image must be `<= 8000x8000`.
   - If an API request contains **more than 20 images**, each image must be `<= 2000x2000`.
   - Separate limit: standard API requests still have a **32 MB total request-size cap**.

2. The more important practical limit is Claude's **auto-resize threshold**:
   - Anthropic recommends keeping images at `<= 1568 px` on the long edge and about `<= 1.15 MP`.
   - Above that, Claude downsamples the image before analysis. That costs latency and removes detail.

3. The current sweep generator regularly exceeds both thresholds that matter for critics:
   - `PANEL_SIZE = (920, 680)`
   - `SWEEP_PANEL_SIZE = (218, 158)`
   - raw render DPI is `200`
   - composed gallery save DPI is `180`
   - Current six-column sweeps export at about `2916 px` wide.
   - Current five-column sweeps export at about `2430 px` wide.
   - `21` current sweeps exceed `2000 px` in at least one dimension.

4. Recommendation:
   - For critic input, keep each uploaded image at `<= 1568 px` on the long edge.
   - Do **not** send large `5-6` column grids.
   - Prefer `3-4` panels per image for coarse visual comparisons.
   - For subtle comparisons such as text alignment or thin border differences, use `2-3` panels per image or individual images.
   - Target a critic-facing tile size of roughly `320x240` for subtle style judgments.

---

## 1. What Anthropic Actually Limits

As of 2026-03-23, Anthropic's vision docs state:

- Claude supports multiple images in one request.
- Images larger than `8000x8000` are rejected.
- If an API request contains **more than 20 images**, the per-image limit becomes `2000x2000`.
- The API may also fail because of total request size before hitting the image-count limit.
- Images with a long edge above `1568 px` are resized down before inference.
- Very small images, especially under `200 px` on an edge, may degrade quality.

Answer to question 1:

- The `2000 px` rule is **per image**.
- It is **not** a total pixel budget shared across all images in a request.
- There **is** a separate whole-request constraint, but it is a **payload-size** constraint (`32 MB` on standard endpoints), not a total-pixel rule.

Practical interpretation for gallery critics:

- Hard reject threshold for many-image API batches: `2000x2000` per image.
- Quality-preserving threshold: `<= 1568 px` on the long edge per image.
- If you care about visual detail, optimize for the `1568` rule, not the `2000` rule.

---

## 2. What The Current Gallery Script Emits

Relevant code:

- `scripts/generate_cosmetic_album.py`
  - `PANEL_SIZE = (920, 680)`
  - `RAW_RENDER_DPI = 200`
  - `ALBUM_DPI = 180`
- `scripts/generate_comprehensive_gallery.py`
  - `SWEEP_PANEL_SIZE = (218, 158)`
  - final figures are saved with `fig.savefig(..., dpi=180, bbox_inches="tight")`
  - width is driven by `fig_width = max(4.2, 2.7 * n_columns * panel_width_scale)`

Approximate current output sizes:

| Layout pattern | Approx output size |
|---|---:|
| 6 columns, 1 row | `2916x540` |
| 6 columns, 2 rows | `2916x945` |
| 5 columns, 1 row | `2430x540` |
| 4 columns, 2 rows | `1944x945` |
| 3 columns, 2 rows | `1458x945` |
| `node_shape` sweep | `2916x2637` |
| `graph_direction` sweep | `2657x945` |

Current failure implications:

- Any request with **more than 20 images** can reject many current sweep images outright, because many are over `2000 px` wide.
- Even when an image is accepted, Claude will resize many current sweeps because they exceed `1568 px` on the long edge.

Current effective tile size after Claude downscales to `1568 px` on the long edge:

| Current composite | Approx effective tile size |
|---|---:|
| 6-column sweep | about `117x85` |
| 5-column sweep | about `141x102` |
| 4-column sweep | about `176x128` |
| 3-column sweep | about `218x158` |

Those numbers matter more than the full-image size. The critic is trying to judge tiny details inside each tile, not the file as a whole.

---

## 3. Resolution Needed For The Specific Visual Judgments

Anthropic does not publish feature-specific thresholds for arrowheads, dash styles, or text alignment. The guidance below is therefore an **engineering inference** from:

- Anthropic's `1568 px` no-resize recommendation
- Anthropic's warning that sub-`200 px` images can degrade quality
- the current sweep tile geometry
- the stroke sizes already used in the gallery renders

### 3.1 Arrowhead shapes: `normal` vs `vee` vs `diamond`

These differ by silhouette and interior negative space. They are fairly robust if the arrowhead itself lands at around `24-32 px` across.

Recommended minimum:

- Tile width: `>= 240 px`
- Better target: `280-320 px`

Verdict:

- `117x85` effective tiles are too small for reliable distinctions.
- `218x158` is probably acceptable for coarse arrowhead shape checks.
- `280-320 px` wide tiles are safer.

### 3.2 Line styles: `solid` vs `dashed` vs `dotted`

The critic needs to see repeated pattern units, not just a single mark. A line segment should contain at least `3` visible dash or dot repeats.

Recommended minimum:

- Tile width: `>= 240 px`
- Visible line segment: `>= 80-100 px`

Verdict:

- `solid` vs `dashed` often survives moderate downscaling.
- `dashed` vs `dotted` is less robust and benefits from `>= 240 px` tiles.

### 3.3 Text alignment: `left` vs `center` vs `right`

This is the most fragile case because the difference is mostly whitespace distribution inside a text box.

Recommended minimum:

- Tile width: `>= 280 px`
- Better target: `320 px`
- Text block width inside the tile: `>= 100-120 px`
- Do not let the tile be auto-downscaled below about `200 px` on its short edge

Verdict:

- Current default tiles are weak for this task.
- Current six-column and five-column composites are especially poor after Claude resizes them.
- Text alignment should be shown in fewer columns or separate images.

### 3.4 Border thickness: `0.5pt` vs `3pt` vs `5pt`

At `180 dpi`, these are roughly:

- `0.5 pt` -> `1.25 px`
- `3 pt` -> `7.5 px`
- `5 pt` -> `12.5 px`

That means the thin border is only about a one-pixel signal before any further downscaling.

Recommended minimum:

- Tile width: `>= 260 px`
- Better target: `300-320 px`
- Avoid any workflow that causes another significant downscale after export

Verdict:

- `3 pt` vs `5 pt` is usually safe.
- `0.5 pt` vs thicker borders becomes unreliable once the tile is shrunk.
- Border-width sweeps should be treated like text-alignment sweeps: fewer panels, larger tiles.

---

## 4. Optimal Size For A Single Critic Image

Assumption: "single sweep panel" here means one image sent to the critic for one sweep, not one raw subpanel inside the Matplotlib grid.

Recommended target for one critic image:

- Long edge: `1400-1568 px`
- Short edge: `900-1200 px` when needed
- Keep the full image under about `1.15 MP` when possible

Recommended internal layout:

- Coarse style sweeps: `3-4` panels per image
- Fine detail sweeps: `2-3` panels per image
- Tile target for subtle judgments: about `320x240`

Why this is the best balance:

- It stays under Claude's no-resize threshold.
- It avoids the `>20 images => 2000x2000` rejection problem.
- It gives each tile enough pixels for small style differences.

Not recommended:

- `5-6` columns in a single critic image
- stacked Dagua/Graphviz rows plus many columns in the same file for subtle comparisons
- huge master grids intended for human browsing

---

## 5. Should Multi-Panel Sweeps Be Split?

Yes, but not blindly into one file per cell.

Best practice:

- Split large sweeps into **small groups** of related panels.
- Keep each image to `2-4` panels.
- Batch multiple requests if needed instead of stuffing dozens of images into one request.

Reasoning:

- Large grids force Claude to analyze tiny subregions after downscaling.
- Fully splitting into one image per value can improve readability, but it also raises image count and may push the request past the `20-image` threshold that triggers the `2000x2000` per-image rule.
- Small grouped images are the best compromise.

Recommended split policy:

| Sweep type | Recommendation |
|---|---|
| Arrowheads, line styles | group `3-4` panels per image |
| Text alignment | group `2-3` panels per image, or individual images |
| Border thickness | group `2-3` panels per image |
| Large multi-row sweeps | always split |
| Dagua vs Graphviz comparisons | prefer separate images or very small paired groups |

---

## 6. Concrete Recommendation For Dagua

If the goal is "Claude can reliably critique these images in multi-image review jobs", the safest export profile is:

- Critic image long edge: `<= 1568 px`
- Critic tile size: around `320x240`
- Max columns: `4` for coarse comparisons
- Max columns for subtle comparisons: `2-3`
- Avoid multi-row composites when the judgment depends on text placement or thin strokes
- Prefer multiple smaller critic batches over one very large image batch

In practice, that means:

1. Keep the current large gallery outputs for humans.
2. Add a separate critic-oriented export mode.
3. In that mode, split current `5-6` column sweeps into multiple images.
4. Treat `text_alignment`, `border_width`, and similar subtle sweeps as high-detail cases.

---

## Short Answers To The Research Questions

### 1. Actual Claude limit?

- Hard limit is per-image, not total pixels across the request.
- Normal: `8000x8000` per image.
- More than `20` images in one API request: `2000x2000` per image.
- Practical quality limit: `<= 1568 px` long edge per image.

### 2. What resolution is needed?

- Arrowheads: tile `>= 240 px`, preferably `280-320 px`
- Line styles: tile `>= 240 px`
- Text alignment: tile `>= 280 px`, preferably `320 px`
- Border thickness: tile `>= 260 px`, preferably `300-320 px`

### 3. Optimal size for a single sweep image?

- About `1400-1568 px` on the long edge
- Use `2-4` tiles inside it
- Target about `320x240` per tile for subtle judgments

### 4. Split multi-panel sweeps?

- Yes for critic workflows.
- Prefer several small grouped images over one giant grid.
- Do not automatically explode everything into one image per cell if that would push the request over `20` images.
