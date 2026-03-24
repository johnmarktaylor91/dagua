# Gallery Design Specification for Dagua

Date: 2026-03-23

Goal: define a gallery structure that is exhaustive enough for rendering
coverage, structured enough for LLM critics, and practical enough for fast
iteration on visual quality.

---

## External Patterns Worth Copying

### Matplotlib

Source: <https://matplotlib.org/stable/gallery/index>

Observed structure:
- Large gallery organized by category.
- Each example is a small, self-contained unit.
- Users can browse by tags.
- Each example links to both the rendered result and source code.

Takeaway for dagua:
- Keep categories obvious and shallow.
- Treat each gallery item as a stable, addressable artifact.
- Add tags and machine-readable metadata, not just images.

### Graphviz

Source: <https://graphviz.org/gallery/>

Observed structure:
- Gallery is grouped by graph type and visual topic.
- Examples are concrete and named by the thing they demonstrate.
- Output is closely tied to the example source.

Takeaway for dagua:
- Prefer named, canonical examples over abstract buckets.
- Preserve the exact rendering recipe for every artifact.

### D3

Source: <https://d3js.org/what-is-d3>

Observed structure:
- D3 documentation is organized around modules and primitives.
- Examples are compositional rather than exhaustive.
- The API index and examples are separate but connected.

Takeaway for dagua:
- Separate the exhaustive reference from the inspirational showcase.
- Organize by primitive families first: nodes, text, edges, clusters, graph.

### Plotly

Source: <https://plotly.com/javascript/reference/index/>

Observed structure:
- Exhaustive reference is derived from a machine-readable schema.
- Attributes are organized by trace type and layout.
- JSON structure is first-class, not an afterthought.

Takeaway for dagua:
- The gallery should have a machine-readable manifest.
- JSON sidecars should be canonical, not optional.

---

## Recommendation Summary

Use a hybrid structure:

1. **Atomic cards** are the canonical unit for review.
   One render case per image file.
2. **Board images** are generated summaries for humans.
   One feature family or combination family per contact sheet.
3. **Comparison triptychs** are the canonical competitor artifact.
   One case, three normalized panels: Dagua, Graphviz, Matplotlib.
4. **Per-image JSON sidecars** store full provenance and parameters.
5. **A gallery-wide JSONL index** supports LLM ingestion, filtering, and
   regression tooling.

This is better than either extreme:

- Not `1 feature value = only 1 standalone file` with no grouping.
  That explodes file counts and makes human browsing painful.
- Not `everything in giant multi-panel boards`.
  That makes LLM review and automated comparisons fragile.

The correct unit is:
- **atomic card for truth**
- **board for navigation**

---

## Recommended Directory Structure

Use two roots:

- `docs/gallery/` for stable, browsable, user-facing artifacts
- `eval_output/gallery_audit/` for volatile review and regression artifacts

Within each root, keep the same internal schema so tooling can target either
location.

```text
gallery/
  index.html
  index.json
  index.jsonl
  README.md

  fixtures/
    fixture_index.json

  cards/
    reference/
      nodes/
        shapes/
        fills/
        borders/
        text/
        effects/
      edges/
        routing/
        arrows/
        labels/
        styles/
      clusters/
        containers/
        labels/
        depth/
      graph/
        background/
        direction/
        margins/
    combos/
      2way/
      3way/
      4way/
      5way/
    comparisons/
      reference/
      combos/

  boards/
    reference/
      nodes/
      edges/
      clusters/
      graph/
    combos/
      2way/
      3way/
      4way/
      5way/
    comparisons/

  meta/
    cards/
    boards/
    comparisons/

  overlays/
    comparisons/
      reference/
      combos/
```

For dagua specifically:
- map stable gallery output to `docs/gallery/`
- map iteration-only comparisons, overlays, and critic runs to
  `eval_output/gallery_audit/`

---

## File Types and Their Roles

### 1. Atomic cards

Purpose:
- precise LLM review
- precise regression diffs
- source-of-truth render for a single case

Examples:
- one node shape value
- one arrowhead value
- one text alignment value
- one 3-way combination case

### 2. Board images

Purpose:
- fast human scanning
- category-level browsing
- release documentation

Examples:
- all node shapes on one sheet
- all arrowheads on one sheet
- all 2-way combos for `shape + fill`

Board images should be generated from atomic cards, not rendered separately.
That keeps board layout separate from rendering truth.

### 3. Comparison triptychs

Purpose:
- one normalized artifact for Dagua vs Graphviz vs Matplotlib
- consistent critic input
- consistent side-by-side human judgment

Triptychs are separate from boards because they are review artifacts, not
reference sheets.

### 4. Overlay diagnostics

Purpose:
- internal debugging only
- alignment checks
- route/spacing inspection

Do not make overlays the primary competitor artifact. They are too hard to read
and too ambiguous for LLM scoring.

---

## Answer to the Main Structural Questions

### 1. Should each feature get its own standalone image?

Recommendation:
- **Yes for the canonical atomic case**
- **Also yes for a grouped board derived from those atomic cases**

Concretely:
- each feature value gets its own atomic card
- each feature family gets one board image

Example:
- `cards/reference/nodes/shapes/ref__nodes__shape__circle__dagua.png`
- `cards/reference/nodes/shapes/ref__nodes__shape__hexagon__dagua.png`
- `boards/reference/nodes/board__nodes__shape__dagua.png`

Why:
- atomic cards are best for LLMs and automated diffs
- boards are best for humans comparing ranges

### 2. What metadata should accompany each image?

Recommendation:
- every image gets a same-basename `.json` sidecar
- all images also appear in `index.jsonl`

This copies the best part of Plotly: machine-readable structure.

### 3. How should competitor comparisons be structured?

Recommendation:
- primary artifact: side-by-side triptych
- secondary artifact: separate raw renders
- optional debug artifact: overlay

Never make overlay the only comparison view.

### 4. What directory structure best supports both humans and LLMs?

Recommendation:
- separate `cards/` from `boards/`
- separate `reference/` from `combos/` from `comparisons/`
- keep a shallow category tree with stable slugs

This makes browsing simple and machine parsing deterministic.

---

## Image Naming Convention

Use lowercase slugs with double-underscore separators.

Pattern:

```text
{kind}__{scope}__{family}__{variant}__{backend}.png
```

Where:
- `kind`: `ref`, `combo2`, `combo3`, `combo4`, `combo5`, `compare`, `board`, `overlay`
- `scope`: `nodes`, `edges`, `clusters`, `graph`, or a comparison scope
- `family`: stable feature family slug such as `shape`, `arrowhead`, `text_align`
- `variant`: stable case slug such as `circle`, `vee`, `center`, or a joined combo slug
- `backend`: `dagua`, `graphviz`, `mpl`, or `dagua-graphviz-mpl`

Examples:

```text
ref__nodes__shape__circle__dagua.png
ref__edges__arrowhead__vee__dagua.png
combo2__nodes__shape-fill__roundrect-linear__dagua.png
combo4__edges__routing-arrow-label-style__bezier-vee-mid-dashed__dagua.png
compare__reference__arrowhead__vee__dagua-graphviz-mpl.png
board__nodes__shape__all__dagua.png
overlay__reference__arrowhead__vee__dagua-graphviz.png
```

Rules:
- Never encode display text in filenames.
- Never use spaces.
- Never include timestamps in asset filenames.
- Put timestamps, git SHA, and seeds in metadata only.

---

## Per-Image Metadata Format

Every image gets a sidecar:

```text
same/path/as/image.png
same/path/as/image.json
```

Recommended schema:

```json
{
  "schema_version": 1,
  "id": "compare__reference__arrowhead__vee__dagua-graphviz-mpl",
  "title": "Arrowhead comparison: vee",
  "kind": "comparison_triptych",
  "section": "comparisons/reference",
  "tags": ["edges", "arrowhead", "vee", "comparison"],
  "image": {
    "path": "cards/comparisons/reference/compare__reference__arrowhead__vee__dagua-graphviz-mpl.png",
    "width_px": 2400,
    "height_px": 900,
    "background": "#ffffff",
    "format": "png"
  },
  "fixture": {
    "id": "edge_arrow_single",
    "version": 1,
    "node_count": 2,
    "edge_count": 1,
    "cluster_count": 0
  },
  "render_context": {
    "direction": "LR",
    "seed": 42,
    "renderer": "mpl",
    "dpi": 200
  },
  "backends": [
    {
      "name": "dagua",
      "variant_role": "primary",
      "params": {
        "arrowhead": "vee",
        "routing": "bezier",
        "stroke_width": 2.0
      }
    },
    {
      "name": "graphviz",
      "variant_role": "competitor",
      "params": {
        "arrowhead": "vee"
      }
    },
    {
      "name": "mpl",
      "variant_role": "competitor",
      "params": {
        "arrowstyle": "-|>"
      }
    }
  ],
  "panels": [
    {
      "panel_id": "dagua",
      "label": "Dagua",
      "bbox_xywh": [0, 0, 800, 900]
    },
    {
      "panel_id": "graphviz",
      "label": "Graphviz",
      "bbox_xywh": [800, 0, 800, 900]
    },
    {
      "panel_id": "mpl",
      "label": "Matplotlib",
      "bbox_xywh": [1600, 0, 800, 900]
    }
  ],
  "review_focus": [
    "arrowhead geometry",
    "tip placement",
    "shaft-to-head transition",
    "label-free clarity"
  ],
  "provenance": {
    "generated_at": "2026-03-23T12:00:00Z",
    "script": "scripts/build_gallery.py",
    "git_sha": "abc1234",
    "dagua_version": "0.0.0"
  }
}
```

Required fields:
- `schema_version`
- `id`
- `kind`
- `image.path`
- `fixture.id`
- `render_context.seed`
- `panels`
- `provenance`

Strongly recommended fields:
- `tags`
- `review_focus`
- `backends`
- `params`

### Gallery-wide Index

Also emit:

- `index.json`: nested manifest for browser tooling
- `index.jsonl`: one JSON object per image for LLM batch review

Use `index.jsonl` as the critic input source. It is easier to stream, filter,
and diff than a single large JSON blob.

---

## Sizing Recommendations

Use fixed sizes by artifact type.

### Atomic cards

Recommended size:
- `1600 x 1200 px`
- aspect ratio `4:3`
- `200 DPI`

Why:
- enough resolution for text, borders, and arrowheads
- consistent crop behavior
- small enough for batch generation

### Board images

Recommended cell size:
- each source card displayed at `400 x 300 px`

Recommended board width:
- `2000-2600 px`

Recommended max columns:
- `4`

Why:
- more than 4 columns makes labels and line detail too small
- fixed cells let LLM tooling use panel bounding boxes reliably

### Comparison triptychs

Recommended size:
- `2400 x 900 px`
- three equal `800 x 900 px` panels

Why:
- equal-width panels make side-by-side judgment easy
- one row avoids ambiguous scanning order

### Overlay diagnostics

Recommended size:
- match the atomic card size of the source case

---

## Comparison Panel Layout

Primary comparison layout:

```text
+----------------+----------------+----------------+
| Dagua          | Graphviz       | Matplotlib     |
| same fixture   | same fixture   | same fixture   |
| same canvas    | same canvas    | same canvas    |
+----------------+----------------+----------------+
| case id | fixture | key params | critic focus   |
+----------------------------------------------- --+
```

Rules:
- same graph fixture in every panel
- same canvas size in every panel
- same direction when the competitor supports it
- no decorative captions inside the graph area
- panel label in the top-left corner, outside or above the plotting region
- footer contains case id, feature family, fixture id, and key params

Secondary artifacts:
- raw single-backend renders saved separately
- overlay diff saved separately in `overlays/`

Do not combine all three concerns into one image.

---

## Coverage Structure

### Reference coverage

Organize by primitive family:

- `nodes/shapes`
- `nodes/fills`
- `nodes/borders`
- `nodes/text`
- `nodes/effects`
- `edges/routing`
- `edges/arrows`
- `edges/labels`
- `edges/styles`
- `clusters/containers`
- `clusters/labels`
- `clusters/depth`
- `graph/background`
- `graph/direction`
- `graph/margins`

Each family should have:
- atomic cards for each value
- one board image

### Combination coverage

Organize by interaction order:

- `combos/2way/`
- `combos/3way/`
- `combos/4way/`
- `combos/5way/`

Within each order, organize by feature-family slug:

```text
combos/2way/shape-fill/
combos/2way/shape-border/
combos/2way/arrow-label/
combos/3way/shape-fill-text/
combos/4way/shape-fill-border-text/
combos/5way/shape-fill-border-text-cluster/
```

Important:
- do not attempt the full cartesian product
- use a curated, fixed case set per combination family
- keep those case sets versioned and stable

That makes iteration meaningful. If the case set changes every run, visual
regression signals become noisy.

### Competitor coverage

Two comparison scopes:

- `comparisons/reference/`
- `comparisons/combos/`

For every important case:
- store raw Dagua render
- store raw Graphviz render
- store raw Matplotlib render
- store one triptych comparison
- optionally store one overlay diagnostic

---

## Fixture Strategy

The gallery becomes much easier to maintain if every render case references a
small fixture library instead of constructing ad hoc graphs.

Recommended fixture IDs:

- `single_node_label`
- `single_node_text_stress`
- `two_node_edge`
- `edge_arrow_single`
- `edge_label_single`
- `routing_crossing_quad`
- `nested_cluster_small`
- `cluster_label_deep`
- `combo_small_flow`
- `combo_dense_flow`

Why:
- stable fixtures make quality changes attributable
- critics can compare like-for-like across runs
- metadata stays compact because it points to a fixture ID

---

## LLM Review Requirements

Design the gallery so an LLM can judge one thing at a time.

Rules:
- one primary question per atomic card
- one primary comparison question per triptych
- stable panel order
- stable sizes
- visible case ID in metadata, not embedded into artwork
- no more than 12 panels per board
- no mixed comparison modes in one file

Best critic inputs:
- atomic card + sidecar JSON
- triptych + sidecar JSON
- `index.jsonl` row subset filtered by tags

Worst critic inputs:
- giant boards with 20+ panels
- overlays without raw comparison renders
- filenames that do not encode case identity

---

## Practical Dagua Mapping

Given the current dagua codebase, the cleanest mapping is:

- `docs/gallery/`
  Stable public gallery: boards, selected atomic cards, HTML index.
- `eval_output/gallery_audit/`
  Full atomic corpus, triptychs, overlays, JSONL manifests, critic outputs.

This matches the current separation between:
- showcase/reference output
- visual audit / iteration output

Do not force one directory tree to serve both release docs and active rendering
calibration.

---

## Final Recommendation

If only one decision is adopted, it should be this:

**Make atomic cards the canonical artifact, and generate boards and comparison
layouts from them.**

That single choice gives dagua:
- exhaustive coverage
- human-browsable summaries
- LLM-friendly review units
- stable metadata
- simple regression tracking

It is the best hybrid of:
- Matplotlib's categorized gallery
- Graphviz's example-to-source traceability
- D3's primitive-first organization
- Plotly's machine-readable reference model
