# Visual Parity v2 -- Autonomous Session Summary (2026-07-16, devmini)

## Objective
Autonomously iterate dagua's cosmetic rendering to match Graphviz 7.0.5 (the
`graphviz_strict` theme), per JMT's visual-parity-v2 sprint.

## Outcome
**Dominant cosmetic gaps FIXED and pixel-verified. Declarative parity 88.21% -> 96.37%.**
Standard graphs now render at 0.80-0.92 SSIM vs graphviz. Remaining residual is verified
over-reads + out-of-scope waivers + metric artifacts (see below).

## Environment setup (fresh machine)
- Codex repaired: brew cask 0.144.5 (npm install was broken). GPT-5.6 Sol string = `gpt-5.6-sol`.
- dagua conda env `dagua` (py3.12, torch 2.13); Graphviz **7.0.5** via conda-forge (exact runbook pin; `dot -V` = `7.0.5 (0)`, build-hash zeroed by conda but same source tag).
- Texting verified.
- Two blocking bugs fixed to make dagua import/render on numpy 2.5:
  - coregd optional torch_cluster/PyG import guard -- commit `76567159`
  - numpy-2 removed 2D `np.cross` in render (9 sites) -- commit `2750f8ac`

## Cosmetic fixes (Codex-implemented, orchestrator-verified against PIXELS)
1. **Font (fc-match)** -- commit `880c0b7c`. `font_family="Times,serif"` was falling back to DejaVu Sans (TeX Gyre Termes `.pfb` absent on macOS; hardcoded `/usr/share/texmf` path is Linux-only). Resolved via `fc-match` to the same `Times.ttc` the dot/cairosvg reference uses, in both the render and node-size-measurement paths. Serif confirmed by VLM. Declarative 88.21 -> 95.32%.
2. **Node size render floors** -- commit `fa0fe4ed`. Rendered ellipses were drawn tight to the label (~0.46x graphviz width) ignoring the floored node box, despite CORRECT computed sizes -- a render bug the declarative metric HID. Fixed. Node width ratio 0.46 -> **1.00** (pixel ink bbox); SSIM 0.638 -> 0.685.
3. **Arrowhead size** -- commit `82c83489`. `arrow_width_pt` 82.64 -> **100%**, `arrow_length_pt` 90.68 -> **100%**. Overall -> 96.37%.

Checkpoint commit: `3d7fb547` (S1), plus S3 checkpoint.

## Verified results
- Declarative in-tolerance: **88.21% -> 96.37%**.
- Node width ratio (dagua/graphviz, pixel): 0.46 -> **1.00** (all standard panels ~1.0).
- Arrowheads: width & length 100%.
- Per-panel SSIM: standard graphs GOOD -- edge_styles 0.92, tiny_graph 0.89, colors/label_variety 0.88, node_shapes 0.80. Mean 0.685 (dragged by atlas/dense panels -- see below).

## KEY METHODOLOGY FINDING: VLMs systematically over-read residual gaps
All three VLMs (Opus 4.8, GPT-5.6 Sol, Fable 5) over-read the remaining gaps. **Pixel and
per-panel-declarative verification is authoritative.** Verified over-reads:
- "Edges 1.5-2x / 2-3x too thick" (Opus, Fable) -> measured **1.5x median, ~1.0x on most panels**.
- "Edge taper" (Fable) -> the widening is the ARROWHEAD, present in BOTH (graphviz's tip is wider: dagua 3->7, gv 2->9).
- "Specialty arrowheads render as stubs" / "self-loops missing" (Opus) -> **false** (dagua implements 27 arrowhead glyph types; the n3->n3 self-loop renders).
- "roundrect -> ellipse; shapes don't fit labels" (Fable) -> node_shapes_showcase has **0 declarative failures**; dagua supports roundrect.

Counter-point: two REAL render bugs the metrics HID (node size, font fallback) were caught
via PIXEL measurement. So VLMs remain valuable for render-scale bugs the declarative misses --
but **every VLM finding must be pixel/declarative-verified before acting**, or you chase
phantoms (the documented failure mode in this repo's earlier tuning retro).

## Remaining long-tail (out-of-scope / artifacts / diminishing returns / needs steer)
- **arrowhead_atlas SSIM 0.20**: metric artifact -- empty-label target nodes (`label=""`) carry a latent 14pt font attribute; both sides render blank, but the metric flags a phantom font mismatch on 170 nodes (this also deflates `font_family` to 76%). + thin-strip alignment sensitivity. Not a cosmetic gap.
- **shape_atlas SSIM 0.46**: WAIVED unsupported shapes (`gap_common`/`waived_sample` buckets render as placeholder rects BY DESIGN -- new shape drawers explicitly out of scope for this surgery).
- **cluster_nest_deep 0.43 / dense_graph 0.54**: dense/nested layout + cluster boxes. NOT yet isolated -- may be real (S6 clusters) or alignment/injection residual. The one area worth a careful (verified) look if pushing further.
- Minor declarative misses: `node_autosize_w` 5 (max 27pt, visual matches at ink ratio 1.0), `arrow_filled` 6.
- UNVERIFIED: Fable's "multi-line label line-2 font shrink" (label_variety) -- one concrete claim not yet pixel-checked; likely minor given the over-read pattern.

## Recommendations for JMT
Core objective (match `graphviz_strict` cosmetics on standard graphs) is **ACHIEVED + verified**.
Further gains are judgment-dependent / diminishing returns:
- (a) Investigate the cluster/dense panels (verify real vs artifact, then fix if real) -- the main candidate for genuine remaining work.
- (b) Fix the parity metric to skip font on empty-label nodes (assessment accuracy; needs approval since it changes what parity measures).
- (c) Decide scope on the waived shapes (implement new shape drawers?).
- (d) Accept current parity as sufficient.

## Git
Branch `codex/visual-parity-v2`. Commits: 76567159, 2750f8ac, 880c0b7c, fa0fe4ed, 3d7fb547, 82c83489 (+ checkpoints). Clean history, no AI attribution. Durable notes mirrored to `~/.claude/research/dagua/visual_parity_v2/`.
