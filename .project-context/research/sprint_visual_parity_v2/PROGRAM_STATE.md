# Full Cosmetic Parity Push -- Program State (Codex-safe copy, outside repo)

**GOTCHA (2026-07-16):** Codex workers `git clean`/`stash` at start and WIPE UNTRACKED files.
Untracked PROGRAM_STATE.md + S8_competitor_plan.md were deleted from the repo mid-run.
=> COMMIT notes immediately (tracked files survive); keep authoritative copies here in
`~/.claude/research/dagua/visual_parity_v2/` (outside the repo, Codex can't touch). Add
"do NOT git clean/stash/reset --hard; only edit your target files" to Codex briefs.

**JMT directive:** faithfully implement EVERY cosmetic feature/dial from graphviz AND
mermaid/cytoscape/d3; compose sensibly under pathological combos; push to the hard ceiling.

## Method (verified)
Drive off per-panel DECLARATIVE out_of_tolerance_features + PIXEL measurement (ink bbox,
stroke-run). All 3 VLMs (Opus/Sol/Fable) OVER-READ residuals -- pixel-verify every VLM
finding. Fable = final ceiling gate + numeric threshold. Parallel Codex where files don't
overlap; SERIAL for the render chokepoint (dagua/render/mpl.py). Verify each: declarative +
pixel + tripwires + ruff, then COMMIT.

## Status
- [DONE] S1 node scale (fa0fe4ed), S2 font fc-match (880c0b7c), S3 arrow size (82c83489),
  arrow fill (a01f3f0c), empty-label metric fix (0f61f2a0). Declarative 88.21 -> 99.75%.
- [DONE] Shapes wave 1: 11 graphviz shapes (1401e681), shape_path_iou 47/47=100%, overall 99.86%. dagua supports 26 shapes.
- [IN FLIGHT] Shapes wave 2: 12 specialized/bio shapes (promoter cds terminator ribosite proteasesite rpromoter rarrow larrow assembly insulator signature invtrapezium).
- [NEXT] S4 fills -> S5 splines/loops -> S6 clusters -> S7 compose/pathological -> S8 competitors -> Fable ceilings.

## S8 competitor gaps (from the research scan -- reconstruct full tables if needed at S8 time)
Prioritized cross-package cosmetic gaps for dagua to implement (mostly GAP-small field additions):
1. Mid-edge / source arrowheads (Cytoscape mid-source/mid-target-arrow) -- EdgeStyle fields.
2. Arbitrary edge dash-pattern array (Cytoscape) -- EdgeStyle line_dash_pattern tuple.
3. Edge label text halo/outline (Cytoscape) -- EdgeStyle text_outline.
4. External label 9-position alignment (Cyto+GV) -- extend NodeStyle.external_label from 4-way.
5. Per-element opacity: fill_opacity + text_opacity (Cytoscape) -- NodeStyle fields.
6. Node outline (Cytoscape outline-*: color/width/offset/style) distinct from double-border.
7. Custom polygon nodes via point list (Cytoscape) -- GAP-large.
8. Cluster/compound padding per-side (Cytoscape) -- ClusterStyle padding_{t,r,b,l}.
9. Per-node Brewer colorscheme index (GV colorscheme=oranges9 color=7).
10. Rounded polygon variants (round_triangle/diamond/pentagon/hexagon/octagon) (Cytoscape).
11. Cross/X arrowhead (Mermaid --x); wavy edge line (Mermaid ~~>).
12. Label text-shadow (Cytoscape); edge label autorotate (Cytoscape).
13. Graphviz l/r half-clip arrowhead modifiers + stacked compounds (GV -- also arrowhead atlas).
14. Cytoscape-specific shapes: barrel, cut_rectangle, bottom_round_rectangle.
15. GAP-large / backend-limited (SVG/HTML only): HTML-table labels, CSS dash animation, ghost/underlay.
(OUT OF SCOPE: layout-algorithm choices; flagged inline in the research.)

## Resume / case routing
git log --oneline -10; parity_metrics --quick --profile v2 | grep in-tolerance; check /tmp/*_fix.log (CODEX_*_DONE). Done+committed -> verify+dispatch next. Running -> yield. Quota -> Claude Agent fallback. 3 rounds no improvement -> waive+note.
Env: /opt/homebrew/Caskroom/miniforge/base/envs/dagua/bin on PATH. Repo copy of this file lives at .project-context/research/sprint_visual_parity_v2/PROGRAM_STATE.md (commit it).
