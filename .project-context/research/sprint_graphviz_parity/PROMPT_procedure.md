<task>
Build rock-solid visual iteration procedure for dagua's graphviz_strict cosmetic parity work. Current state: 95.74% in tolerance on declarative attributes (per scripts/parity_metrics.py), but the iteration loop lacks PIXEL-LEVEL visual verification — declared attributes can match while actual rendered output differs (font kerning, AA, sub-pixel rounding). The next iteration phase needs a procedure where Opus audit subagents can be MAXIMALLY PICKY without missing real visual gaps and without flagging metric-extraction artifacts as cosmetic problems.

Read for context (mandatory):
- `/home/jtaylor/projects/dagua/.project-context/knowledge/visual_tuning_workflow.md` (the postmortem driving the metric-driven approach)
- `/home/jtaylor/projects/dagua/scripts/parity_metrics.py` (existing declarative-attribute metric)
- `/home/jtaylor/projects/dagua/scripts/graphviz_theme_comparison.py` (the existing 3-way comparison harness)
- `/home/jtaylor/projects/dagua/AGENTS.md`
- `/home/jtaylor/projects/dagua/CLAUDE.md`

Repo: `/home/jtaylor/projects/dagua` (already on `develop` branch). Single working branch policy.

## What to build

### 1. `scripts/parity_pixel_diff.py` — pixel-level diff infrastructure

For each test panel from `_iter_cases()`:

a. Render the source graph through native graphviz `dot -Tpng -Gdpi=200` to produce the reference PNG at controlled DPI.
b. Render dagua with `graphviz_strict` theme at MATCHING canvas size and DPI (use `dagua.graphviz_utils.layout_with_graphviz` for positions if it provides them, or use `dot -Tjson` to extract positions and feed to dagua's render). The two outputs must be at the SAME pixel dimensions so pixel diffs are meaningful.
c. Compute per-pixel L1 RGB error AND SSIM (structural similarity index — use scikit-image `ssim` if available; otherwise reasonable fallback).
d. Produce a difference heatmap PNG showing where the renders disagree (red = high error, transparent = no error).
e. Compute per-region scalar errors:
   - whole-image L1 RGB error / pixel
   - SSIM (global)
   - text-region error (rough heuristic — bounding boxes of node ellipses; or use OCR-style text detection if simple)
   - background error (regions outside any node)
   - edge/arrow region error

Output to `eval_output/parity_pixel_diff/` with:
- `<slug>.png` = side-by-side: native dot | dagua strict | diff heatmap
- `<slug>.json` = per-panel scalar metrics
- `summary.json` = aggregate over all panels (mean/median/max L1, mean/median/min SSIM, panels by worst SSIM)
- `summary.md` = human-readable Markdown report

CLI:
```
python scripts/parity_pixel_diff.py [--cases <slug,slug,...>] [--out <path>] [--dpi 200]
```

### 2. Hi-res single-panel inspection rendering

Add to `parity_pixel_diff.py` (or as a sibling helper) a mode that renders BOTH dot and dagua for a chosen panel at HIGH DPI (e.g. 400 or 600) AS SEPARATE FILES, not concatenated, so a VLM auditor can look at each at full zoom without hitting the multi-image dimension cap. Output to `eval_output/parity_pixel_diff/hires/<slug>/{dot,dagua}.png`.

CLI:
```
python scripts/parity_pixel_diff.py --hires <slug,slug,...> --hires-dpi 400
```

These hi-res images are intended for VLM audit subagents to inspect at maximum detail. Each image must be ≤2000px on its longest side (the audit harness's image cap). For very dense graphs, scale DPI down so dimensions stay under 2000px.

### 3. Updated `parity_metrics.py` report emission

Extend the existing `parity_metrics.py` to emit a Markdown report alongside its JSON output. The Markdown should:
- Summarize global in-tolerance percentage
- List per-feature breakdowns (already in CLI output; reproduce in MD)
- List the top 10 worst-delta features with panel/element identifiers
- Include a "Locked features" section listing 100%-in-tolerance items
- Include a "Suggested next investigations" section flagging any feature where median delta is small but max delta is large (a tail signal)

Output path: `eval_output/parity_metrics_summary.md`.

### 4. Audit prompt template

Create `.project-context/knowledge/visual_audit_prompt_template.md` — a reusable template for Opus visual audit subagents that:
- Takes them through the inputs they have (declarative metric JSON, pixel-diff heatmap, hi-res per-panel images)
- Demands MAXIMUM pickiness: per-element measurements, no "looks similar" verdicts, must list at least N specific findings or justify why fewer
- Distinguishes between "real cosmetic gap" and "metric/measurement artifact" — auditor must classify each finding
- Distinguishes between "fixable via theme/render" and "rendering-stack residual" (font hinting, AA, B-spline geometry) — auditor must classify each
- Produces a structured output with: PASS/PARTIAL/FAIL verdicts on prior items, NEW findings ranked by severity, recommendations for next round of fixes
- Includes the no-cheating rule: if auditor finds < N items they must justify with explicit per-panel inspection

### 5. End-to-end procedure documentation

Update `.project-context/knowledge/visual_tuning_workflow.md` with a new "Operational procedure" section that describes the rock-solid loop:
1. Make changes (theme/render code)
2. Run `pytest tests/test_parity_metrics.py` — must pass (regression gate)
3. Run `python scripts/parity_metrics.py` — confirms declarative parity hasn't dropped
4. Run `python scripts/parity_pixel_diff.py` — confirms pixel parity (NEW)
5. Identify worst panels by L1/SSIM; render hi-res with `--hires <those>`
6. Dispatch Opus audit subagent using the audit-prompt-template, passing it: metric Markdown, pixel-diff summary Markdown, hi-res panel paths
7. Audit produces structured findings with classifications
8. Apply fixes; commit; loop
9. STOP when audit produces NO findings classified as "fixable cosmetic gap" — only "rendering-stack residual" or "metric artifact" remain

## Verification

After building infrastructure:
- Run `python scripts/parity_pixel_diff.py --cases pipeline,diamond,arrow_types,nested_clusters,tiny_graph` — should complete without errors and produce all expected outputs.
- Run `python scripts/parity_pixel_diff.py --hires pipeline` — should produce hi-res files.
- Run `pytest tests/test_parity_metrics.py` — should still pass.

ONE commit at end: `feat(parity): rock-solid visual iteration infrastructure (pixel diff, hi-res inspection, audit template)`.

Reply with: brief summary of what you built, the round-19 baseline pixel-diff stats (mean L1, mean SSIM, worst panels), and the commit SHA.

## Out of scope

- DO NOT change `dagua/styles.py` or any theme values.
- DO NOT change `dagua/render/` or any render code.
- DO NOT modify `tests/test_parity_metrics.py` thresholds.
- This is pure infrastructure work — the next phase will use it to drive theme/render changes.
</task>

<completeness_contract>
Not done until:
1. `scripts/parity_pixel_diff.py` works on all 45 panels with `--cases` and `--hires` modes.
2. `scripts/parity_metrics.py` now emits Markdown alongside JSON.
3. Audit prompt template created.
4. Procedure section added to `visual_tuning_workflow.md`.
5. `pytest tests/test_parity_metrics.py` still passes.
6. ONE commit with descriptive message.
</completeness_contract>

<verification_loop>
After building each script, run it on a small subset (3-5 panels) to confirm output. Don't move to next script until current works.
</verification_loop>

<missing_context_gating>
Default to most reasonable interpretation. If the dot-Tpng + dagua-Tpng dimension matching turns out to be tricky (figsize/dpi math), use a known-good approach: render dot at e.g. 200 DPI to determined dimensions, then call dagua's render with `figsize=(target_w/dpi, target_h/dpi), dpi=dpi`. The existing graphviz_theme_comparison.py shows how to invoke dagua's render with controlled figsize.
</missing_context_gating>

<action_safety>
Pure infrastructure. No theme/render code touched. develop branch. ONE commit at end.
</action_safety>
