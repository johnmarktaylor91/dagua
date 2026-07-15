# Visual Parity v2 Runbook

This loop drives dagua toward Graphviz 7.0.5 visual parity without trusting
visual judgment alone. Track G optimizes geometry-injected Graphviz comparisons
(`svg_declared`, with `png_raster` reported separately); Track D closes manifest
coverage cards. Done means the Track G and Track D stop criteria in section 7
pass, including dual-lab ceiling audits with zero unwaived HIGH/MED fixable
real cosmetic gaps.

## 1. Preflight

Run from repo root `/home/jtaylor/.claude/worktrees/dagua-visual-parity-v2`.
Abort on any failure.

- Worktree and SHA:
  `pwd && git status --short --branch && git rev-parse HEAD`
- Graphviz pin:
  `dot -V`
  Expected: `dot - graphviz version 7.0.5 (20221231.0122)`. Mismatch stops the
  loop and goes to JMT before any baseline or dial work.
- Python raster deps:
  `python - <<'PY'
import cairosvg, matplotlib, PIL
print("cairosvg", getattr(cairosvg, "__version__", "unknown"))
print("matplotlib", matplotlib.__version__)
print("pillow", PIL.__version__)
PY`
- Competitor versions:
  `python -m scripts.competitor_renderers.capabilities --print-versions --out refcache/versions.json`
  If versions differ from the ledger environment block, record a rebase-label
  warning before comparing numbers.
- Codex image-input probe:
  `codex exec --help | rg -- ' -i|--image'`
  If absent, record in `STATE.md` that rival-lab ceiling audits use a second
  independent Claude subagent instead of `codex exec -i`.
- Kill stale watchers:
  `pkill -f 'codex-watch\.sh|bg-watch\.sh' || true`
- Pause sentinels:
  `find ~/.claude/state -maxdepth 1 -name 'paused-*.sentinel' -print`
  If any file prints, ask before continuing.
- Disk and memory:
  `df -h . && free -h`
  Keep render output lean; stop if less than 10% free.
- Tripwires:
  `python -m scripts.visual_parity.tripwires --all`
- Store parse checks:
  `python -m scripts.visual_parity.io --self-test`
  `python - <<'PY'
from pathlib import Path
from scripts.visual_parity.io import read_card_manifest, read_coverage_matrix, read_ledger
root = Path(".project-context/research/sprint_visual_parity_v2")
read_card_manifest(root / "card_manifest.json")
read_coverage_matrix(root / "coverage_matrix.json")
read_ledger(root / "ledger.json")
print("stores ok")
PY`

## 2. Model Selection

Run at loop start and weekly.

1. Read `~/.claude/knowledge/max_power_models.md`; if it is more than 48 hours
   stale, note that in `STATE.md` and continue.
2. Web-search with latest-state discipline:
   `best vision language model fine detail comparison <month year>`
   `Anthropic Claude vision model <year> release`
   Open at least one primary source if results contradict the roster.
3. Render the calibration package:
   `python -m scripts.visual_parity.render_calibration_probe --out eval_output/visual_parity_v2/model_probe`
4. Dispatch one Agent subagent per candidate model with
   `.project-context/research/sprint_visual_parity_v2/prompts/audit_v2.md`,
   all probe images, and instructions to return structured JSON. Cap each call
   at 10 images and 2000 px per image.
5. Score responses:
   `python -m scripts.visual_parity.select_vision_model --score eval_output/visual_parity_v2/model_probe/candidate_responses.json --probe-manifest eval_output/visual_parity_v2/model_probe/defect_manifest.json --out .project-context/research/sprint_visual_parity_v2/model_selection.json`
   Reject any model that misses the injected stem or truncation defect, or
   flags either false-positive control.
6. Record `{primary, ceiling, fallback, scores, evidence_urls, probed_at}` in
   `STATE.md` and the ledger auditor block.

## 3. Baseline And Resume

Fresh start: `state.json` is missing or has no launched round beyond `g000`.
Resume: `python - <<'PY'
import json
print(json.load(open("state.json", encoding="utf-8")))
PY`
If `round` is greater than `g000`, skip to section 6 case routing.

Lane E1 already ran the PRELAUNCH baseline: tripwires, g000/d000, dashboard,
ledger seed. It leaves `sub_sprint="S0"` and
`s0_round0_vlm_census="required"`. On first launch, verify that state, then
the round-0 VLM census is the first action. S0 is complete only when that
census audit is filed; no S1 dial round may start before it.

Track G baseline commands:

`python scripts/parity_pixel_diff.py --quick --reference svg-cairo --bit-equivalent --inject-splines --dpi 120 --out eval_output/visual_parity_v2/trackG/round_g000`

`python scripts/parity_metrics.py --quick --profile v2 --out eval_output/visual_parity_v2/trackG/round_g000/parity_metrics.json`

png_raster lane report:

`python scripts/parity_pixel_diff.py --quick --reference dot-png --bit-equivalent --inject-splines --dpi 120 --out eval_output/visual_parity_v2/png_raster/round_g000`

Track D baseline:

`python scripts/generate_calibration_suite.py --two-panel --manifest .project-context/research/sprint_visual_parity_v2/card_manifest.json --output-dir eval_output/visual_parity_v2/trackD/round_d000 --case-id arrowhead_normal --case-id linestyle_dashed`

Round-0 VLM census package, first spend of the launched loop:

`python -m scripts.visual_parity.audit_package --round-id g000 --geometry-mode injected --prompt .project-context/research/sprint_visual_parity_v2/prompts/audit_v2.md --round-image eval_output/visual_parity_v2/trackG/round_g000/svg_cairo/pairs/tiny_graph.png --metric-summary eval_output/visual_parity_v2/trackG/round_g000/parity_metrics.json --out .project-context/research/sprint_visual_parity_v2/audits/audit_g000_package`

Then dispatch primary auditor subagents. File results under
`.project-context/research/sprint_visual_parity_v2/audits/audit_g000_<model>.md`
and flip `s0_round0_vlm_census` from `required` to that path.

Ingestion and dashboard:

`python -m scripts.visual_parity.ledger --init`

`python -m scripts.visual_parity.ledger --seed-baseline`

`python -m scripts.visual_parity.ledger --generate-lock-tests`

`python -m scripts.visual_parity.dashboard --out eval_output/visual_parity_v2/dashboard/index.md`

## 4. Round Procedure

1. Read state:
   `python - <<'PY'
import json
print(json.dumps(json.load(open("state.json", encoding="utf-8")), indent=2))
PY`
2. Run tripwires if due:
   `python -m scripts.visual_parity.tripwires --all`
3. Select the worst active sub-sprint gate, P0 first:
   `python - <<'PY'
import json
data=json.load(open(".project-context/research/sprint_visual_parity_v2/ledger.json", encoding="utf-8"))
rows=[r for r in data["rows"] if r.get("priority") in {"P0","P1"} and r.get("parity_status") not in {"in_tolerance","matched","waived_improvement","waived_out_of_scope"}]
for r in rows[:20]:
    print(r["row_id"], r.get("priority"), r.get("target_kind"), r.get("parity_status"))
PY`
4. Sweep, do not step:
   `python -m scripts.visual_parity.sweep --dial <field> --values <a,b,c,d,e> --cases <case_ids>`
   For coupled dials only, add `--grid-dial <field2> --grid-values <a,b,c>`.
5. Dispatch one Codex worker using section 5. Worker edits one dial family,
   runs fast checks, commits, and stops.
6. Orchestrator verifies:
   `python -m scripts.visual_parity.tripwires --all`
   `pytest tests/test_visual_parity_* tests/test_parity_* -q -m "not slow"`
   `python -m scripts.visual_parity.ledger --generate-lock-tests`
7. Freeze passing rows by updating `ledger.json`, then regenerate locks:
   `python -m scripts.visual_parity.ledger --generate-lock-tests`
8. Commit format:
   `feat(parity-v2): round g012 -- <dial> swept to <value>, <metric> <before>-><after>`
9. Regenerate and mirror:
   `python -m scripts.visual_parity.dashboard --out eval_output/visual_parity_v2/dashboard/index.md`
   `rsync -a .project-context/research/sprint_visual_parity_v2/ ~/.claude/research/dagua/visual_parity_v2/`

Sub-sprints, in order:

| id | scope | primary dials | done when | ratchet |
| --- | --- | --- | --- | --- |
| S0 | infra, tripwires, baselines, round-0 VLM census | none | tripwires pass, baseline ledger written, round-0 audit filed | 85 |
| S1 | scale/autosize unlock | autosize, padding, min sizes, density retirement | autosize w/h >= 99% in tolerance | 94 |
| S2 | typography | font metrics, kerning, baseline | font/label >= 99.5% | 94 |
| S3 | arrowhead atlas | primitives, aliases, modifiers, compounds | gv-mappable arrows pass or waived | 96 |
| S4 | fills | gradient, striped, pie, hatch | fill cards in tolerance or waived | 96 |
| S5 | splines and trim | injection, endpoints, loops, back-edges | spline_path_dist <= 1.0 pt mean and corridor gates pass | 98 |
| S6 | clusters | nested rects, labels, fills, depth | cluster features >= 99% | 99 |
| S7 | combos and pathological | legacy evil cards | zero HIGH VLM findings | 99 |
| S8 | competitor tier B | mermaid/cytoscape/d3 cells | every cell non-untested | 99 |
| S9 | user reference guide | aesthetics loop | guide audit PASS | 99.5 |

VLM cadence: round 0 full census before dials; every fifth round audit three
best panels plus a corrupted canary; every sub-sprint close audit worst 8 plus
ROIs; ceiling requires primary and rival-lab audits. Build packages with
`python -m scripts.visual_parity.audit_package --help` as the argument guide.

## 5. Dispatch Templates

Codex worker brief:

```
Implement one visual parity v2 round in /home/jtaylor/.claude/worktrees/dagua-visual-parity-v2.
Active sub-sprint: <S#>. Dial family: <dial>. Sweep winner: <value>. Cases: <ids>.
Do not delegate to reference docs or change unrelated dials. Apply the value,
run: python -m scripts.visual_parity.tripwires --all and targeted pytest/ruff
for touched files. Commit early with conventional commit text and stop. Do not push.
```

Codex background pattern:

`~/.claude/scripts/codex-bg.sh --cd /home/jtaylor/.claude/worktrees/dagua-visual-parity-v2 --log .research/round_g012_codex.log -- '<brief file or quoted brief>'`

`~/.claude/scripts/codex-watch.sh .research/round_g012_codex.log`

If the harness lacks Monitor/ScheduleWakeup, record the PID in `state.json` and
use `~/.claude/scripts/bg-watch.sh <pid> .research/round_g012_codex.log`.

Non-Codex render/sweep jobs:

`bash -lc '<command>' > .research/round_g012_render.log 2>&1 & echo $!`

`~/.claude/scripts/bg-watch.sh <pid> .research/round_g012_render.log`

Audit subagent: use `prompts/audit_v2.md` for parity or
`prompts/aesthetics_v1.md` for S9, <=10 images, <=2000 px, reports in
`.project-context/research/sprint_visual_parity_v2/audits/`.

## 6. Wake-Up Routing

Every wake starts with:

`python - <<'PY'
import json
print(json.dumps(json.load(open("state.json", encoding="utf-8")), indent=2))
PY`

Then inspect:
`git log -3 --oneline`
`kill -0 <pid> && echo running || echo done`
`tail -80 <log>`

- A: in-flight done and committed. Verify gates on the SHA, update ledger,
  select the next target, and dispatch the next round in the same response.
- B: in-flight still running. Acknowledge, re-arm watcher if missing, yield.
- C: quota or pause sentinel. Use Codex -> Claude subagent -> ScheduleWakeup at
  reset+5m. Codex-only phases use the state file and `bg-watch.sh` fallback.
- D: gates regressed or tests red. Revert the round commit, mark attempt
  failed, increment anti-flail counter.
- E: tripwire failure. HALT dial work for global failures; dispatch a metric
  fix. Scoped failures mark dependent metrics untrusted and continue elsewhere.
- F: stop criteria met or `max_rounds` hit. Run shutdown.

Floor rule: never end a turn with nothing in flight and no wakeup armed. In a
Codex-driven turn, documented PID/state-file polling via `bg-watch.sh` satisfies
the floor.

## 7. Stop Criteria And Ratchets

Track G is done when all are true: declarative global in-tolerance is >= 99.5%
for `svg_declared`; every feature is >= 98% excluding named waivers; corridor
gates pass on all edges; dual-lab ceiling audit reports zero HIGH and zero MED
real_cosmetic_gap x fixable_theme_or_render findings unless a MED has a
documented waiver object; `png_raster` lane report exists with discrepancies
documented but not gated; all tripwires pass; no stalled P0/P1 row remains
without an evidence waiver or filed metric-fix residual.

Track D is done when every coverage cell is non-untested, blocked_upstream is
reported separately, 100% of P0 cells are supported+in_tolerance/matched or
evidence-waived, >=98% of P1 cells are in_tolerance or evidence-waived, and
card gate metrics are within per-card tolerances.

Rebase rule: whenever reference kind, rasterizer, or tolerance profile changes,
all affected metrics get a rebase label. Never compare old-lane numbers to
new-lane numbers without the label.

Anti-flail: 3 rounds on one knob without gate improvement marks the knob
`stalled`; 5 attempts on one issue requires waive, split, or escalate.

Ratchet schedule: 85.0 prelaunch/S0, 94 after S1, 96 through S4, 98 after S5,
99+ by S6, 99.5 final. S0 exit gate is the round-0 VLM census filed and
`s0_round0_vlm_census` changed from `required` to the audit path. The prelaunch
baseline alone never satisfies S0.

## 8. Records

Every round updates `ledger.json`, generated lock tests, dashboard, state,
and a round log. Use:

`python -m scripts.visual_parity.ledger --generate-lock-tests`

`git diff --exit-code tests/test_visual_parity_locks.py`

`python -m scripts.visual_parity.dashboard --out eval_output/visual_parity_v2/dashboard/index.md`

`rsync -a .project-context/research/sprint_visual_parity_v2/ ~/.claude/research/dagua/visual_parity_v2/`

Round log format:
`.project-context/research/sprint_visual_parity_v2/round_logs/round_g012.md`
with target, sweep table, chosen value, before/after metrics, tests, tripwire
status, commit SHA, and audit links.

## 9. Shutdown

1. Run final primary and rival-lab ceiling audits.
2. Regenerate dashboard, reference guide, generated tests, tripwires, and
   `png_raster` report.
3. Write `SUMMARY.md` and mirror
   `.project-context/research/sprint_visual_parity_v2/` to
   `~/.claude/research/dagua/visual_parity_v2/`.
4. Orchestrator sends JMT with
   `~/.claude/scripts/send-to-jmt.sh`: five representative before/after pairs
   plus dashboard headline with lane and `geometry_mode` labels. Agents never
   send directly.
5. Mark state DONE and leave the worktree for merge review.

## 10. Trap Appendix

| trap | countermeasure |
| --- | --- |
| T1 metric lies | Tripwires, canary audits, and best-panel checks; see sections 1, 4, 6. |
| T2 VLM magnitudes | VLMs provide detection and direction only; sweeps choose values. |
| T3 oscillation | One dial family per round, sweeps, ordered sub-sprints. |
| T4 soft STOPs | Dual-lab max-strictness ceiling audits before acceptance. |
| T5 guardrail floors | S1 unlock, row locks, no size-dependent freeze before S1. |
| T6 reference drift | Graphviz pin, refcache keys, versions file, rebase labels. |

## 11. Failure Modes

- 3 no-improve rounds: mark stalled, record residual hypothesis, move on.
- 5 attempts: choose waive, split into smaller dial, or escalate to JMT.
- Tripwire failure: global dependency HALTs dial work; scoped failure marks
  dependent metrics `metric_untrusted`.
- Quota: fallback chain is Codex, Claude subagent, then ScheduleWakeup at
  reset+5m; Codex-only work uses PID/state-file polling.
- Watchers: poll exact PIDs with `kill -0`; never use bare `pgrep -f`.
- Audit passes corrupted canary: reject that audit, re-probe the model, and do
  not use its verdict for gates.
