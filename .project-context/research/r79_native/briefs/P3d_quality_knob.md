<task>
Worktree: /home/jtaylor/.claude/worktrees/dagua-native-p4. FIRST:
`git checkout r79/native && git pull --ff-only 2>/dev/null; git checkout -b r79/p3d-quality`
(the worktree is currently on the merged r79/p4-scale; branch from current r79/native which
now contains native_stress, native_stress_ml, hybrid_v2, and the layered fixes).
Python: .venv/bin/python.

GOAL: implement dagua's public TIME-VS-QUALITY knob, approved by JMT (design in
/home/jtaylor/.claude/research/dagua/r79-native/r79_DESIGN.md section 3 "Quality knob"):

API (dagua/config.py LayoutConfig):
- `quality: float | str = "balanced"` -- accepts float in [0,1] or names:
  draft=0.25, balanced=0.5 (default), high=0.75, max=1.0. Validate at construction
  (clear error for out-of-range/unknown name). Store normalized float.
- `time_budget_s: float | None = None` -- optional hard wall-clock cap, enforced via the
  existing stall/early-break machinery in the gradient core (add a wall-clock check to the
  EarlyBreak/StallCount op family as a small config extension; when exceeded, finish the
  current step, run the cheap final polish (overlap projection + aspect fit), return).

MAPPING (implement in dagua/layout/resolve.py where auto-budgets are computed; a single
pure function `resolve_quality_budgets(quality: float, ...) -> QualityBudgets` frozen
dataclass so it is testable and documented):
- steps multiplier: log-linear 0.4x (q=0) -> 1.0x (q=0.5) -> 2x (q=0.75) -> 4x (q=1.0)
- multi_start_k: 1 for q<0.7; 3 for 0.7<=q<0.9; 5 for q>=0.9
- stress pivot count: 32/64/128/256 at draft/balanced/high/max (interpolate, cap by N)
- SMACOF polish iters (stress core, N<=5K): 0/8/24/50
- polish battery (layered core best-of): off for q<0.35; class-gated subset default;
  full battery for q>=0.75
- multilevel refinement rounds per level (native_stress_ml): scale 0.5x/1x/2x/3x
- BH theta / sampling rates where applicable: looser at low q, tighter at high q
EVERY existing explicit config field OVERRIDES the knob (knob only fills values the user
did not set -- respect the existing "explicitly set" detection if config has one, else
compare against dataclass defaults and document that limitation).

WIRING: thread QualityBudgets through prepare_pipeline_config / the native pipelines
(layered, native_stress, native_stress_ml) at the points where those budgets currently come
from constants/auto-scaling. Keep diffs surgical; decomposable-ops philosophy holds.

TESTS: mapping function unit tests (monotonicity: higher q never reduces any budget;
name/float equivalence; override precedence -- explicit steps beats knob); integration
smoke: same graph at quality=0.1 vs 0.9 -> the 0.9 run does >= steps and >= wall time and
composite(0.9) >= composite(0.1) - 0.5 on 3 seeded small graphs; time_budget_s: a run with
time_budget_s=2 on a 2000-node graph returns in < 6s wall with finite positions.

GATE: scripts/r79_baseline.py --dagua-only sweep (timeout 9000) at DEFAULT quality must be
IDENTICAL W/T/L to the pre-change sweep (default quality=0.5 must reproduce current
default budgets exactly -- calibrate the mapping so balanced == today's behavior; this is
the critical constraint). Restore store churn before committing.

DOCS: docstrings + one section appended to the config docstring table; do not rebuild
generated docs (separate maintenance task).

CONSTRAINTS: ASCII; conventional commits; no AI attribution; COMMITS REQUIRED on gate pass
(repo AGENTS.md orchestrator-git notes do NOT apply -- this brief authorizes commits).
Evidence to .project-context/research/r79_native/P3D_EVIDENCE.md.
</task>

<operational_rules>
1. Any assistant message WITHOUT a tool call TERMINATES your session; final no-tool-call
   message = report, only after commits verified. 2. stdin closed. 3. Long runs in ONE exec
   call with generous timeout; first corpus import takes minutes. 4. ENOSPC -> stop, report.
</operational_rules>

<default_follow_through_policy>
Most reasonable low-risk interpretation; keep going; note choices. The
balanced-equals-today calibration and override precedence are non-negotiable.
</default_follow_through_policy>

<completeness_contract>
Done = quality + time_budget_s implemented and wired to all three native cores; mapping
unit tests + integration smokes green; default-quality sweep IDENTICAL W/T/L; committed on
r79/p3d-quality with P3D_EVIDENCE.md.
</completeness_contract>

<verification_loop>
1) Mapping monotonicity tests. 2) Override precedence test. 3) quality 0.1-vs-0.9 smoke.
4) time_budget_s smoke. 5) Default-quality sweep W/T/L identical. 6) ruff/pytest green;
git clean.
</verification_loop>

FINAL REPORT: mapping table as implemented; sweep confirmation; wall-time samples at the 4
named quality levels on 2 representative graphs; commit shas.
