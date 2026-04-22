# Sprint 0 -- Pipeline Decompose + Default Flip + MVP Iteration

REVISED per 2026-04-22 round-2 adversarial review: original goal section
conflicted with the Sprint 0 / Sprint 0.5 split. Sprint 0 is now ONLY
pipeline decomposition, default flip, MVP iteration harness, rubric sync,
and memory-parity baseline. All held-out / salt / benchmark-authority /
graph_generator work moves to Sprint 0.5 (defined in 02_sprint_map.md).

## Goal

At the end of Sprint 0:
1. `dagua.layout(g)` (no algorithm kwarg) routes to the ops pipeline. Legacy
   `_layout_inner` is either deleted or moved to `dagua/layout/_archive/legacy_engine/`.
2. The `dagua_native` pipeline file imports ONLY from `dagua.layout.ops.*`.
   Zero imports from `dagua.layout.engine`. Zero inline config helpers.
3. A scalar scorer + image emitter for one graph runs in <=8 seconds end-to-end.
4. An iteration-suite baseline metrics JSON committed at the Sprint-0 SHA
   (under the EXISTING seed=42 convention; opaque held-out and competitor
   matrix land in Sprint 0.5).
5. `scripts/iterate_native.sh <graph_id>` MVP (chain-only) is live and prints
   a scalar + path to an image.
6. Rubric-code sync (Task 0.8) + composite-profile split (Task 0.8.1).
7. Memory-parity baseline (Task 0.10).

## Entry criteria

- This plan approved.
- `feat/native-algo-sprint-0` branch created on top of current main.
- No in-flight overnight benchmark run (current salvage-finish done or cancelled).

## Exit criteria (binary, revised round 2)

1. `python -c "import dagua; dagua.layout(g)"` on a 100-node graph returns a
   position tensor via the ops pipeline. Verified by a one-line trace that
   `build_dagua_pipeline` was called.
2. `grep -r "from dagua.layout.engine import" dagua/layout/ops/pipelines/dagua_native.py`
   returns empty.
3. `scripts/iterate_native.sh chain_100 40 --no-image` completes in <=8s
   and prints a scalar quality score (image path NOT required for the
   score-only iteration). Default `scripts/iterate_native.sh chain_100`
   (with rendering) completes in ~10-12s on the dev machine -- slower
   because matplotlib startup adds 3-5s. Both paths print an image path;
   the image is rendered only on the default (image-on) path.
4. `eval_output/native_algo/baseline_sprint_0/metrics.json` exists with metric
   values for the iteration suite (seed=42) at the Sprint-0 commit SHA.
   NOTE: held-out baseline comes in Sprint 0.5, not here.
5. Rubric-code sync: 04_evaluation_rubric.md matches the `composite()`
   function in `dagua/metrics.py` exactly, field-by-field. (Task 0.8)
6. Composite-profile split: `composite()` either ships a `composite_large()`
   variant for N>2000 quick mode, or fails loudly when required inputs are
   missing. No silent defaults. (Task 0.8.1)
7. Memory-parity baseline: measured RSS for current ops pipeline vs legacy
   `_layout_inner` on 1K and 10K node graphs, written to
   `eval_output/native_algo/baseline_sprint_0/memory_profile.json`. Sprint 1
   entry gate requires closing any gap >20%. (Task 0.10)
8. Sprint 0 adversarial Codex review PASS -- no CRITICAL/HIGH unaddressed.

DEFERRED TO SPRINT 0.5 (not required for Sprint 0 exit):
- Held-out suite generation, opacity manifest, salt infrastructure.
- `dagua/eval/benchmark.py` seed-strategy / sprint-tag / salt-path flags.
- `dagua/eval/graph_generator.py`.
- Held-out + rolling baseline metrics.
- Authoritative competitor matrix freeze + first refresh.
- `scripts/pick_weak_graphs.py`, `scripts/refresh_competitors.sh`.
- Per-family Pareto ladder calibration.

See 02_sprint_map.md Sprint 0.5 for the full list.

## What gets built

### Task 0.1 -- Decompose config prologue

Move the following helpers into registered ops or pure config-resolver utilities
that live in `dagua.layout.ops.pipeline_resolve.py` (new):
- `_normalize_node_sizes`
- `_final_projection_iterations`
- `_stall_config`
- `_build_flex_constraints`
- `_prepare_pipeline_config`
- `_build_loss_ops`

Their engine.py dependencies (`_adaptive_spacing`, `_auto_layout_steps`,
`_overlap_interval`, `_override_for_tree`, `_prepare_flex_data`) either move
to the same resolver module or become registered ops. Decide per-helper
in Task 0.1.1.

### Task 0.1.1 -- Op vs resolver classification

Each helper is either:
- a **Pipeline op** (runs inside the pipeline, mutates SolveState): use
  `@register_op`. Example candidate: node-size normalization.
- a **Resolver** (pre-pipeline, returns config dict): lives in
  `pipeline_resolve.py`, plain function. Example candidate: `_auto_layout_steps`.

This classification is a Claude-side decision. Document in
`.project-context/knowledge/decisions.md`.

### Task 0.2 -- Flip the default

In `dagua/layout/engine.py:layout()`, change the code so that `config.algorithm
is None` dispatches to the pipeline registered as `"dagua_native"` in
`PIPELINE_REGISTRY`. The legacy `_layout_inner` path becomes an explicit
opt-in via `config.algorithm="_legacy"`, which is not advertised.

Dispatch Codex with a spec that covers:
- Existing callers of `layout(...)` continue to work.
- `graph.compute_node_sizes()` still runs.
- Direction transform still applied.
- `config.relax_steps` honored in the new path (this may require a new
  `RelaxationStage` op).

### Task 0.3 -- Legacy engine archive

If Q1 in 09_open_questions.md resolves to "archive," move the file to
`dagua/layout/_archive/legacy_engine/engine.py` with a README explaining it
is frozen reference. If Q1 resolves to "delete," delete along with tests.
Hold on this until the user answers.

### Task 0.4 -- MVP fast iteration harness

Create `scripts/iterate_native.sh` (MVP; Sprint 0.5 expands it). Fix per
adversarial review: do NOT reference `TEST_GRAPHS` -- that symbol does not
exist in `dagua/eval/graphs.py`. Use the actual generator factories.

```
#!/bin/bash
set -euo pipefail
GRAPH="${1:-chain_100}"
STEPS="${2:-40}"
python - <<PY
from dagua import LayoutConfig
from dagua.layout import layout as do_layout
from dagua.metrics import quick, composite
from dagua.eval.graphs import make_chain
# MVP: chain-only. Sprint 0.5 adds a full registry via graph_generator.py.
name = "${GRAPH}"
if name.startswith("chain_"):
    n = int(name.split("_")[1])
    g = make_chain(n, seed=42).graph
else:
    raise SystemExit(f"MVP harness: unknown graph '{name}' (Sprint 0.5 adds registry)")
pos = do_layout(g, LayoutConfig(steps=${STEPS}))
m = quick(pos, g.edge_index)
print(f"score={composite(m):.2f}")
print(f"overlaps={m.get('overlap_count',0)}, crossings={m.get('crossing_rate',0)}")
PY
```

Plus an `--image` flag that renders and saves to
`eval_output/native_algo/iter_<timestamp>_<graph>.png`.

Sprint 0.5 replaces this with the full registry-backed harness.

### Task 0.5 -- DEFERRED to Sprint 0.5

Prior draft had Sprint 0 committing 15 held-out JSON topology files from a
public seed. This CONFLICTS with the revised opacity design (salt-derived
graphs, hashes-only manifest). Per 2026-04-22 adversarial review, the
opaque held-out suite and the `graph_generator.py` module move to
Sprint 0.5 (see 02_sprint_map.md). Sprint 0 retains iteration-suite
baseline under the existing seed=42 convention; held-out baseline is
regenerated and measured in Sprint 0.5.

### Task 0.6 -- Baseline metrics (Sprint 0 partial, completed in 0.5)

In Sprint 0: run the default on every iteration-suite graph under the
existing seed=42 convention at the Sprint-0 commit SHA. Save to
`eval_output/native_algo/baseline_sprint_0/metrics.json`.

In Sprint 0.5 (NOT here): re-run with held-out and rolling sets under the
new opacity + 16-variant competitor matrix. That becomes the calibration
baseline for the Pareto ladder.

### Task 0.7 -- DEFERRED to Sprint 0.5

Rolling-seed / sprint-tag / salt-path benchmark flags moved to Sprint 0.5
per 2026-04-22 round-3 adversarial review. Sprint 0 does NOT add these
flags; Sprint 0 uses only fixed seed=42.

### Task 0.7.1 -- DEFERRED to Sprint 0.5

Generator overhead measurement moved to Sprint 0.5. Requires
`graph_generator.py` which is itself a Sprint 0.5 deliverable.

### Task 0.8 -- Rubric-code sync (NEW, from adversarial review)

The drafted 04_evaluation_rubric.md used field names that do not match
`dagua/metrics.py:composite`. Reconcile:

- 0.8a: Rewrite the composite documentation in 04 to exactly match code
  (done in this revision pass -- verify committed).
- 0.8b: For every metric referenced in plan language that does NOT exist in
  code (`symmetry_score`, `cluster_violations`, `label_clipping`,
  `bbox_outside_frame`, `angular_resolution_penalty`), decide per-metric:
  implement (new metric function + unit test) OR remove from plan text.
  Default: remove. Exceptions require user sign-off.
- 0.8.1: Add `composite_large()` for N>2000 quick mode that either uses
  only quick-available fields and renormalizes to 0-100, OR fails if inputs
  are missing. Preferred: fail loudly; a silent default is what created the
  original blind spot.
- 0.8.2: Extend `scripts/quality_runtime_analysis.py` to compute the four
  gating formulas from 04: `overfit_gap`, `rolling_gap`, `cumulative_drift`,
  `family_cumulative`. Missing inputs = exit fail.

### Task 0.9 -- DEFERRED to Sprint 0.5

Secret-salt infrastructure and opaque held-out generation moved entirely
to Sprint 0.5. Sprint 0 has NO salt / opacity / held-out work. The
original Task 0.9 detailed spec is consolidated in 02_sprint_map.md
Sprint 0.5 exit criteria + the historical appendix at the end of this file.

### Task 0.10 -- Memory parity baseline and porting plan (NEW)

The current ops pipeline (`dagua_native.py` line 429-455) uses
`LossGroup(backward_mode="combined")` which retains all loss graphs until a
single backward. The legacy `_layout_inner` has per-loss backward,
checkpointing, and hybrid device support that the ops pipeline lacks.

Tasks:
- 0.10a: Measure RSS for both paths (ops pipeline and `_layout_inner`) on
  1K, 10K, 100K graphs (100K skipped if lack of RAM). Commit as
  `memory_profile.json`.
- 0.10b: If ops pipeline RSS exceeds legacy by >20% at any tier, Sprint 1
  must port per-loss backward as its FIRST exit criterion. This turns the
  scaling promise in 03 from aspirational to grounded.
- 0.10c: Register three ops for Sprint 1: `LossPerLossBackward`,
  `GradientCheckpoint`, `HybridDeviceOffload`. Skeleton only in Sprint 0;
  implementation in Sprint 1.

## Test plan

- Unit: each new op has a direct unit test matching its pre-decomposition
  behavior.
- Integration: `pytest tests/test_pipeline_dagua_native.py` -- extend to check
  bit-for-bit equality before/after decomposition on a 100-node random DAG
  with seed=42.
- Smoke: `scripts/iterate_native.sh chain_100` completes and emits a score.
- Regression: baseline metrics JSON must match today's default within floating
  point noise on the iteration suite.

## Adversarial review plan

Dispatch `/codex:adversarial-review` once before Task 0.2 with focus:
"attack the decomposition for hidden coupling -- did we miss any engine.py
helper that the pipeline silently needs? Any op that secretly mutates state
outside SolveState?"

Dispatch a second adversarial review after Sprint 0.5's rolling-seed work
(NOT in Sprint 0) with focus:
"attack the rolling-seed strategy for repeatability and for anti-overfit
weakness -- can a clever engineer still overfit even with rolling seeds?"

Fix CRITICAL/HIGH findings before exit.

## Rollback plan

- Decomposition rollback: `git revert` the Task 0.1 commits.
- Default flip rollback: revert Task 0.2 commit. Legacy path stays intact
  until Task 0.3.
- If Task 0.4 iteration harness is slow in reality (>15s), investigate
  `torch.compile`, lazy-load metrics, reduce default `steps` to 20.
- Held-out suite rollback handled in Sprint 0.5 (this sprint does not
  touch held-out).

## Open questions (for Sprint 0 specifically)

- Should the decomposed resolver module live under `dagua.layout.ops/` or under
  `dagua.layout/`? See 09 Q8.
- Where does `_layout_inner` live post-archive? See 09 Q1.
- Does the user want the pre-decomposition default exactly preserved (bit-for-bit
  when possible) or are we free to let small numerical differences land?
  See 09 Q9.
