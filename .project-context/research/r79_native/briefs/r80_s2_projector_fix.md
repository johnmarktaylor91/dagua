# r80-S2: Convergent overlap projector (solo correctness fix, metric-gated)

## Context
dagua's size-aware overlap projector (dagua/layout/projection.py, the `_project_exact`
small-graph path around lines 222-246) uses torch advanced-index in-place adds
(`pos[x_r] += ...`) where repeated node indices DO NOT accumulate (last-write-wins).
On dense overlap cliques it therefore never converges: 50 iterations leave 37+ overlaps
on sbm_4x30. Each unresolved overlap pair costs the layout its 20-point overlap term in
our composite. Forensics: .project-context/research/r79_native/P3B2_STRESS_FORENSICS.md
(read the "Ranked Fix List" items 1 and 5).

A previous attempt lives on branch r79/p3b-wip (commit cd04c7e) -- read
`git show cd04c7e` output for prior art on index_add_ accumulation, but implement clean;
that bundle failed its sweep gate for unrelated reasons.

## Task
Work in /home/jtaylor/.claude/worktrees/dagua-native-p1. First:
`git checkout -b r80/projector $(git -C /home/jtaylor/.claude/worktrees/dagua-native rev-parse r79/native)`
(create the branch at the current r79/native head; do NOT touch the dagua-native worktree).
If the worktree lacks a .venv, reuse the main one for running:
`/home/jtaylor/.claude/worktrees/dagua-native/.venv/bin/python` works for pytest/scripts
but you must run from the p1 worktree dir so ITS source is imported -- verify with
`python -c "import dagua, pathlib; print(pathlib.Path(dagua.__file__).resolve())"` and if it
resolves to the wrong tree, create a venv: `uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]" python-igraph`.

### Part 1: convergent projector
In projection.py, fix the exact-path displacement accumulation:
- Accumulate per-node pushes with `index_add_` (or scatter_add) over ALL overlapping pairs,
  then apply with a damping factor (start 0.7; make it a parameter).
- Iterate until zero overlaps OR no-progress (overlap count not strictly decreasing for
  3 consecutive iterations) OR max_iters (default generous, e.g. 200 for the exact path).
- Preserve the public op/function signatures. Keep it pure torch, deterministic, no RNG.

### Part 2: metric-gated acceptance (the safety net)
Wrap the projection call sites used by native pipelines (native_stress.py final projection
stage; find others via grep for the projector entry point) with a gate:
- Compute a cheap proxy composite BEFORE and AFTER projection using the REAL metric terms
  (overlap count, sampled crossings with the project's fixed metric seed, edge-length CV).
  Use existing functions from dagua.metrics -- do not reimplement formulas.
- Keep the projected result iff proxy(after) >= proxy(before); else return the input
  positions unchanged. Log a debug line when rejected.
- Implement as a registered op (decomposable-ops philosophy -- no private helper functions
  doing op-level work), e.g. `overlap_projection_gated`.

## Gates (in order; STOP and report if a gate fails twice)
1. Scoped tests: `pytest tests -k "projection or overlap" --tb=short` plus any test file
   touching native_stress. Consult
   .project-context/research/r79_native/KNOWN_RED_TESTS.md and deselect those. NEVER run
   bare `pytest tests/ -x` (full suite whack-a-mole trap).
2. Unit proof of convergence: script a 30-node dense overlap clique (all nodes at nearly
   the same point, real label-size boxes), run the fixed projector, assert 0 overlaps and
   report iterations used. Add this as a pytest test.
3. Full dagua-only sweep: `python scripts/r79_baseline.py --dagua-only` (writes
   eval_output/r79_baseline/). NOTE: another sweep may be running in the dagua-native
   worktree -- yours runs in dagua-native-p1 so output dirs are separate; confirm the
   script writes within the worktree (relative path) before running.
   Acceptance: net composite delta across the corpus >= 0 AND no graph flips WIN->LOSS
   vs the pre-change baseline (record the per-graph before/after table). Individual
   composite drops are acceptable if a graph stays in the same W/T/L bucket.
4. `ruff check` on touched files only.

## Output contract
- Commit(s) on r80/projector with conventional messages (no AI attribution).
- Evidence file .project-context/research/r79_native/P7_PROJECTOR_EVIDENCE.md in the p1
  worktree: what changed, convergence-proof numbers, per-graph sweep delta table,
  W/T/L before/after, rejected-gate statistics (how often the gate rejected projection).
- Final message: summary + the W/T/L line + any concerns. If the sweep gate fails, report
  the failing graphs and their per-term deltas instead of trying broad rewrites.

## Hard rules
- Touch ONLY: projection.py, native_stress.py (call-site gating), ops registry glue, new
  test file(s). Do NOT touch routing (_choose_native_pipeline), dagua_flat, layered code.
- No runtime delegation to external layout binaries. Pure torch.
- ASCII only in all files. Watch disk (19GB free): no large scratch, clean any /tmp files.
