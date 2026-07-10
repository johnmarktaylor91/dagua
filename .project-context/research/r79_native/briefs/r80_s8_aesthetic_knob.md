# r80-S8: User-facing aesthetic-priority knob

## Context (JMT directive 2026-07-08, promoted into this sprint)
Users can already tweak differentiable loss weights and write custom constraints, but the
portfolio SELECTION composite has fixed weights -- engine choice is not steerable. Add an
aesthetic-priority knob that plumbs into BOTH the differentiable losses AND the candidate-
selection composite, so the whole stack (routing, core, refinement) optimizes the user's
point on the quality frontier.

IMPORTANT: this is a PUBLIC API surface. JMT signs off on the final shape BEFORE merge.
You implement behind a clean internal seam and present 2-3 API-shape options with worked
examples in your report; the internal machinery must support all of them so the sign-off
is a thin rename/wrapper decision, not a rework.

## Setup
Worktree /home/jtaylor/.claude/worktrees/dagua-native-p3 is free (S7 merged).
  git checkout -b r80/s8-aesthetic-knob $(git -C /home/jtaylor/.claude/worktrees/dagua-native rev-parse r79/native)
Venv exists; verify import resolves in p3.

## Design requirements
1. Internal representation: a normalized priority profile over the composite's term
   families (edge-length uniformity, crossings, overlaps, angular resolution, cluster
   quality; optionally the drawing terms for routing). Default profile == exactly
   today's weights (None -> identity).
2. Wire-through A (selection): the portfolio contest's scoring op reweights terms by the
   profile. Same profile must be used for ALL candidates in one contest (fairness).
3. Wire-through B (losses): map the profile to the differentiable loss-weight multipliers
   used by the native cores (read how loss weights resolve in the engine/loss_engine;
   apply multiplicative adjustments, document the mapping table).
4. Candidate API shapes to present (implement the machinery once, expose behind
   LayoutConfig): (a) preset strings: prioritize="crossings" | "uniform_edges" |
   "compactness" | "readability" (document what each maps to); (b) explicit dict:
   aesthetic_weights={"crossings": 2.0, "edge_length_cv": 0.5}; (c) both, dict overrides
   preset. Recommend one in your report with rationale.
5. The BENCHMARK never sets the knob (its scoring stays the frozen honest composite --
   the knob changes what the user optimizes for, not how we score benchmarks).

## Gates
1. Default-identity proof: with the knob unset, full dagua-only --fresh sweep is
   bit-identical W/T/L and zero movers vs the committed store (52/13/28 + 6/3/6).
   Launch with nohup, log /tmp/r80_s8_sweep.log.
2. Efficacy test (pytest): on at least one real corpus graph, prioritize="crossings" vs
   prioritize="uniform_edges" selects DIFFERENT contest winners, and each winner is
   better than the other on the prioritized term (prove the knob does what it says).
3. Loss-path test: a priority profile measurably shifts the corresponding loss weight
   in the resolved config (unit test, no full layout needed).
4. Scoped tests + ruff.

## Output contract
Commits on r80/s8-aesthetic-knob; evidence .project-context/research/r79_native/
P15_AESTHETIC_KNOB.md (API options with worked examples + recommendation, mapping
tables, gate results); final message: summary + the API-shape question for JMT.

## Hard rules
- Default behavior bit-identical everywhere. The knob is opt-in.
- Do not modify the benchmark harness scoring or the frozen store.
- ASCII; nohup for the sweep; disk check first.
