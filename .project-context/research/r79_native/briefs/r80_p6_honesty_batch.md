# r80-P6: Harness honesty batch (implementation only -- the re-freeze run comes later)

## Context
The S1 adversarial audit (.project-context/research/r79_native/P8C_HARNESS_AUDIT.md) found
three defects we deferred to one batch, plus a stale-resume hole found later (agent-issue
ledger 2026-07-09). This stream IMPLEMENTS all fixes and validates them on a small graph
subset. The full 9-engine benchmark re-freeze runs LATER on the final merged trunk -- do
NOT run it here.

## Setup
Worktree: /home/jtaylor/.claude/worktrees/dagua-native-p2 is free (S4 done, merged).
  cd there; git checkout -b r80/p6-honesty $(git -C /home/jtaylor/.claude/worktrees/dagua-native rev-parse r79/native)
Its .venv exists from S4; verify import dagua resolves in p2.

## Deliverables (each its own commit)

### 1. Provably-fresh gate sweeps (stale-resume hole)
scripts/r79_baseline.py: add --fresh flag that ignores/clears the resume jsonl and
refuses cached rows; stamp every written row with the current git sha + timestamp; add
a loud warning line when a sweep DOES reuse resumed rows (count them). Unit test with a
tiny fake store.

### 2. Size-aware external engines (S1 HIGH-2)
Externals are laid out size-blind but scored with dagua's label-size boxes -- biased FOR
dagua; we want the honest comparison. Pass per-node sizes to engines that support them:
- graphviz (dot/sfdp/neato): width/height node attrs in points (graphviz uses inches --
  convert; fixedsize=true). Mind the existing spline capture (S6) -- do not break it.
- elk_layered: node width/height in the JSON request.
- dagre: width/height per node.
- igraph/nx_spring/others without size support: leave as-is, document in the adapter
  docstring that they are size-blind (their overlap term will simply reflect that).
Add an adapter-level flag so the OLD size-blind behavior remains available
(--size-blind-externals) for store-compatibility experiments. Validate on 5 graphs:
externals get FEWER overlaps with sizes passed (print before/after overlap counts).

### 3. Degeneracy guard in the composite (S1 HIGH-3)
The composite awards ~65/100 to a point-collapapsed layout (zero-length edges ace
edge_length_cv + crossing_rate despite total overlap). Add a guard INSIDE the composite
functions (composite/composite_undirected/composite_auto): when mean edge length <
0.25 * mean node diagonal (degenerate scale), the length-uniformity and crossing terms
score 0 (not their vacuous maxima). Document the threshold. This CHANGES THE RULER --
that is sanctioned for this batch, but you MUST quantify the blast radius: rescore the
CURRENT frozen store (--rescore-only path) and report every row whose composite moves
by >0.5 and any W/T/L change. Do not tune the threshold to protect dagua rows; report
what happens honestly.

### 4. composite_large undirected variant (S1 MEDIUM-1)
composite_large has no undirected flavor (dead code today, landmine at scale). Add it
mirroring composite_undirected's term structure at the large-graph tier; unit test both
flavors.

### 5. Validation summary
.project-context/research/r79_native/P11_HONESTY_BATCH.md: what changed, the blast-radius
table from deliverable 3, before/after overlap counts from deliverable 2, and the exact
command for the LATER full re-freeze (all engines, --fresh, size-aware).

## Gates
1. Scoped tests (new + touched files; KNOWN_RED deselects; no bare pytest -x).
2. Deliverable-3 blast radius report REQUIRED before commit (even if ugly).
3. ruff on touched files.
4. Do NOT run the full 9-engine benchmark. Do NOT re-freeze the store. Implementation
   + small-subset validation only.

## Hard rules
- Do not touch dagua/layout/** (pipelines, projection, routing) -- this is eval/metrics/
  adapters/scripts only.
- Additive where possible; the composite guard is the ONE sanctioned scoring change.
- ASCII; disk check before big steps; clean /tmp scratch.
