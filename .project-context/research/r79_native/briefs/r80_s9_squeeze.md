# r80-S9: Placement squeeze round (clustered-undirected candidate + weighted semantics)

## Context
Trunk r79/native (head ddeeb74) is at 89/108 best-or-tied (verified from the S2b branch
sweep: legacy 64/14/15 + extended 9/2/4; the committed trunk store still shows the older
87-state -- your gate sweep will certify the merged head). Remaining undirected-class
losses include two named, evidence-backed opportunities:

1. **Clustered-undirected graphs never reach the portfolio** -- the engine's cluster
   driver preempts routing (S4 evidence: P8_PORTFOLIO_EVIDENCE.md "Caps/residuals").
   S4's probe showed flat sfdp would flip r79_undirected_sbm_high_mix_3x30. Candidates
   r79_undirected_sbm_{low,mid}_* are already wins; high_mix loses by 3.8.
2. **Weighted community graphs treat edge weights as DISTANCES in shortest-path targets**
   while community semantics mean SIMILARITY (heavy intra-community edges should pull
   nodes together, not push apart). P3B2_STRESS_FORENSICS.md Ranked Fix 4 (file refs
   there). Now partially addressed by the portfolio winning those graphs anyway --
   remaining weighted losses to inspect: check the current store for weighted graphs
   still losing (e.g. r79_weighted_small_world_120 at ~34 vs external; heavy-tail cases).

## Setup
Worktree /home/jtaylor/.claude/worktrees/dagua-native-p1 (free; on merged branch).
  git checkout -b r80/squeeze $(git -C /home/jtaylor/.claude/worktrees/dagua-native rev-parse r79/native)
Venv exists. Verify import resolves in p1.

## Deliverable 1: clustered-undirected portfolio access
For declared-undirected graphs WITH clusters, let the portfolio contest run with a
cluster-aware challenger set: (a) incumbent (current cluster-driver output), (b) the
cluster-aware sfdp driver path if available, (c) flat sfdp + both cleanup variants
(ignoring clusters for placement, keeping them for scoring). The referee + degeneracy
guard decide, as always. Cluster containment metrics must be part of the scoring frame
(they are in the composite's cluster term -- verify it is actually computed for these
graphs in the contest proxy). Gate expectation: r79_undirected_sbm_high_mix_3x30 flips
LOSS->WIN or TIE; low/mid stay wins; the 3 nested_clusters graphs and
clustered_medium_5x20 must not regress.

## Deliverable 2: weighted-similarity semantics (predicate-gated, contest-protected)
Add a challenger VARIANT (not a global change): for declared-undirected weighted graphs,
a candidate whose distance targets use transformed weights (w -> 1/w or 1/sqrt(w) --
pick by a 3-graph mini-probe, document choice) in the Dijkstra targets for the stress/
force paths. It enters the contest as one more candidate; the referee decides per graph.
NO changes to default weight handling anywhere.

## Gates
1. Scoped tests + new tests for both deliverables (KNOWN_RED deselects; no bare -x).
2. Full gate sweep with the NEW --fresh flag (scripts/r79_baseline.py --dagua-only
   --fresh). Acceptance: zero WIN->LOSS vs the 89-state (legacy 64/14/15 + extended
   9/2/4 -- compare per-graph against the p1 branch store you inherit or regenerate),
   net >= 0, and at least +1 best-or-tied. Record per-graph deltas.
3. ruff on touched files.
4. Launch the sweep with nohup (survives turn end); if your monitor dies, the architect
   harvests -- leave the log at /tmp/r80_s9_sweep.log.

## Output contract
Commits on r80/squeeze; evidence .project-context/research/r79_native/P12_SQUEEZE.md
(probe tables, per-graph deltas, W/T/L, candidate win-rate changes); final message with
the W/T/L line and concerns.

## Hard rules
- Portfolio philosophy: ADD candidates, never replace or remove existing ones.
- Do not touch: projection.py internals, metrics composite functions, eval adapters,
  scripts/r79_baseline.py, routing/edges code (S7's domain).
- ASCII; disk check; clean /tmp scratch except the named sweep log.
