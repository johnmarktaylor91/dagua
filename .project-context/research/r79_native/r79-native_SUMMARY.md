# r79/r80 Native Placement + Full-Drawing Sprint -- FINAL SUMMARY

Status: r80 COMPLETE 2026-07-10. Branch r79/native (worktree ~/.claude/worktrees/dagua-native),
head = merge 8d4787b. Durable mirror: ~/.claude/research/dagua/r79-native/. This file is the
cold-start entry point; per-round detail in r79-native_STATE.md and the P-series evidence docs.

## Final numbers (all on the HONEST ruler: size-aware+prism externals, degeneracy-guarded
## composite, provably-fresh sweeps)

| Measure | Value | Evidence |
|---|---|---|
| Iteration corpus (108 graphs) | 52W/12T/29L + 6/3/6 = 73/108 (67.6%) best-or-tied | frozen store @ head |
| Pre-sprint on SAME ruler | 55W/8T/45L = 63/108 (58.3%) | P13_COUNTERFACTUAL |
| TRUE sprint gain | +11 gross, -1 honest reversion = +10 net, ZERO regressions | P13 + P17 |
| HOLDOUT (271 unseen graphs) | 126W/74T/71L = 73.8% best-or-tied | P14_HOLDOUT |
| Holdout per corpus | rome 65.1%, north 91.6%, suitesparse 25.0% | P14 |
| Drawing quality (10-graph probe) | dagua rows +2.36 mean, 10/10 improved, 0 drops | P10 + P9 |
| Adversarial review | SAFE WITH FIXES -> both fixes landed + verified | P16 |
| Visual audit | 1 metric-gamed win found + honestly reverted; hex lattice CLEAN WIN | P17 |

HEADLINE: holdout (73.8%) > iteration (67.6%) -- the algorithm GENERALIZES; zero overfitting.
The old-ruler numbers (74->90/108) overstated gains because the harness had been favoring
dagua (size-blind externals, degenerate-collapse exploit, edgeless-node blindness); both the
algorithm improvement (+10 honest) and the ruler repair are now separately quantified.

## What shipped (merged to r79/native, all invariance/gate-proven)
1. **Undirected portfolio route** (S4 + fixes): declared-undirected/reciprocal graphs run a
   contest -- incumbent vs dagua's own sfdp/neato reimplementations (+cluster-aware and
   weighted-similarity variants) finished with overlap cleanup; honest-composite argmax wins;
   degeneracy + isolated-fling guards; N<=1500 + time-budget caps. Directedness provenance
   plumbing (declared vs inferred) + span-aware deep-layering inference fix.
2. **Convergent overlap projector** (S2b): index_add_-accumulated, damped, iterate-to-zero --
   OPT-IN and challenger-only (default path bit-preserved); both cleanup variants contest.
   Referee honesty PROVEN (zeros on all divergence hypotheses, P7).
3. **Routing upgrades** (S7/S7b): node-bbox avoidance (chord-scaled, crossing-aware per-edge
   referee), port angular spread (density-scaled; dot-parity direction), orphaned
   BezierControlPointOpt wired at quality>=high, edge-label search widened. Placement
   bit-invariant (proven pre+post merge). Non-finite-position guards.
4. **Full-drawing measurement** (S6): composite_drawing (routed crossings, bends, edge-node,
   ports, labels), graphviz/ELK native spline capture, routes store blob. Additive-only,
   bit-identical invariance proof.
5. **Harness honesty batch** (P6): --fresh provably-fresh sweeps + row git-sha stamping;
   size-aware externals + overlap=prism (sfdp 1774->0 overlaps); composite degeneracy guard;
   composite_large undirected variant; full 9-engine re-freeze.
6. **Isolated-fling guard + repair-on-detect** (visual-audit blocker): pack singletons only
   when >8x median fling detected; 2 fake wins honestly reverted (bipartite, er_500 -- raw
   candidate fling 15-43x masked by post-projection store positions).
7. **Holdout infrastructure**: Rome/North/SuiteSparse corpora + loaders + spawn-isolated
   eval (3 infra bugs found+fixed: 101GB in-process OOM, fork-after-torch deadlock,
   publish-replace data loss).

## Built, pending JMT decision (NOT merged)
- **S8 aesthetic-priority knob** (r80/s8-aesthetic-knob, p3): priority profile -> portfolio
  selection reweighting + loss multipliers. All gates passed (default bit-identical; efficacy
  proven: different priorities select different winners). AWAITING API-shape sign-off:
  (a) presets only / (b) dict only / (c) both, dict overrides [recommended]. P15.

## r81 backlog (named, evidence-backed)
1. Mesh/structural class (suitesparse 25%, iteration-corpus lattices): the one weak class.
   Candidates: param-matched neato/sfdp candidates, resistance-distance targets (never tried).
2. Rome small-undirected tail: graphviz_neato leads losses; candidate param-matching.
3. Pre-projection repair competing (er_500 could be won back honestly).
4. Drawing gap to dot native splines (~8.6pt mean): optimizer default-on cost (226s -> needs
   idle re-measure), corridor-aware routing, enX zero-out (3/10 currently).
5. Dense-graph memory blowup: 124-node/5972-edge graph -> >50GB RSS (repro: suitesparse/
   Journals). Real bug.
6. Load-nondeterminism: contest selection can flap under CPU contention (weighted_small_world
   45.28 vs 34.41; silent challenger drop suspect). Determinism-under-load audit.
7. Deferred LOWs from P16: composite_large_undirected not wired into benchmark; reciprocity
   O(E) on classify path (capped 100k); _grid_candidates set-order.
8. Composite-vs-human calibration: S6 proxies + P17 failure modes (readability collapse on
   large-spread layouts; composite indifferent to tiny-node illegibility).
9. Aesthetic knob follow-ups: same-engine selectors (multi-start, polish) not yet
   priority-aware; per-quality-level benchmark validation.

## Key doctrine wins this sprint (memorialize)
- Portfolio + honest referee turns improvement monotone: 6 merge gates, zero regressions.
- Instrument-over-inference won EVERY contested diagnosis (8 bisections, all named causes).
- The visual audit caught what every metric missed (edgeless-node fling) -- keep Opus audits
  in the loop; composites cannot referee alone.
- Holdout discipline (zero tuning) converted a good iteration number into a PROVEN
  generalization claim -- this is the publication evidence base.
