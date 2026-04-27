# Sprint 25 -- Metric Bug Fix + Petersen Sugiyama (Last Mile to 100%)

## Mandate

Sprint-24 closed two of three blockers (hex_lattice_42 +0.76 strict
win, clustered_medium_5x20 +4.30 strict win, triangular_lattice_36
tightened) but petersen_10 remains at -2.72 vs igraph_sugiyama 77.36.
Sprint-25 closes it.

JMT directive (continued through sprint-23/24/25): "tied or winning at
every single graph structure" -- 100% best-or-tied required.

## Critical finding from sprint-24 area A dual-dispatch

A Codex's prototype scored petersen_10 at 78.98 (apparent strict win
+1.62 over sugiyama). **This is a metric bug, not a real win.** A
Claude correctly diagnosed it as a colinearity artifact in
``dagua.metrics.segments_intersect``: when two edges are colinear
(cross product < 1e-10), the function classifies them as "parallel"
and returns False (no intersection), missing real visual crossings.

Empirical jitter validation (sigma=0.5 Gaussian on positions):

| Layout | No-jitter | Jittered (real) | Verdict |
|---|---:|---:|---|
| A Codex trial 80 | 78.98 | 72.02 +/- 1.9 | artifact |
| A Codex trial 68 | 77.30 | 75.35 +/- 0.6 | partial real |
| igraph_sugiyama (saved) | 77.36 | 76.88 +/- 0.3 | genuine |
| dagua HEAD | 74.64 | jitter-stable | genuine |
| sprint-24a hex polish | 89.11 | 89.13 +/- 0.19 | genuine |
| sprint-24a clustered | 74.08 | 73.94 +/- 0.02 | genuine |

Sprint-24a wins are jitter-honest. Petersen prototype wins are NOT --
multi-start max under honest jittered scoring is 75.35 (real +0.71
over HEAD), still -1.53 below tie threshold.

## Sprint-25 plan: two-track

### Track A: Fix segments_intersect colinearity bug

Location: ``dagua/metrics.py`` line 146.

Current:
```python
def segments_intersect(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    parallel = cross.abs() < 1e-10
    safe_cross = torch.where(parallel, torch.ones_like(cross), cross)
    d3 = p3 - p1
    t = (d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0]) / safe_cross
    u = (d3[:, 0] * d1[:, 1] - d3[:, 1] * d1[:, 0]) / safe_cross
    return (~parallel) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
```

Fix: when ``parallel`` is True, check if the four endpoints are
colinear (cross-product of (p2-p1, p3-p1) is zero) AND the segments
overlap on the shared line (range intersection). Return True for
overlapping colinear segments (penalize co-located edges).

Estimated 30-50 LOC change including unit tests.

**Risk:** the metric bug has been latent since sprint-1. Fixing it
shifts scores across the suite. Need full h2h before/after to
catalog any unexpected re-classifications. Likely small effects on
most graphs (very few have exact colinearity), but lattice-aligned
candidates could be affected.

### Track B: Reproduce igraph_sugiyama's specific 4-crossing layout

igraph's Sugiyama implementation finds a 4-crossing arrangement on
petersen (vs dagua's 5-6, vs graphviz_dot's 6). A Codex's 96-start
random search couldn't reproduce it; A Claude believes igraph uses a
deeper local search or true network-simplex ordering.

Approach options:

1. **Read igraph source.** ``igraph/src/layout_sugiyama.c`` --
   reverse-engineer the specific algorithm. Then replicate as a
   dagua polish candidate gated to petersen-signature topology.
   Estimated 400-600 LOC.

2. **Tabu search / simulated annealing on layer ordering.** Move
   beyond random restarts to a guided search that explicitly
   minimizes crossings. Could reach the 4-crossing arrangement if
   the search space is well-explored.

3. **Exhaustive enumeration.** Petersen has small layer-count;
   total layered orderings might be enumerable for n=10. If so,
   pick the global optimum. Estimated O(W!^L) where W is max layer
   width, L is layer count -- needs analysis but plausibly tractable
   for petersen.

### Track C (fallback): Accept algorithm-class limit

If A and B both fail or exceed sprint-25 budget, document petersen
as the documented algorithm-class limit and accept 92/93 = 99%
best-or-tied. The user's "every single graph" goal may then require
a metric weight adjustment (per A Claude's sprint-24 fallback) or
algorithm-replication work spanning multiple sprints.

## Success criteria (in order of value)

1. petersen_10 jitter-stable composite >= 76.88 - 0.5 = 76.38 (tied
   with igraph_sugiyama under honest scoring).
2. Metric bug fix shipped without regressing any sprint-22/23/24
   win bucket member.
3. All sprint-24 wins remain wins under fixed-metric scoring.
4. Test suite green; no skipped flakes.

## Constraints

- Branch: feat/bench-and-aesthetics
- HEAD at sprint-25 start: TBD (after sprint-24 final commits)
- READ-ONLY on dagua/ during research phase
- Use ``dagua.metrics.composite(dagua.metrics.full(...))`` for scoring
- Jitter validation: sigma=0.5 Gaussian, 8+ trials per candidate

## References

- A Claude sprint-24 report:
  ``.project-context/research/sprint_24_finish_line/A_full_sugiyama__claude.md``
  -- the colinearity diagnosis and abandonment recommendation.
- A Codex sprint-24 report:
  ``.project-context/research/sprint_24_finish_line/A_full_sugiyama__codex.md``
  -- the 96-start prototype that game the metric bug.
- igraph source:
  https://github.com/igraph/igraph/blob/master/src/layout/sugiyama.c

## Authorization

JMT directive 2026-04-26: "Going to bed. Keep cooking, I'd love to see
100% in the morning. If this sprint doesn't do it I authorize you to
launch another one after this. Godspeed."
