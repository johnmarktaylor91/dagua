# Bug Notes: Competitor Algorithm Pipeline Retro

## Category: Verification Failures

### Bug: Silent seed duplication (Incident 6)
**Symptoms:** 10-seed benchmark appeared to run correctly. No errors.
**Root cause:** 15 adapters hardcoded seed=42, ignoring the seed the pipeline
thought it was passing. Ten "different" runs = ten identical outputs.
**Fix:** Add seed parameter to CompetitorBase interface, thread through all adapters.
**Architectural lesson:** Verify outputs differ, not just that the pipeline runs.
A pipeline that produces identical outputs silently is worse than one that crashes.

### Bug: Aggregate metrics hiding bad data (Incidents 1, 8)
**Symptoms:** "dagua has 24M crossings" / "all reimplementations are faithful"
**Root cause:** Averaging across heterogeneous data (different graph sizes,
different algorithm families) hides structure. One outlier or confound invalidates
the mean.
**Fix:** Per-graph reporting, stratification by graph family, adversarial review.
**Architectural lesson:** Never report only aggregates. Always show the distribution.

### Bug: Weighted test graph masquerading as simple (Incident 9)
**Symptoms:** 0.18 Procrustes disparity on "simple" karate club graph.
**Root cause:** nx.karate_club_graph() has edge weights 1-7. NX spectral uses
weights; our code treats all edges as weight 1.
**Fix:** Use explicitly unweighted graphs for fidelity comparison.
**Architectural lesson:** Never assume you know a test graph's properties. Print
them. Check for weights, self-loops, multi-edges, disconnected components.

## Category: RNG Mismatches

### Bug: torch.rand vs random.random (Incident 10)
**Symptoms:** FA2 disparity 0.74 despite correct formulas.
**Root cause:** Different RNG engines (PyTorch vs Python stdlib) produce
different sequences from the same seed integer.
**Fix:** Use the same RNG as the reference (random.random() for FA2).
**Architectural lesson:** "Same seed" means nothing across different RNG
implementations. Match the exact RNG source.

### Bug: C-level shuffle unreproducible from Python (Incident 11)
**Symptoms:** Stress-SGD disparity 0.65 despite correct formulas and init.
**Root cause:** s_gd2's C++ uses randomkit for shuffling. Python can't reproduce
this RNG. The shuffle order determines the optimization trajectory.
**Fix:** Compare objective values (stress) instead of positions (Procrustes).
**Architectural lesson:** For C-extension references, accept that position-level
exact match may be impossible. Use objective-level comparison.

## Category: Infrastructure Assumptions

### Bug: Python bindings assumed necessary (Incidents 12, 13)
**Symptoms:** "OGDF unavailable" for 7 algorithms.
**Root cause:** Assumed ogdf-python (cppyy bindings) was the only way to call
OGDF. It wasn't — subprocess works fine, same as Graphviz/dagre/ELK.
**Fix:** 50-line C++ subprocess wrapper, compiled against libOGDF.so.
**Architectural lesson:** "Can we run it and get positions back?" is the only
question. The mechanism is irrelevant.

### Bug: ProcessPoolExecutor + TorchLens fork (Incident 7)
**Symptoms:** Workers spawned but did zero work (0 CPU time).
**Root cause:** TorchLens imports are not fork-safe. Worker processes deadlocked
during graph loading.
**Fix:** Serial execution mode (--workers 1).
**Architectural lesson:** Test execution modes on tiny data before full runs.

## Category: Process Failures

### Bug: Paper-first vs code-first (Incident 14)
**Symptoms:** Multiple Codex rounds produced code that didn't match references.
**Root cause:** Specs said "match the paper" but papers are ambiguous.
Reference implementations often differ from papers (different defaults, extra
heuristics, implementation choices).
**Fix:** Read the actual source code. Translate line by line.
**Architectural lesson:** The code IS the algorithm. Papers describe it.

### Bug: Premature victory declaration (Incident 8)
**Symptoms:** "All faithful. No red flags." — demolished by adversarial review.
**Root cause:** Used one weak metric (Procrustes ratio), averaged across
heterogeneous graphs, interpreted optimistically.
**Fix:** Adversarial review before claiming results.
**Architectural lesson:** The adversary found 10 real problems in 5 minutes that
I missed. Always get adversarial review before declaring fidelity.

## Summary Table

| # | Bug | Category | Time Wasted | Could Have Been Prevented By |
|---|-----|----------|-------------|------------------------------|
| 6 | Silent seed duplication | Verification | ~10 hours | Checking two outputs differ |
| 8 | Premature fidelity claim | Verification | 2+ dispatches | Adversarial review first |
| 9 | Weighted test graph | Verification | ~1 hour | Printing graph properties |
| 10 | Wrong RNG engine | RNG | 2+ dispatches | Matching exact RNG source |
| 11 | C shuffle barrier | RNG | Investigation time | Knowing C RNG limits upfront |
| 12-13 | Python bindings assumed | Infrastructure | ~30 min | "Can we run it?" question |
| 7 | Fork-unsafe imports | Infrastructure | ~30 min | Smoke test before full run |
| 14 | Paper vs code mismatch | Process | 2+ dispatches | Reading code first |
| 1 | Misleading aggregates | Verification | None (caught by user) | Per-graph reporting |
| 3 | Wrong resume flag | Process | ~20 min | Understanding flags first |
