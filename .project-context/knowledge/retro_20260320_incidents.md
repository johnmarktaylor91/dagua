# Incident Log: Competitor Algorithm Pipeline (March 18-20, 2026)

## Incident 1: Graphviz Timeout Bug Discovery
**When:** Session start (March 18)
**What happened:** User asked to review competitor results. I reported aggregate
numbers showing dagua had 24M crossings vs competitors' 8K. User asked if the
comparison was apples-to-apples. It wasn't — dagua was the only engine running
on 50K-100K node graphs, inflating its averages.
**Deeper issue found:** graphviz_utils.py had a hardcoded 30-second subprocess
timeout, ignoring the 300s the benchmark passed. graphviz_sfdp was failing on
graphs it should have handled.
**Fix:** Added timeout parameter to layout_with_graphviz() and
render_graphviz_native(), threaded through _GraphvizBase.layout().
**Time wasted:** None — this was a legitimate discovery.
**Lesson:** Always verify aggregate metrics aren't mixing apples and oranges.

## Incident 2: dagua_competitor Hardcoded CPU
**When:** March 18, discovered during adapter audit
**What happened:** dagua_competitor.py hardcoded `device="cpu"`, never showing
GPU performance. The benchmark was systematically underreporting dagua's speed.
**Fix:** Auto-detect CUDA, add .detach().cpu() to returned positions.
**Lesson:** Audit ALL adapter configurations, not just the ones that error.

## Incident 3: Benchmark Re-Run Waste
**When:** March 18, first salvage attempt
**What happened:** I launched `--no-resume` which restarted the ENTIRE benchmark
from scratch instead of just re-running failed competitors. User caught this:
"why are we rerunning everything rather than just the failures?"
**Fix:** Killed, relaunched without --no-resume.
**Root cause:** I didn't think through the flag semantics before running.
**Time wasted:** ~20 minutes of unnecessary computation.
**Lesson:** Think before running. Understand what the flags do.

## Incident 4: OOM Kill During Benchmark
**When:** March 18, benchmark at 75%
**What happened:** The 200M node layout from another Claude session consumed
83GB RAM, triggering the OOM killer which also took out our benchmark process.
12,642 results were saved thanks to checkpointing.
**Fix:** Resumed with --resume.
**Lesson:** Per-competitor checkpointing (which we added) saved us here.

## Incident 5: Three Benchmark Scripts
**When:** March 18-19
**What happened:** User asked "why three scripts to capture everything?" when I
launched run_all_layouts.py, generate_reimpl_layouts.py, and
generate_ground_truth.py separately. Good question — they do the same core job.
**Fix:** Wrote unified scripts/run_benchmark.py.
**Root cause:** Historical accumulation of scripts written at different times.
**Lesson:** Consolidate before running, not after.

## Incident 6: Seed Handling Bug (CRITICAL)
**When:** March 19
**What happened:** User asked if stochastic algorithms generate different layouts
across seeds. Investigation revealed 15 adapters hardcoded seed=42, meaning
10 "different seed" runs produced 10 IDENTICAL layouts. The multi-seed pipeline
was completely broken for most engines.
**Root cause:** CompetitorBase.layout() had no seed parameter. Each adapter
used its own hardcoded default.
**Fix:** Added seed parameter to all 30+ adapters, threaded through the
unified benchmark script.
**Time wasted:** An entire previous benchmark run (~10 hours) produced useless
data for stochastic engines.
**Lesson:** Verify the MECHANISM works before trusting the output. The pipeline
ran without errors but produced garbage data.

## Incident 7: ProcessPoolExecutor Hang
**When:** March 19, first unified benchmark run
**What happened:** run_benchmark.py with --workers 2 spawned workers that had
0 CPU time and did zero work. The script appeared to run (wrote 1152 "running"
records) but never completed any. TorchLens graph loading in forked worker
processes caused a deadlock.
**Fix:** Added serial execution path (--workers 1).
**Root cause:** TorchLens imports are not fork-safe. Each worker tried to
re-import TorchLens, which involves running PyTorch models.
**Time wasted:** ~30 minutes debugging + figuring out why workers were idle.
**Lesson:** Test the execution mode on a tiny dataset before launching the
full run.

## Incident 8: Reimplementation Fidelity Overstatement
**When:** March 19
**What happened:** I declared "All reimplementations are faithful. No red flags."
User dispatched an adversarial Codex that demolished this claim: Procrustes is
too weak a metric, comparison pairs were unfair (FA2 parameters mismatched,
tsNET was a proxy), averaging hid per-graph variation, seed propagation was
broken.
**Root cause:** I used one aggregate metric (Procrustes ratio) and interpreted
it too optimistically without rigorous validation.
**Time wasted:** The entire first benchmark's fidelity claims were invalid.
**Lesson:** Don't declare victory on vibes. Adversarial review before claiming
results.

## Incident 9: Karate Club Weighted Edges
**When:** March 20
**What happened:** After fixing reimplementations, FR/KK/Spectral still showed
0.18 Procrustes disparity vs NetworkX. Spent time investigating formula
differences. The actual cause: Karate Club graph has WEIGHTED edges (weights
1-7). NX uses weights; our code doesn't. On unweighted graphs: 0.000000.
**Time wasted:** ~1 hour trying to fix non-existent algorithm bugs.
**Lesson:** Verify test data properties. Don't assume a "standard" test graph
is simple.

## Incident 10: FA2 Init RNG Mismatch
**When:** March 20
**What happened:** After extensive formula fixes, FA2 still showed 0.74
disparity. The cause was trivial: reference uses `random.random()` (Python
stdlib), our code used `torch.rand()`. Same seed, different RNG, different
sequences. One-line fix brought it to 0.000002.
**Root cause:** Assumed `torch.rand(generator=seed)` produces the same
sequence as `random.random()` with the same seed.
**Time wasted:** Multiple Codex dispatches trying to fix formulas that were
already correct.
**Lesson:** When matching a reference, match the EXACT RNG source, not just
the seed value.

## Incident 11: Stress-SGD C RNG Barrier
**When:** March 20
**What happened:** Even with all formulas matching, stress-SGD showed 0.65
disparity vs s_gd2. Investigation revealed: s_gd2's C++ code uses a C-level
RNG (randomkit) for shuffling pairs. This RNG cannot be reproduced from Python.
Same init + different shuffle = 0.83 disparity. Proven by running s_gd2 with
identical init positions but different shuffle seeds.
**Resolution:** Accepted as fundamental barrier. Compared stress values instead
(0.993 ratio — identical objective quality).
**Lesson:** For C-extension references, position-level exact match may be
impossible. Compare objectives, not coordinates.

## Incident 12: OGDF Python Bindings Failure
**When:** March 19-20
**What happened:** Installed ogdf-python, built OGDF from source, but the Python
bindings (cppyy) couldn't parse OGDF's C++20 headers (concepts, forward_iterator).
Spent time trying to make it work.
**Fix:** Built a C++ subprocess wrapper (ogdf_runner.cpp) instead.
**Root cause:** ogdf-python relies on cppyy's JIT compiler which doesn't support
C++20. The right approach was always subprocess — same pattern as dagre/elk.
**Time wasted:** ~30 minutes on cppyy debugging.
**Lesson:** When a Python binding fails, go directly to subprocess. Don't debug
binding infrastructure.

## Incident 13: Subprocess Was Always the Answer
**When:** March 20
**What happened:** User pointed out (frustrated) that subprocess was always an
option for OGDF and I should have done it from the start. "We are not
communicating properly!" They were right — I had a blind spot about subprocess
being "too much work" when we literally already use it for Graphviz, ELK, and
dagre.
**Root cause:** Mental model failure. I categorized OGDF as "needs Python
bindings" instead of "needs any mechanism to run and return positions."
**Lesson:** For ANY external tool: can we run it? Can we feed it a graph? Can
we read positions back? If yes to all three, do it. The mechanism (Python import
vs subprocess vs file I/O) is irrelevant.

## Incident 14: "Match the Paper" vs "Match the Code"
**When:** March 20
**What happened:** Multiple Codex dispatches tried to fix reimplementations
based on paper descriptions. The adversarial review showed papers are ambiguous
and our implementations had incidental differences. User said: "the code should
be ground truth over the paper."
**Resolution:** Shifted to reading actual source code (NX Python, fa2 Python,
s_gd2 C++, OGDF C++) and translating line by line.
**Time wasted:** 2+ Codex dispatches based on paper descriptions that didn't
match the reference code's actual behavior.
**Lesson:** Always match the CODE, not the paper. Papers describe; code IS.
