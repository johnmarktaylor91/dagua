# Adversarial Critique: Competitor Pipeline Retro Principles

## 1. Incident-to-Principle Coverage Matrix

| Incident | Covered By | Actually Prevented? | Notes |
|----------|-----------|---------------------|-------|
| 1: Graphviz timeout / apples-oranges aggregates | P7 | Partially | P7 says "show breakdowns" but doesn't say HOW to detect that inputs differ (graph sizes). A per-graph table still hides scale mismatches unless you compare input sizes too. |
| 2: dagua_competitor hardcoded CPU | **NONE** | No | No principle addresses auditing adapter configurations. P6 smoke test would not catch this — the adapter runs fine on CPU, just slowly. |
| 3: --no-resume waste | P9 | Yes | Straightforward. |
| 4: OOM from other session | **NONE** | No | No principle covers cross-session resource contention. Checkpointing saved us, but no principle says "verify other processes aren't consuming resources." |
| 5: Three redundant scripts | **NONE** | No | No principle addresses script consolidation or "verify you're not duplicating existing infrastructure before writing." |
| 6: Seed handling bug | P1, P6 | Yes, if followed | P1 is well-targeted. P6 reinforces it. |
| 7: ProcessPoolExecutor hang | P6 | Partially | P6 says "smoke test" but doesn't specifically address fork safety. A smoke test with --workers 1 would pass. You'd need to smoke test WITH the parallelism config you plan to use. |
| 8: Reimplementation fidelity overstatement | P2 | Yes, if followed | Well-targeted. |
| 9: Karate Club weighted edges | P8 | Yes | Well-targeted. |
| 10: FA2 init RNG mismatch | P4 | Yes | Well-targeted. |
| 11: Stress-SGD C RNG barrier | P10 | Yes | Well-targeted. |
| 12: OGDF Python bindings failure | P5 | Yes | Well-targeted. |
| 13: Subprocess was the answer | P5 | Yes | Duplicate of 12. |
| 14: Match paper vs code | P3 | Yes | Well-targeted. |

**Uncovered incidents: 3 out of 14 (Incidents 2, 4, 5).**

Incident 2 is particularly damning — a systematic performance misreport from a
hardcoded device, and NO principle addresses adapter configuration auditing. This
is the same class of bug as Incident 6 (hardcoded value in adapter), yet only the
seed case got a principle.

---

## 2. Ten Specific Ways the Principles Are Insufficient

### W1: P1 (Verify Outputs Differ) only catches identical outputs, not correlated ones

**Failure scenario:** An adapter uses `seed % 3` internally, so seeds 0,3,6,9
produce run A and seeds 1,4,7,10 produce run B. `assert not allclose(run1, run2)`
passes for seeds 0 and 1, but you only have 2 truly distinct layouts across 10
seeds. The principle is satisfied. The data is still mostly garbage.

**Fix needed:** "Verify outputs differ AND that variance across N seeds is
consistent with expectations for the algorithm's stochasticity."

### W2: P2 (Adversarial Before Claiming) has no enforcement mechanism

**Failure scenario:** You're at hour 14 of a benchmark session. Results look
great. The user is excited. You say "results look solid" in a summary message
before formally "claiming." The user treats this as a claim. P2 technically
wasn't violated because you didn't use the word "faithful," but the practical
effect is identical.

**Fix needed:** Any positive assessment of results triggers adversarial review,
not just formal "claims." The trigger condition is too narrow.

### W3: P3 (Match the Code) doesn't address WHICH version of the code

**Failure scenario:** You read the NetworkX source for `spring_layout` at HEAD
on GitHub. The user's installed version is networkx==3.1, which has different
default parameters. Your reimplementation perfectly matches the wrong version.
Procrustes disparity is nonzero. You waste hours debugging code that's correct
against the wrong reference.

**Fix needed:** "Match the code AT THE EXACT VERSION installed in the test
environment." Pin it. Cite the version, not just the file:line.

### W4: P4 (Match the Exact RNG) doesn't cover non-RNG sources of nondeterminism

**Failure scenario:** You match the exact RNG. But the reference uses
`dict.items()` iteration order (Python 3.7+ insertion order, but varies across
runs if dict was built from set operations), or uses `torch.mm` with
nondeterministic CUDA kernels, or sorts equal-key elements with unstable sort.
Same seed, same RNG, different output.

**Fix needed:** "Match ALL sources of nondeterminism, not just RNG. Audit hash
ordering, parallel reduction order, unstable sorts, and GPU kernel selection."

### W5: P5 (Can We Run It?) doesn't cover "should we run it ourselves at all?"

**Failure scenario:** You spend 2 days building a subprocess wrapper for an
obscure layout engine. It works. The engine produces terrible layouts on every
test graph. The entire effort was wasted because nobody checked whether the
engine was even competitive before investing in integration.

**Fix needed:** "Before investing in integration: run the tool manually on 2-3
test graphs, inspect outputs visually, and confirm it's worth integrating."

### W6: P6 (Smoke Test Before Full Run) doesn't specify testing with production configuration

**Failure scenario:** You smoke test with `--workers 1 --seeds 2 --graphs 2`.
It passes. You launch the full run with `--workers 4 --seeds 10 --graphs 50`.
The fork-safety bug (Incident 7) only manifests with `--workers > 1`. The smoke
test was green because it didn't test the configuration that mattered.

**Fix needed:** "Smoke test with the EXACT configuration you plan to use for
the full run, but on a tiny dataset. Do not simplify the config — simplify
the data."

### W7: P7 (Never Report Only Aggregates) doesn't say what to DO with the breakdown

**Failure scenario:** You dutifully produce a per-graph table. It has 50 rows.
The one bad row (graph_47 has 10x the crossings of everything else) is buried
on page 2. You skim the table, see mostly green, and report "results look good
with per-graph breakdown attached." The outlier is right there in the data and
still gets missed.

**Fix needed:** "Flag outliers automatically. Any result >2 sigma from the mean,
or any single-graph metric worse than the worst competitor, must be explicitly
called out."

### W8: P8 (Know Your Test Data) is reactive, not systematic

**Failure scenario:** You print graph properties for Karate Club because you
got burned there. Next week you use a different graph (say, Dolphin social
network) and it has self-loops. You don't check because P8 is a mental
reminder, not an automated check.

**Fix needed:** "Every graph used for fidelity comparison MUST pass through a
`validate_test_graph()` function that prints properties and asserts no
surprising features (weights, self-loops, multi-edges, disconnected components)
unless explicitly expected."

### W9: P9 (Understand Flags Before Running) is unenforceable by Claude

**Failure scenario:** Claude reads `--help` for a CLI tool. The help text says
`--no-resume: Start fresh`. Claude interprets "start fresh" as "don't try to
load a checkpoint" when it actually means "delete all previous results and
restart." The help text was ambiguous. Claude followed P9 and still got it wrong.

**Fix needed:** "For destructive flags (anything that deletes, overwrites, or
restarts), read the implementation, not just the help text."

### W10: P10 (C Extensions Have RNG Barriers) is too narrow — applies to any opaque execution boundary

**Failure scenario:** You're comparing against a Java-based layout engine (ELK)
via subprocess. ELK uses Java's RNG internally. P10 only mentions "C extensions"
so you don't think it applies. You spend hours trying to achieve position-level
match against ELK when the same fundamental barrier exists.

**Fix needed:** Generalize to: "ANY external engine with its own RNG (C, C++,
Java, Rust, subprocess) has an RNG barrier. Compare objectives, not positions,
for all non-Python references."

---

## 3. Will Claude Actually Follow These?

### 3a. The principles file is orphaned

The file lives at `.project-context/knowledge/retro_20260320_principles.md`.
It is NOT referenced from:
- `CLAUDE.md` (project instructions, always loaded)
- `AGENTS.md` (worker instructions, always loaded)
- `.project-context/knowledge/gotchas.md` (sometimes loaded)
- `MEMORY.md` (always loaded)

**This means no Claude session will ever read this file unless explicitly told
to.** A principle that Claude doesn't read during the relevant task is a
principle that doesn't exist. This is the single most critical failure of the
entire retro document.

### 3b. The principles are task-type-specific but not wired to task triggers

P1-P10 are relevant specifically when:
- Running benchmarks (P1, P6, P7, P9)
- Comparing against references (P3, P4, P8, P10)
- Claiming fidelity results (P2)
- Integrating external tools (P5)

But nothing in the dispatch workflow says "before running a benchmark, read
retro_20260320_principles.md." The scaling_principles.md had the same problem
until its key rules were condensed into AGENTS.md's "Scale Work Rules" section.
These principles need the same treatment or they're dead letters.

### 3c. Ten principles is too many for reliable recall

Even if Claude reads the file at session start, by the time it's writing a
benchmark spec 2000 tokens later, it will have deprioritized most of them.
Claude's instruction-following degrades with distance from the instruction.
The scaling principles work because they're in AGENTS.md (always present) AND
they're structured as a reference table with clear triggers.

**Recommendation:** Condense to 5 hard rules (the ones that would have saved
the most time) and embed them in AGENTS.md or CLAUDE.md where they'll actually
be read. The rest can stay in the knowledge file as a reference, but the
critical ones must be in the always-loaded files.

### 3d. Principles phrased as observations won't trigger action

P10: "C extensions use their own RNG..." is an observation. It doesn't tell
Claude WHAT TO DO when it encounters the situation. Compare to the scaling
principles which say "MUST," "abort if," "never." The retro principles are
too politely phrased to override Claude's default behavior of trying to
make things match.

---

## 4. Meta-Problems With the Debugging Process

### M1: The same class of bug recurs with different surface symptoms

Incidents 2 and 6 are the same bug: adapter hardcodes a value instead of
accepting a parameter. In #2 it's `device="cpu"`. In #6 it's `seed=42`.
The principle (P1) only covers the seed case. A general principle — "audit
ALL adapter parameters that should be configurable" — would prevent both AND
the next variant (hardcoded iteration count, hardcoded tolerance, hardcoded
graph size limit, etc.).

### M2: The architecture doesn't enforce contracts at the interface level

CompetitorBase.layout() had no seed parameter. That's not a "gotcha" — that's
a missing interface contract. The adapter base class should have declared
every parameter that the benchmark pipeline can pass, with abstract methods
requiring implementation. Python ABCs with `@abstractmethod` would have
caught Incident 6 at import time, not after 10 hours.

### M3: Verification happens at the end, not inline

The pattern is: run everything, then check if results make sense. This is
backwards for expensive pipelines. Verification should happen per-step:
- After each adapter run: did it produce different output from the last seed?
- After each graph: are the metrics in sane ranges?
- After each engine: is the timing plausible?

Inline assertions would have caught Incidents 1, 2, 6, 7, and 8 within
minutes instead of hours.

### M4: No distinction between "it ran" and "it worked"

This is the deepest pattern. Incidents 2, 6, 7, and 8 all involve code that
RUNS without error but produces wrong results. The pipeline has no concept
of correctness, only completion. The principles (P1, P6) try to address this
with spot checks, but the real fix is to build correctness assertions into
the pipeline itself:
- Post-condition: positions are finite and within expected bounds
- Post-condition: different seeds produce different results for stochastic engines
- Post-condition: timing is within 10x of expected range for graph size
- Post-condition: metric values are in sane ranges (no negative edge lengths, etc.)

### M5: Codex dispatches are treated as atomic when they're actually hypothesis tests

Incidents 10 and 14 both involve multiple Codex dispatches that each "fix"
something without fixing the root cause. The workflow is: observe symptom,
hypothesize cause, dispatch fix, observe same or different symptom, repeat.
Each dispatch costs quota and time but doesn't compound learning because the
hypothesis wasn't validated before the fix was attempted.

**Structural fix:** Before dispatching a fix, run a DIAGNOSTIC dispatch first.
"Read these 3 files, compare line by line against these references, and report
every difference." THEN dispatch the fix with a precise list of changes. This
halves the dispatch count.

---

## 5. Would These Principles DEFINITELY Change Behavior Tomorrow?

### Verdict: NO for 4 of 10 principles, PROBABLY for 4, YES for 2.

| Principle | Tomorrow Test | Verdict |
|-----------|--------------|---------|
| P1: Verify outputs differ | Would Claude actually run this check before trusting a benchmark? Only if reminded. The principle isn't in any always-loaded file. | **NO** — file won't be read |
| P2: Adversarial before claiming | Would Claude dispatch an adversary before saying "looks good"? No — there's no mechanical trigger, and Claude defaults to optimism. | **NO** — too easy to skip |
| P3: Match the code | Would Claude read source instead of papers? Probably, if prompted. But without version pinning, it might read the wrong source. | **PROBABLY** |
| P4: Match the exact RNG | Would Claude check which RNG the reference uses? Yes, this is specific enough to remember. | **YES** |
| P5: Can we run it? | Would Claude try subprocess before giving up on bindings? Probably, if it remembers. But the principle isn't in the dispatch path. | **PROBABLY** |
| P6: Smoke test | Would Claude smoke test before a full run? Sometimes. But it wouldn't necessarily test with the production config (W6). | **PROBABLY** |
| P7: Never report only aggregates | Would Claude show per-graph tables? Probably. Would it flag outliers? No — the principle doesn't say to. | **PROBABLY** |
| P8: Know your test data | Would Claude print graph properties? Only if it remembers. No automated guard. | **NO** |
| P9: Understand flags | Would Claude read --help? Usually does anyway. Low signal principle. | **YES** (but adds little) |
| P10: C extension RNG | Would Claude remember this for C extensions? Maybe. For Java/Rust? No — too narrow. | **NO** for the general case |

### The "definitely" bar

For a principle to DEFINITELY change behavior, it needs ALL of:
1. To be in a file Claude reads during the relevant task (CLAUDE.md, AGENTS.md, or MEMORY.md)
2. To specify a concrete, mechanical action (not a mental reminder)
3. To have a clear trigger condition that matches the task type
4. To be short enough to not get lost in a wall of text

Only P4 and P9 come close, and P9 is largely redundant with Claude's existing
behavior.

---

## 6. Recommendations

### R1: Extract the 3 highest-value rules into AGENTS.md

Add a "Benchmark Rules" section to AGENTS.md, parallel to "Scale Work Rules":

```
## Benchmark & Comparison Rules (mandatory for eval/ work)
- Before any multi-seed run: verify 2 seeds produce different outputs
- Before claiming fidelity: dispatch adversarial review agent
- Before integrating an external tool: try subprocess first, bindings second
- When matching a reference: read the EXACT source code at the installed version,
  match the exact RNG engine, not just the seed value
- When reporting results: show per-graph breakdown, flag any outlier >2 sigma
```

This is 5 lines. It fits. It will be read.

### R2: Add inline assertions to the benchmark pipeline

Not principles — CODE. The pipeline itself should assert:
- Different seeds produce different positions
- Positions are finite and bounded
- Timing is within expected range
- No adapter parameter is hardcoded when it should be configurable

### R3: Add a `validate_adapter()` function to CompetitorBase

At import time, verify every adapter accepts and uses seed, device, and
timeout parameters. This is an interface contract, not a principle.

### R4: Create a pre-benchmark checklist command

`dagua benchmark-preflight` that:
- Runs 2 graphs x 2 seeds x all adapters
- Asserts outputs differ for stochastic engines
- Prints timing per adapter
- Prints graph properties for all test graphs
- Reports any adapter errors

This replaces P1, P6, P7, P8 with a single automated check.

### R5: Write a MEMORY.md entry pointing to these principles

At minimum, add to MEMORY.md's feedback section:
```
- **[retro_20260320_principles.md]** — Benchmark pipeline operating rules.
  Read before any eval/ work or benchmark dispatch.
```

Without this, the file is invisible.

---

## 7. Summary Judgment

The principles accurately diagnose the bugs they came from. They fail as a
prevention system for three reasons:

1. **Location failure.** They're in a file nothing references. Claude won't
   read them when it matters.

2. **Generality failure.** They're fitted to exact incidents instead of
   extracted to bug classes. The next variant (hardcoded iteration count
   instead of hardcoded seed) slips through.

3. **Enforcement failure.** They're mental reminders, not mechanical checks.
   Mental reminders degrade with context window distance and competing
   priorities. The highest-leverage fix is to encode the checks in code
   (pipeline assertions, adapter validation, preflight commands), not in
   markdown files.

The principles are a decent incident log. They are not yet a prevention
system. To become one, the critical rules need to move into always-loaded
instruction files, the bug classes need to be generalized, and the most
important checks need to become code, not prose.
