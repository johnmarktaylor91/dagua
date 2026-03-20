# Operational Principles: Competitor Algorithm Pipeline Retro (v2)

Revised after adversarial review by Claude and Codex critics.

## P1: VERIFY MECHANISM BEFORE TRUSTING OUTPUT
Before any multi-run benchmark: run 2 seeds, verify outputs DIFFER meaningfully
(not just floating-point noise). Check variance is consistent with algorithm's
expected stochasticity. Don't just check `not allclose` — inspect the actual
difference magnitude.
- **Trigger:** Dispatching any benchmark with stochastic engines or multiple seeds
- **Prevents:** Incident 6 (hardcoded seeds = 10 identical runs)
- **Enforcement:** Add a pre-flight check to the benchmark script itself

## P2: ADVERSARIAL REVIEW BEFORE ANY POSITIVE ASSESSMENT
Never present positive results (fidelity, quality, completeness) to the user
without first dispatching an adversarial agent to challenge them. This applies
to informal summaries too, not just formal "claims."
- **Trigger:** ANY positive statement about benchmark results or algorithm quality
- **Prevents:** Incident 8 (premature "all faithful" claim demolished by review)
- **Enforcement:** Habit — treat "looks good" as a draft, not a conclusion

## P3: MATCH THE INSTALLED CODE, NOT THE PAPER
Read the reference source code AT THE EXACT VERSION installed in the environment.
Translate line by line. Cite `package==version` and file:line, not just the paper.
Papers describe; code IS. Version mismatches between installed code and GitHub
HEAD cause real differences.
- **Trigger:** Writing or fixing any reimplementation
- **Prevents:** Incident 14 (paper-based specs that didn't match reference code)
- **Enforcement:** `pip show <package>` before reading source. Pin version in spec.

## P4: MATCH ALL SOURCES OF NONDETERMINISM
"Same seed" means nothing across different RNG engines. When matching a reference:
identify the EXACT RNG (torch, numpy, Python random, C randomkit), use the same
one, AND audit non-RNG nondeterminism (hash ordering, unstable sorts, GPU kernels).
- **Trigger:** Matching any stochastic reference implementation
- **Prevents:** Incident 10 (FA2 wrong RNG: torch.rand vs random.random)
- **Enforcement:** Add "RNG source" to reimplementation spec template

## P5: CAN WE RUN IT AND GET POSITIONS BACK?
For ANY external algorithm: can we execute it (any mechanism: import, subprocess,
IPC, file)? Can we feed it a graph? Can we read positions from the output?
If yes to all three, build the adapter. Python bindings are not special —
subprocess works for every compiled tool.
- **Trigger:** Evaluating whether a reference implementation is "available"
- **Prevents:** Incidents 12-13 (OGDF blocked by cppyy when subprocess worked)
- **Enforcement:** Default to subprocess for any non-Python tool

## P6: SMOKE TEST WITH PRODUCTION CONFIG
Before any multi-hour dispatch: run on 2-3 small graphs with the EXACT same
flags/workers/seeds you plan to use. Verify: (a) completes, (b) outputs exist
and are valid, (c) different seeds differ, (d) workers actually do work.
- **Trigger:** Any benchmark or long-running dispatch
- **Prevents:** Incidents 6, 7 (silent failures that waste hours)
- **Enforcement:** Add a `--smoke` flag that runs 3 graphs then exits

## P7: SHOW DISTRIBUTIONS, NOT JUST AGGREGATES
Always report per-graph, per-algorithm breakdowns. Never present only a mean.
Include input properties (node count, edge count) alongside outputs so readers
can spot when comparisons mix different scales.
- **Trigger:** Reporting any benchmark results
- **Prevents:** Incidents 1, 8 (misleading averages)
- **Enforcement:** Report format template includes per-graph table

## P8: INSPECT TEST DATA BEFORE COMPARING
Before using any graph for fidelity comparison: print and verify its properties
(weighted? directed? connected? self-loops? multi-edges? number of components?).
Don't assume standard benchmark graphs are simple.
- **Trigger:** Selecting test graphs for algorithm comparison
- **Prevents:** Incident 9 (karate club weighted edges wasted 1 hour)
- **Enforcement:** Helper function that prints graph properties

## P9: AUDIT ALL ADAPTER CONFIGURATIONS
When adding or modifying competitor adapters: audit EVERY configuration choice
(device, timeout, iterations, graph model, seed handling). Don't just check the
ones that error — check the ones that silently misconfigure.
- **Trigger:** Building or reviewing competitor adapters
- **Prevents:** Incident 2 (dagua hardcoded CPU = underreported performance)
- **Enforcement:** Adapter review checklist in AGENTS.md

## P10: C EXTENSIONS HAVE RNG BARRIERS
C/C++ extensions use internal RNG that can't be reproduced from Python.
For these references: compare OBJECTIVE VALUES (stress, energy, crossing count)
not positions. Document this explicitly — position-level exact match is impossible.
- **Trigger:** Comparing against C-extension references (s_gd2, igraph C, OGDF)
- **Prevents:** Incident 11 (chasing impossible s_gd2 position match)
- **Enforcement:** Comparison script auto-selects metric based on reference type

## P11: CONSOLIDATE BEFORE RUNNING
Before launching multiple scripts that do similar things: ask "can this be one
script?" If the same data could be produced by one run, merge first.
- **Trigger:** About to launch 2+ related scripts
- **Prevents:** Incident 5 (three benchmark scripts doing the same job)
- **Enforcement:** Habit — ask "is there already a script for this?"

## P12: UNDERSTAND EVERY FLAG
Before running any CLI command: verify what each flag does by reading --help
or the argparse source. --no-X doesn't always mean what you think.
- **Trigger:** Running CLI commands with non-trivial options
- **Prevents:** Incident 3 (--no-resume caused full re-run)
- **Enforcement:** Quick --help check

## META-PRINCIPLES (from critic feedback)

### M1: THESE PRINCIPLES ARE ADVICE, NOT ENFORCEMENT
Most principles depend on Claude remembering them at the right moment.
The ones that matter most should be AUTOMATED (built into the benchmark
script, the adapter template, the spec template) not just written down.

### M2: THE RECURRING PATTERN IS "TRUST THEN VERIFY LATER"
Incidents 6, 7, 8, 10 all follow the same pattern: run something, trust the
output, discover later it was wrong. The fix is: verify BEFORE trusting, not
after discovering problems. Build verification into the pipeline itself.

### M3: ADAPTER BUGS ARE A CLASS, NOT INDIVIDUAL INCIDENTS
Incidents 2, 6, 10 are all "adapter configured wrong." A single adapter audit
checklist (device, seed, timeout, graph model, output format, scaling) applied
to EVERY adapter would have caught all three.
