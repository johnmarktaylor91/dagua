# Codex Critique of `retro_20260320_principles.md`

## Bottom Line

These principles are mostly post-hoc reminders, not behavior-changing controls.
On the standard "if the exact same situation happened tomorrow, would Claude
DEFINITELY behave differently?", the answer is mostly no.

Why: nearly every principle depends on Claude remembering the rule at the right
moment, interpreting the trigger correctly, and voluntarily interrupting its own
momentum. That is not a reliable process. It is advice. Advice does not beat
time pressure, tunnel vision, or ambiguous ownership.

The matrix below makes the main problem obvious: the principles are narrowly
derived from the incidents, but they do not form a coherent operating system.
Several incidents still have no strong prevention coverage at all.

## Full Incident x Principle Cross-Reference

Legend:

- `S` = strong prevention if enforced before action
- `P` = partial mitigation or post-hoc detection only
- `N` = no meaningful prevention
- `R` = risks pushing the wrong behavior

| Principle | I1 | I2 | I3 | I4 | I5 | I6 | I7 | I8 | I9 | I10 | I11 | I12 | I13 | I14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 Verify outputs differ | N | N | N | N | N | S | N | P | N | N | N | N | N | N |
| P2 Adversarial before claiming | P | P | N | N | N | P | N | S | P | P | P | P | P | P |
| P3 Match the code, not the paper | N | N | N | N | N | N | N | P | P | S | P | N | N | S |
| P4 Match the exact RNG | N | N | N | N | N | N | N | P | N | S | R | N | N | P |
| P5 Can we run it? | N | N | N | N | N | N | N | N | N | N | N | S | S | N |
| P6 Smoke test before full run | N | N | N | N | N | S | S | P | N | P | P | N | N | P |
| P7 Never report only aggregates | P | P | N | N | N | N | N | P | P | N | N | N | N | N |
| P8 Know your test data | N | N | N | N | N | N | N | P | S | N | N | N | N | N |
| P9 Understand flags before running | N | N | S | N | N | N | N | N | N | N | N | N | N | N |
| P10 C extensions have RNG barriers | N | N | N | N | N | N | N | P | N | N | S | N | N | P |

## What The Matrix Proves

- Only a small fraction of the 140 incident-principle intersections are strong.
- Incident 2, Incident 4, and Incident 5 have zero strong protection.
- Incident 1 has no strong protection because the principle set addresses the
  misleading aggregate, but not the hidden Graphviz timeout contract failure.
- Most principles strongly cover only one retrospective incident each. That is
  a hallmark of patching memory, not fixing process.
- Several principles are detection steps before reporting, not prevention steps
  before spending hours or making architectural choices.

## Principle-by-Principle Critique

### P1: VERIFY OUTPUTS DIFFER

1. Would this actually have prevented the target incident?
   Partially at best. If enforced before the long run, it would likely have
   exposed the hardcoded `seed=42` problem in Incident 6. As written, it is a
   manual reminder, so it would not DEFINITELY prevent the failure.
2. Concrete failure scenario:
   Two runs differ because of unrelated nondeterminism, floating-point noise, or
   thread scheduling, while the benchmark still ignores the user-provided seed.
   The check passes, but seed plumbing is still broken.
3. Will Claude actually read and follow this at the right moment?
   Not reliably. Claude often goes from "pipeline seems plausible" straight to
   "launch benchmark" unless the benchmark command itself forces a preflight.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Moderately specific, but too narrow. `assert not torch.allclose(...)` is
   concrete, but it does not define where the check lives, which engines it
   applies to, what tolerance is acceptable, or how to handle deterministic
   engines.

### P2: ADVERSARIAL BEFORE CLAIMING

1. Would this actually have prevented the target incident?
   It would likely have prevented the specific overclaim in Incident 8, but only
   before reporting, not before wasting compute or building false confidence.
   It is a communications gate, not a debugging process.
2. Concrete failure scenario:
   Claude requests adversarial review after a benchmark is already complete and
   already wrong. The claim is blocked, but the expensive run still happened and
   the core defects still slipped through the build/run workflow.
3. Will Claude actually read and follow this at the right moment?
   Unreliable. This depends on Claude noticing that it is "about to claim" and
   choosing to pause. That is exactly the kind of self-policing that fails under
   momentum.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Actionable in spirit, weak in mechanism. "Dispatch adversarial agent" is
   concrete, but there is no definition of required evidence, no stop-ship
   checklist, and no criteria for what "clearance" means.

### P3: MATCH THE CODE, NOT THE PAPER

1. Would this actually have prevented the target incident?
   Yes for Incident 14 if done early and literally. It also would have helped
   with Incident 10. But as prose, it still relies on the human or model to
   choose source review before speculative fixing.
2. Concrete failure scenario:
   Claude reads the reference code, but misses hidden defaults, preprocessing,
   data conventions, compiled branches, or call-order differences. "Line by
   line" translation still fails because the environment contract was not copied.
3. Will Claude actually read and follow this at the right moment?
   Sometimes, not definitely. When under pressure, Claude often starts from the
   paper or from intuition because that is faster than source archaeology.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Stronger than most. The `file:line` citation requirement is good. But it
   still lacks an enforcement point: no reference source means no merge, no run,
   or no fidelity claim.

### P4: MATCH THE EXACT RNG

1. Would this actually have prevented the target incident?
   Likely yes for Incident 10. It would not have prevented Incident 11 and could
   actively waste time there.
2. Concrete failure scenario:
   Claude matches the RNG family but not the call order, shuffle logic, or
   hidden random draws. The outputs still diverge, and the principle creates
   false confidence that "same RNG" should have solved it.
3. Will Claude actually read and follow this at the right moment?
   Only if Claude has already narrowed the problem to RNG. This principle does
   not help with the much harder step: realizing that RNG is the issue.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Specific, but overgeneralized. It is good advice for stochastic references,
   but it lacks boundaries and can mislead when exact RNG matching is impossible
   or irrelevant.

### P5: CAN WE RUN IT?

1. Would this actually have prevented the target incident?
   Yes for Incidents 12 and 13 if applied aggressively from the start. This is
   one of the better principles, but it still does not guarantee correctness,
   only availability.
2. Concrete failure scenario:
   Claude successfully builds a subprocess wrapper, but the I/O contract is
   wrong, positions are misparsed, or graph semantics are lost in translation.
   "We can run it" is not the same as "we can trust it."
3. Will Claude actually read and follow this at the right moment?
   Not definitely. Claude frequently overweights the most obvious integration
   path and treats subprocess as "extra work" unless the workflow defaults there.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Fairly actionable. The three checks are concrete. The weakness is that this
   principle addresses one decision point only and says nothing about validation
   after the external tool is reachable.

### P6: SMOKE TEST BEFORE FULL RUN

1. Would this actually have prevented the target incident?
   Probably for Incident 6 and likely for Incident 7, but only if the smoke test
   is mandatory, representative, and checked automatically. As written, it is
   still a reminder, not a gate.
2. Concrete failure scenario:
   The smoke test uses trivial graphs that do not exercise multiprocessing,
   checkpointing, memory pressure, or the specific adapter that fails in the
   full run. Everything passes; the real run still hangs or corrupts outputs.
3. Will Claude actually read and follow this at the right moment?
   Better odds than most principles, but still not definite. Claude often knows
   smoke tests are good and still skips them to save time.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Reasonably specific, but the chosen sample size is arbitrary and underspecified.
   "2 graphs with 2 seeds" sounds concrete while still being easy to satisfy with
   unrepresentative cases.

### P7: NEVER REPORT ONLY AGGREGATES

1. Would this actually have prevented the target incident?
   It would have reduced the misleading framing in Incident 1 and some of the
   damage in Incident 8, but it would not have found the Graphviz timeout bug,
   the fairness mismatch, or the broken seed mechanism by itself.
2. Concrete failure scenario:
   Claude includes a per-graph appendix but still leads with a misleading
   headline aggregate, or shows per-graph results without normalizing for graph
   eligibility, parameter parity, or failure reasons.
3. Will Claude actually read and follow this at the right moment?
   Not reliably. Report compression pressure pushes toward summary numbers, and
   without a required report template Claude will revert to aggregates.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Actionable, but incomplete. "Show breakdowns" is clear. It is not enough to
   define which breakdowns are mandatory or how to keep apples-to-apples cohorts.

### P8: KNOW YOUR TEST DATA

1. Would this actually have prevented the target incident?
   Yes for Incident 9 if done before comparison. This is a sound local rule.
2. Concrete failure scenario:
   Claude prints `weighted=True` and still compares weighted reference output to
   an unweighted implementation because there is no hard requirement to align the
   data contract after printing the metadata.
3. Will Claude actually read and follow this at the right moment?
   Not definitely. Data inspection is one of the first steps skipped when the
   graph is assumed to be "standard" or familiar.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Quite specific. The failure is not vagueness; the failure is that printing the
   metadata is optional and non-blocking.

### P9: UNDERSTAND FLAGS BEFORE RUNNING

1. Would this actually have prevented the target incident?
   Yes for Incident 3 if the flag semantics were actually checked first. This is
   the clearest direct lesson in the set.
2. Concrete failure scenario:
   Claude reads `--help`, but the real semantics are buried in code, changed by
   defaults, or interact with other flags in non-obvious ways. The command still
   does the wrong thing.
3. Will Claude actually read and follow this at the right moment?
   Not definitely. Under time pressure, command familiarity creates false
   confidence and Claude tends to execute first, inspect later.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Specific enough for manual use, but still missing an enforcement mechanism.
   A wrapper that echoes the resolved execution plan would do more than this
   sentence ever will.

### P10: C EXTENSIONS HAVE RNG BARRIERS

1. Would this actually have prevented the target incident?
   It likely would have prevented the wasted chase in Incident 11. But the rule
   is stated too absolutely to trust broadly.
2. Concrete failure scenario:
   Claude classifies a reference as "C extension" and prematurely gives up on
   exact matching even though the extension exposes deterministic seeding or the
   relevant randomness occurs entirely in Python-callable code.
3. Will Claude actually read and follow this at the right moment?
   Only if Claude has already identified the implementation boundary. That is
   not guaranteed, especially when the binding layer hides what is C and what is
   Python.
4. Is it specific enough to be actionable, or vague enough to ignore?
   Specific, but overbroad. This should be framed as a diagnostic branch with
   evidence requirements, not as a categorical law.

## Meta-Problems With The Debugging Process

These are not individual bugs. These are recurring failures in how debugging and
benchmark work is being run.

1. The process depends on memory instead of gates.
   Most principles require Claude to remember a lesson at the right time. If it
   is not encoded as a test, a wrapper, a required field, or a failing check, it
   will regress.

2. Validation happens after expensive work instead of before it.
   Multiple incidents involved hours of compute or repeated dispatches before
   basic assumptions were checked: seed propagation, worker liveness, graph
   properties, fidelity criteria, and flag semantics.

3. There is no canonical execution path.
   Three scripts for one job, adapter-specific defaults, and ad hoc execution
   modes created room for drift. A system with multiple "real" entry points is
   a system with no guaranteed behavior.

4. There is no explicit contract for adapters and benchmarks.
   Seed, timeout, device, output normalization, graph semantics, and eligibility
   rules were not enforced by a shared interface with conformance tests.

5. Reporting standards are weaker than run standards.
   Claims were allowed from aggregates, vibes, and incomplete fidelity checks.
   There was no evidence bundle required before saying "faithful" or "fixed."

6. Debugging starts from hypotheses instead of minimal reproductions.
   Repeatedly, the workflow jumped to "fix formulas" or "debug bindings" before
   reducing the problem to the smallest graph, seed, or integration path that
   could falsify assumptions cheaply.

7. Tooling visibility is too weak.
   The system did not loudly show device selection, effective timeout, resolved
   seeds, per-worker progress, graph properties, or apples-to-apples cohorts.
   Hidden state produced hidden failure.

8. Resource isolation is missing.
   An unrelated 83 GB job killed the benchmark. That is not "bad luck"; it is a
   structural failure to isolate long-running workloads.

9. Architecture tolerates historical sprawl.
   The existence of overlapping scripts and inconsistent wrappers means the team
   accepts accumulation first and consolidation after pain. That guarantees more
   incidents of the same class.

## Structural Changes That Would Prevent Categories Of Bugs

These are the kinds of changes that could make tomorrow's behavior DEFINITELY
different because they move from advice to enforced workflow.

1. Make preflight mandatory in the benchmark CLI.
   Every full benchmark should automatically run a fixed preflight suite before
   the real run starts. It should fail closed on:
   seed-insensitive stochastic adapters, hung workers, missing checkpoints,
   invalid device resolution, and empty output files.

2. Define a single `BenchmarkRunSpec` and validate it at startup.
   Required fields should include adapter name, timeout, device, seed policy,
   worker mode, graph cohort, objective metric, and eligibility filters.
   Print the resolved spec before execution. No hidden defaults.

3. Add adapter conformance tests.
   Every adapter should be forced through the same tests:
   seed propagation, timeout propagation, device propagation, output shape,
   deterministic/stochastic classification, and subprocess round-trip validity.
   Incident 2 and Incident 6 should have been impossible to ship.

4. Collapse to one canonical benchmark entry point.
   Keep one script as the truth and convert old scripts into thin wrappers or
   delete them. Multiple full-featured entry points guarantee divergence.

5. Replace principle-based claims with evidence-based claim gates.
   No "faithful" claim unless the report includes:
   per-graph metrics, parameter parity, source-code citations, seed behavior
   validation, and an explicit statement of what was compared: coordinates,
   objectives, or both.

6. Encode data auditing as part of graph loading.
   Graph selection should automatically emit and store node count, edge count,
   weightedness, directedness, connectivity, self-loops, multiedges, and any
   normalization/transformation applied before the algorithm sees the graph.

7. Add a reference-matching workflow for reimplementations.
   The workflow should require:
   reference source citation, deterministic fixture, parity checklist, and a
   signed-off explanation for any intentional deviation. No paper-only matching.

8. Build resource isolation into long benchmarks.
   Run benchmarks under explicit memory budgets, separate job groups, or
   container/cgroup limits so unrelated workloads cannot silently kill them.

9. Require minimal repro before broad fixes.
   If a mismatch appears, the next step should be a tiny, version-controlled
   reproduction case, not an open-ended debugging spree on the full pipeline.

10. Promote observability from optional logging to first-class state.
   The benchmark should surface effective seed, device, timeout, worker mode,
   current graph, checkpoint cadence, and per-adapter status as explicit output
   and persisted metadata.

## Final Verdict

The current principles are useful reminders, but they are not a reliable
operating model. They mostly say "be more careful next time." That is not a
serious prevention strategy for Claude or for humans.

If the exact same situations happened tomorrow, these principles alone would not
make Claude DEFINITELY behave differently. The only things that reach that bar
are structural controls: mandatory preflight, canonical entry points, adapter
conformance tests, run-spec validation, evidence-gated reporting, and resource
isolation.

Without those, this retro remains a memory aid, not a system.
