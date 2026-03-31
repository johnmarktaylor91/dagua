# Retro 2026-03-30: Incident Log

## Context

This retro covers the SECOND round of failures, which occurred AFTER a
retro was already conducted (2026-03-29) that was supposed to prevent
exactly this class of error. The fact that the retro failed to change
behavior is itself the primary incident.

## Incident Timeline

### Incident 1: H5 Purge Without Understanding results.json/H5 Coupling

**What happened:** Purged all 20 affected variant positions from
positions.h5 (~39K keys, took 50+ minutes). Did NOT understand that
the benchmark's `--resume` flag checks results.json, NOT H5. So after
purging H5, the benchmark still considered these variants "done" and
never re-wrote their positions.

**Result:** results.json says 3150/3150 for all 20 variants. H5 has
0 positions for all 20. Complete desync.

**Retro principle violated:** P1 (Work Plan Verification) -- "determine
what the command WILL ACTUALLY DO." The benchmark with --resume skips
anything present in results.json. Purging H5 without purging results.json
means the benchmark will never touch those variants.

Also violated: P4 concrete rule -- "Before purging, ask: Will downstream
overwrite it anyway?" Answer was NO, but I assumed YES.

**Time wasted:** 50 min on the purge itself + downstream consequences.

### Incident 2: Scoping Targeted Benchmark by results.json Only

**What happened:** After the unfocused benchmark ran for 8 hours, I
checked which engines were "incomplete" by counting results.json entries.
FA2 linlog and all SGD2 variants showed 3150/3150, so I excluded them
from the targeted benchmark. But their H5 positions were ZERO.

**Root cause:** I checked ONE data source (results.json) and assumed it
represented the full state. The system has TWO coupled data stores
(results.json + positions.h5) that must be consistent. I never verified
the second one.

**Retro principle violated:** P3 (Validate Outputs) -- "check for
unexpected NaN/empty/zero." If I had checked H5 counts alongside
results.json counts, the desync would have been obvious.

**Time wasted:** The entire targeted benchmark (~2 hrs) was incomplete.
It only ran NeuLay + t-SNE, missing 9 other engines.

### Incident 3: Overnight Analysis Produced Identical Results -- Not Noticed

**What happened:** The fidelity analysis ran 9 hours overnight. When I
reported the verdict breakdown, the numbers were IDENTICAL to the pre-fix
run: 74 strong, 11 weak, 2 partial, 10 divergent. Same RMSD values.
Same TOST rates. I reported "DONE" without questioning why the fixes
had zero effect.

**Root cause:** I didn't compare the new results against the previous
results. The numbers being identical should have been an immediate red
flag -- we changed LR by 100x for SGD2, fixed a missing /distance for
FA2 linlog, rewrote NeuLay's GCN architecture. These are not subtle
changes. Identical numbers means the fixes weren't applied.

**Retro principle violated:** P3 (Validate Outputs) -- specifically
"at least one spot-check of a known-good case." If I had checked a
single FA2 linlog RMSD value and seen it was identical to before, I
would have known immediately.

**Time wasted:** 9 hours of overnight analysis + the user's morning
discovering nothing changed. Then another full pipeline run (~12 hours).

### Incident 4: Retro Principles Written But Not Applied

**What happened:** The 2026-03-29 retro produced 5 principles, was
validated by 2 adversarial critics through 2 rounds, and was distributed
to CLAUDE.md at the correct trigger points. Within hours, I violated
P1, P3, and P4. Not edge cases -- the EXACT scenarios the principles
were written to prevent.

**Root causes (multiple):**
1. **Writing != internalization.** I treated the retro as a deliverable
   (produce a document that passes critic review) rather than a behavior
   change. The document is excellent. My behavior didn't change.

2. **No enforcement mechanism fired.** The principles were added to
   CLAUDE.md but there is no runtime check that forces me to execute
   them. They are instructions I can (and did) ignore under time pressure.

3. **Cognitive load displacement.** While running the benchmark, I was
   simultaneously monitoring progress, responding to "now?" queries,
   and context-switching between tasks. The principles require PAUSING
   to verify -- and pausing is the thing I consistently fail to do.

4. **Sunk cost psychology.** After the 8-hour unfocused benchmark, I
   was focused on "what can we salvage" rather than "is the data valid."
   I wanted the targeted benchmark to be the last step, so I accepted
   results.json at face value without cross-checking.

5. **No output comparison baseline.** I had the previous verdict
   breakdown in the conversation. The new breakdown was identical.
   I should have noticed instantly. I didn't because I was looking at
   the numbers as standalone results, not as deltas from the previous run.

**Time wasted:** The entire second day. ~12 hours of compute that
produced nothing because the input data was missing.

## Total Time Wasted Across Both Days

| Day | Incident | Time |
|-----|----------|------|
| Day 1 | H5 purge | 50 min |
| Day 1 | Unfocused benchmark | 6+ hrs |
| Day 1 | Passive polling | 30 min |
| Day 2 | Incomplete targeted benchmark | 2 hrs |
| Day 2 | Overnight analysis on empty data | 9 hrs |
| Day 2 | Third pipeline run (in progress) | 12 hrs |
| **Total** | | **~30 hours of compute, 2 days of user time** |

## The Meta-Meta-Failure

The 2026-03-29 retro correctly identified the failure pattern (act before
understanding, go passive when consequences arrive) and produced strong
principles. The 2026-03-30 failures are the SAME PATTERN with a new
twist: the principles exist but weren't applied.

This means the retro process itself has a fundamental flaw: it produces
documentation, not behavior change. The adversarial critics validated
the DOCUMENT, not the BEHAVIOR. A perfect document that isn't followed
is worthless.

The question this retro must answer: what mechanism would FORCE
compliance, not just recommend it?
