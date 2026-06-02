# RNG-Matching Port -- Shared Protocol (read fully before porting)

You are making ONE dagua layout engine bit-identical to its reference, on SMALL graphs, at
MATCHED random seeds, by reimplementing the reference's RNG STREAM + arithmetic order from
the reference's SOURCE CODE. Project: /home/jtaylor/projects/dagua.

## The goal
- ACCEPTANCE: per-seed Procrustes RMSD < 1e-7 on the small-graph fixtures at seeds {1,2,3}.
- TARGET: drive to the float64 noise floor (~1e-13 to 1e-15) -- squeeze all the juice.
  Don't stop at 1e-7 if the path is float64 and can go lower.
- The reimplementation runs in the engine's `fidelity_mode` path in its pipeline file.

## How to verify (THE loop)
`python scripts/rng_match/check_engine.py <variant_id>` runs the matched-seed harness for one
variant and prints per-fixture RMSD + the max. Run it after every change. Your engine is done
when its max RMSD over all fixtures x seeds {1,2,3} is < 1e-7 (ideally ~1e-15).
The current baseline (your "before") is in `.project-context/research/sprint_rng_matching/STATUS.md`.

## How to match the RNG (the core technique)
1. READ the reference's actual source to find its RNG: which generator (drand48 / glibc rand /
   igraph RNG / numpy RandomState / Mersenne Twister), how it's seeded, and the EXACT sequence
   of draws (order of rand() calls relative to the algorithm: init positions, tie-breaks,
   permutations, etc.).
2. Reimplement that generator + draw-sequence in pure Python/PyTorch in the pipeline's fidelity
   path so dagua produces the SAME random stream for the SAME seed -- for ANY seed, not just
   one. Match the arithmetic ORDER too (force accumulation, reductions) since float64 is
   order-sensitive.
3. For graphviz engines: use the instrumented build `~/tools/graphviz-7.0.5-instr/bin/dot`
   with `GV_TRACE=1 GV_TRACE_FILE=/path` to dump graphviz's internal per-iteration positions
   (%.17g) and RNG-driven values. Compare dagua's trace step-by-step to find the FIRST diverging
   step, fix it, repeat. (graphviz 7.0.5 source is at ~/tools/graphviz-7.0.5-src/.) The
   instrumented build is logging-only and proven identical to stock 7.0.5.

## ANTI-CHEAT (hard rules -- violation = work rejected)
1. NO RUNTIME DELEGATION: the pipeline must NOT import or call the reference package/binary
   (no `import igraph`, no subprocess to dot/neato/sfdp, no `from fa2util`, no calling
   dagua/eval/competitors/*). It computes its own positions. Common Python libs (numpy/scipy)
   are fine; calling the reference ALGORITHM is not.
2. NO RNG-REPLAY: you must reimplement the RNG GENERATOR from source so it produces the stream
   for ANY seed. Do NOT read/hardcode the reference's emitted random values from the trace.
   The trace is for finding WHERE you diverge, never for feeding values into dagua's output.
   (The multi-seed {1,2,3} check catches replay: a generator reimpl matches all 3 seeds; a
   replay matches only the captured one.)

## Scope discipline (avoid parallel-codex conflicts)
- Edit ONLY your engine's pipeline file(s) under dagua/layout/ops/. Do NOT edit
  dagua/eval/variants.py (other codexes touch it in parallel). If your variant needs a
  different fidelity selector to be set in variants.py, DON'T change variants.py -- instead
  REPORT the exact change needed in your output, and CC will apply it centrally.
- Do NOT touch other engines' pipeline files.
- Reference adapters in dagua/eval/competitors/ are read-only references (read their source to
  understand the reference, don't modify them).

## Records (mandatory)
In your final report state: (a) which RNG generator you reimplemented and from which source
file/lines, (b) the before -> after max RMSD from check_engine.py, (c) which fixtures/seeds (if
any) still exceed 1e-7 and your diagnosis, (d) any variants.py selector change CC must apply.
Do NOT commit (CC commits after verifying + grepping your diff for delegation).

## Stop conditions
- Done: max RMSD < 1e-7 on all fixtures x seeds {1,2,3} (report the actual floor reached).
- Stuck: after genuine effort + trace analysis you cannot get below 1e-7 -- STOP and document
  the precise irreducible reason (e.g. reference uses a non-reproducible source, an unported
  C-library primitive, genuine chaotic divergence even on small graphs). Don't flail; don't fake.
