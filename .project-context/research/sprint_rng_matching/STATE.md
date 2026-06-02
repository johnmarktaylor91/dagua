---
run: sprint_rng_matching
created: 2026-06-02
state: SETUP
goal: RNG-stream-match dagua's fidelity ports to their references so SMALL graphs are
      bit-identical (per-seed Procrustes RMSD < 1e-7) at MATCHED seeds, for as many of the
      24 algorithms as physically possible. Stop at <1e-7-all OR documented can't-go-further.
      Then TEXT JMT to talk. Do NOT run the full all-graphs sweep (that's a later, separate
      spot-check JMT will direct).
---

# RNG-Stream-Matching Sprint -- State (single source of truth lives in STATUS.md)

## Why this sprint (JMT directive 2026-06-02)
Prior "bit-exact everything" was an over-claim: only ~6 families (fa2/tsnet/lgl/graphopt/
linlog/rt) were truly bit-exact; the graphviz/igraph/ogdf force layouts were only
TOST-equivalent and the "BIT-EXACT vs instrumented graphviz" results were ~1e-5 on 4 tiny
graphs vs a /tmp build that's now gone. Full reconciliation: ../sprint_algo_fidelity/WHERE_WE_STAND.md.
JMT wants a REAL attempt at RNG-stream matching, with ACCURATE RECORDS this time.

## Scoping decisions (CC, can be revised when JMT and CC meet)
1. REFERENCE = a single PINNED graphviz 7.0.5 build, rebuilt permanently + logging-only
   instrumented, PROVEN identical to stock 7.0.5 output (veridical). The benchmark/harness
   compares against THIS SAME binary (trace-build == reference-build) to kill build-level
   tie-break drift. Location: ~/tools/graphviz-7.0.5-instr/ (NOT /tmp). Build script committed.
2. Other refs matched via their seeded RNG read from source: igraph (its RNG), OGDF (srand/
   C++ RNG), fa2 (python), sklearn (numpy RandomState), networkx (numpy), umap-learn.
3. BAR (JMT refined 2026-06-02): SUCCESS/ACCEPTANCE = per-seed Procrustes RMSD < 1e-7 on
   SMALL graphs at MATCHED seeds {1,2,3}. But the TARGET is the float64 NOISE FLOOR
   (~1e-13 to 1e-15) -- "squeeze all the juice": do NOT stop a float64 engine at ~1e-7 if it
   can be driven lower; a perfect RNG+arithmetic-order match in float64 should reach ~1e-15.
   1e-7 is the floor only for genuinely float32 paths (float32 eps ~1.2e-7). Per-engine, push
   to its true noise floor and RECORD the actual magnitude achieved + dtype + reference output
   precision (so "stopped at the noise floor" is always backed by the number).
   Reference run at seed=N (graphviz: -Gseed=N -Gstart=N, adapter supports it; use the
   instrumented %.17g build for full-precision reference coords), dagua replicates seed=N's
   exact RNG generator.
4. SMALL graphs = curated tiny fixtures (path, star, cycle, grid, complete, small tree,
   small random/bipartite; ~6-20 nodes; ~12-15 fixtures). Small enough that chaotic FP
   cascade is negligible, so a clean RNG+arithmetic match should hit <1e-7.
5. STOP: all algos < 1e-7 OR documented irreducible reason per engine. No full-suite run.

## ANTI-CHEAT (enforced on every P1 port -- JMT explicit 2026-06-02)
Two forbidden cheats:
1. RUNTIME DELEGATION: no import/call of the reference package or binary from
   dagua/layout/ops/ (no `import igraph`, no subprocess to dot/neato/sfdp, no fa2util, no
   calling dagua/eval/competitors/*). Common Python libs (numpy/scipy) OK. See
   [[feedback_no_runtime_delegation_to_reference]].
2. RNG-REPLAY (subtle, sprint-specific): the port must reimplement the RNG GENERATOR
   (e.g. graphviz drand48, igraph RNG) from SOURCE so it produces the same stream for ANY
   seed -- it must NOT read/replay the reference's actual emitted random values from the
   instrumented trace. The instrumented build is DEBUG-ONLY (locate divergence), never feeds
   dagua's output.
ENFORCEMENT: (a) every port spec states both rules; (b) CC greps each port diff for
delegation signatures before accepting; (c) multi-seed {1,2,3} verification is the built-in
detector -- a generator reimpl matches all 3 seeds, a replay cheat matches only its captured
seed. STATUS.md records, per engine, WHICH generator was reimplemented from WHICH source.

## Phases
- P0 SETUP (parallel): (A) permanent instrumented graphviz 7.0.5 + prove logging-only/veridical
  + reproducible build script; (B) small-graph matched-seed bit-exact harness + fixtures +
  STATUS.md table (the single source of truth). CC verifies B on a known-bit-exact engine
  (fa2 -> ~0) and a known-divergent one (sfdp default -> high) before trusting it.
- P1 PORTS (parallel, many codex): per-engine RNG-stream match. Each codex: (1) read the
  reference's RNG + algorithm from source, (2) replicate exact RNG seq + arithmetic order in
  the dagua fidelity path, (3) verify <1e-7 via the harness on small graphs at seeds {1,2,3},
  (4) if stuck, use the instrumented build to trace the first diverging step. Update STATUS.md.
- P2 ITERATE: re-dispatch the engines not yet <1e-7 (one retry each, with trace evidence).
  Anti-flail: 3 rounds same engine no progress -> document irreducible reason, accept.
- DONE: STATUS.md shows every engine BIT_EXACT(<1e-7) or documented-stuck. Text JMT.

## Wake-up case routing
| Observable | Action |
|---|---|
| codex alive | yield, watcher will fire |
| codex DONE + STATUS.md engine now <1e-7 | mark done in STATUS; dispatch next engine or P2 |
| codex DONE but engine still >1e-7 | read trace; one targeted retry; else mark stuck |
| codex quota | (JMT upgraded -- should be fine) fall back to Opus Agent if hit |
| all engines BIT_EXACT or stuck | run DONE procedure |

## Records discipline (JMT: "keep accurate records so we don't get confused again")
- STATUS.md = THE per-engine table, updated after EVERY engine result. Columns:
  engine | reference | RNG source matched | best max-RMSD (small graphs, seeds 1-3) |
  verdict (BIT_EXACT<1e-7 / CLOSE / STUCK) | evidence file | notes.
- Every claim cites the harness output file. NO claim without a harness number. NO /tmp.
- Per JMT rule [[feedback_verify_against_reference_or_dont_claim]]: per-seed Procrustes only;
  no aggregate/TOST hand-waving; re-verify, don't trust commit messages.

## DONE procedure
1. Finalize STATUS.md (every engine classified, with numbers + evidence paths).
2. Write SUMMARY.md (what hit <1e-7, what's stuck + why).
3. Text JMT: "RNG-matching sprint done. N/24 algos bit-exact <1e-7 on small graphs; M stuck
   (reasons). Ready to talk + then spot-check before the full sweep."
4. Do NOT launch the full all-graphs sweep -- JMT directs that next.

## Log
| Round | Phase | engines dispatched | result |
|---|---|---|---|
| 0 | P0a | instrumented graphviz | DONE: ~/tools/graphviz-7.0.5-instr/ built; VERIDICAL PROOF PASS (54/54 bit-for-bit == stock, max_rmsd=0 -> logging-only confirmed); GV_TRACE %.17g works. Artifacts in scripts/rng_match/ (build script + patch + README) -- UNCOMMITTED (commit after P0b). Caveats: dot_builtins wrapped as bin/dot (no libltdl); fdp/neato/sfdp work. |
| 0 | P0b | bit-exact harness | running (pid in /tmp/rng_p0b.pid) |
