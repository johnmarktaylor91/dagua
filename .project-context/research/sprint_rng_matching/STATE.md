---
run: sprint_rng_matching
created: 2026-06-02
state: DONE
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

## INVARIANT -- MATCHED PARAMS (JMT emphatic 2026-06-02, see [[feedback_always_parameter_match_comparisons]])
EVERY analysis going forward MUST compare at matched parameters: the REFERENCE runs the SAME
config (iters/rounds/steps/perplexity/etc.) as the dagua variant, NOT reference defaults.
Also matched: seed (both seed=N), dtype (float64), graphviz build (instrumented==stock).
GUARANTEE MECHANISM:
- variants.py original_params mirrors reimpl_params (param-matching codex 2186380 fixing this).
- adapters pass params through to the reference call (graphviz -G / igraph kwargs / OGDF runner / etc.).
- HARNESS GUARDRAIL (must add to scripts/rng_match/bitexact_harness.py): record BOTH sides'
  effective params per row; FLAG/refuse any pair where reference ran defaults or params don't
  correspond -- never emit a "clean" RMSD for a mis-parameterized comparison. (Add this in the
  post-param-matching verification step; dispatch a tiny harness-guardrail codex if needed.)
- NEVER conclude an engine diverges without confirming matched params + seed first.
gem proved it: looked "divergent 1.15", was bit-exact 3.86e-13 at matched rounds.

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
| 0 | P0b | bit-exact harness | DONE + CC-VALIDATED. Baseline: 52 BIT_EXACT, 44 DIVERGENT, 1 CLOSE, 10 no-ref, 8 unavail, 6 err. Foundation committed 2b3efd0. Harness discriminates correctly (tsnet/linlog/kk exact ~1e-16; fa2/graphopt/lgl ~3e-8 BIT_EXACT but SQUEEZABLE to 1e-15; divergent are real). |
| 1 | P1 wave1 + param-match + OGDF | (done) | COMMITTED f60944e. Matched-params authoritative baseline: 60 BIT_EXACT / 36 DIVERGENT / 6 neulay-ERROR / 8 sgd2-UNAVAILABLE / 10 no-ref / 1 CLOSE. Newly bit-exact: gem, pivot_mds, stress_maj, umap. Anti-cheat clean. |
| 2 | P1 wave2 (ports+iterate+depfix) | neato(2440040) sfdp(2440141) fmmm(2440413) maxent(2440612) classical_mds(2440803) drl(2440989) sgd2multi(2441175) neulay(2441364) | RUNNING. Specs PORT2_*/FIX_*.md. Each owns distinct file (drl owns _igraph_rng.py; classical_mds deterministic-no-RNG). On each DONE: anti-cheat grep + check_engine + update STATUS + commit clean. |

## Wave-2 results (partial -- batch-assess when all 8 done; anti-cheat clean so far)
- drl: igraph RNG MOSTLY matched -> 35/42 fixtures bit-exact, 7 diverge (max 1.0). Likely
  chaotic-anneal minority OR a residual RNG-draw-order case. Candidate for 1 targeted retry then accept.
- classical_mds: 1.09 -> 0.77. Bit-exact on non-degenerate fixtures (numerical floor); STUCK on
  igraph vendored-LAPACK dsyevr eigenvector basis for REPEATED eigenvalues (degenerate-eigenvector
  convention) -- genuine numerical wall, NOT RNG. Strong "documented can't-go-further" candidate.
- sgd2multi, neato, sfdp, fmmm, maxent, neulay: DONE (narrative; clean re-verify pid 2481689 running):
  * neato BIT_EXACT 6.07e-16 (42/42) -- convergence-stop fix WORKED.
  * maxent_stress BIT_EXACT 4.78e-16 (42/42).
  * neulay BIT_EXACT 6.63e-16 (42/42) + dep-fix made it RUN (old-code dim/budget bug, not missing PyG).
  * sgd2_multi: dep-fix -> now RUNS (was UNAVAILABLE) -> 0.141 DIVERGENT (needs a PORT next, wave 3).
  * fmmm 1.39->2.08e-2 (35/42 exact; residual = 1-unit integer-coord drift in OGDF final packing).
  * sfdp 0.81->0.44 (partial, multilevel hard). drl 35/42 (partial). classical_mds STUCK (LAPACK degenerate basis).
  * NOTE: status.json got CORRUPTED by 8 concurrent codex check_engine writes -> the single-writer
    re-verify (2481689) regenerates it clean. Anti-cheat grep clean across all wave-2 changes.
- ENDGAME: when all done -> clean full re-verify + anti-cheat grep ALL + commit + classify each
  engine (BIT_EXACT / STUCK-with-documented-reason / dep-fixed). Then wave 3 (davidson_harel +
  sugiyama using drl's _igraph_rng work; squeeze fa2/graphopt/lgl/gem to floor). Then SUMMARY + text JMT.
- DRIVE-OR-WALL: per JMT, each engine either <1e-7 OR a precise documented irreducible reason
  (LAPACK degenerate basis, chaotic FP cascade, non-reproducible source). No flailing, no faking.

## Wave 3 DISPATCHED 2026-06-02 18:03 (committed wave-2 = 33d4f5b, 68 bit-exact)
6 codexes: igraph(2842735: drl+davidson+sugiyama, owns _igraph_rng.py) | sgd2multi(2842848) |
ogdf_finish(2843116: fmmm+maxent) | sfdp(2843316) | fr_fa2(2843528) | classical_mds(2843735 final).
Specs W3_*.md. ACCEPTED WALLS (no codex): gem_iters2000 (chaotic FP at 2000 rounds; bit-exact at
100/500), fcose (no python port -> flag JMT), NO_REFERENCE chains (no ref to compare).
ON ALL 6 DONE: clean full re-verify -> commit -> classify each -> SUMMARY.md -> TEXT JMT (sprint done).
Standing: 68 BIT_EXACT / 2 CLOSE / 41 DIVERGENT / 10 no-ref. Targets after wave3: close the
finishable (fr_steps100 1.86e-7, sgd2_multi_batch128 9.86e-6, fmmm ~0.01, fr_steps200/fa2_linlog
~1e-3) + crack the ports (igraph family, sgd2_multi, sfdp) OR document each wall precisely.

## FINALIZATION (wave-3 committed 51d7ebf; final re-verify pid 2879837, watcher br4ut2mvd, ~90min/neulay)
ON re-verify3 DONE (writes STATUS.md): read final verdict counts -> write SUMMARY.md -> TEXT JMT.
NOTE: commits skip detect-secrets via SKIP= (verified TIMESTAMP-ONLY churn, NO real secret; all
other hooks run; NOT --no-verify). status.json gitignored; ogdf_runner binary gitignored.

### Documented IRREDUCIBLE WALLS (for SUMMARY -- these are "genuinely can't get further", precise):
- classical_mds: igraph vendored LAPACK 3.4.2 dsyevr eigenvector basis for DEGENERATE eigenvalues
  (multiplicity>2) is implementation-dependent; SciPy/torch can't reproduce without porting LAPACK
  tridiag-reduction + inverse-iteration. Bit-exact on NON-degenerate fixtures.
- sfdp: compiler/libm-level FP drift in transcendentals, amplified by chaotic multilevel iterations.
- sugiyama: igraph layered tie-breaking on SYMMETRIC graphs (complete5/petersen/wheel/two_triangles).
- drl: igraph RNG mostly matched (35/42 fixtures bit-exact); diverges on a few symmetric/chaotic cases.
- davidson_harel: igraph anneal RNG; diverges on specific fixtures (path8/grid3x3/complete5 seeds).
- fmmm: force-arithmetic order on SYMMETRIC cases before OGDF integer export (35/42; ~0.01-0.02).
- gem_iters2000: chaotic FP cascade at 2000 rounds (bit-exact at 100/500 rounds).
- fcose: NO python port exists (cytoscape fCoSE) -- needs from-scratch port, flag to JMT.
- NO_REFERENCE (10): fr_kk/kk_fr chains, spectral_random_walk/unnormalized, rt_horizontal -- no
  paired reference to compare against (not a fidelity failure, just nothing to match).
### THEME (the honest finding): remaining walls cluster on SYMMETRIC/DEGENERATE small graphs where
the reference's implementation-specific choices (LAPACK degenerate basis, igraph tie-breaking, libm
rounding) are not reproducible in pure python/torch. This is the genuine floor, precisely characterized.

## Held (now wave-3) -- superseded by dispatch above
- davidson_harel, sugiyama (igraph RNG -- wait for drl to settle _igraph_rng.py).
- fr_steps200/500 iterate (fr_steps100 already CLOSE 1.86e-7), fa2_linlog iterate (2.1e-3).
- SQUEEZE bit-exact-but-not-floor: fa2/graphopt/lgl (~3e-8), gem (~8e-8 -> pin exact OGDF commit for ~1e-13).
- fcose (NO python port -- flag to JMT, not a quick fix). NO_REFERENCE chains (fr_kk etc -- can't compare).
- FINAL: clean full re-verify (OGDF rows were mid-rebuild in the f60944e baseline -> refresh), then SUMMARY + text JMT.

## Wave-1 results (as they land)
- **gem DONE 2026-06-02**: NO code change needed. dagua GEM ALREADY bit-matches OGDF at MATCHED
  rounds: 100rounds=3.86e-13, 500rounds=3.93e-08 (both <1e-7!). The baseline "1.15" was a
  HARNESS PARAMETERIZATION artifact: harness runs ogdf_gem ref at DEFAULT 30000 rounds (OGDF
  runner ignores rounds param) vs dagua variant's 100/500/2000. 2000rounds genuinely chaotic
  (1.65e-1, late FP drift). ACTION (CC, central): make ogdf_gem ref honor rounds (runner/variant
  change) so iters100/500 show bit-exact; document iters2000 as chaotic-floor. NO delegation.
  >> LESSON: check reference PARAMETERIZATION -- some "divergences" = ref run at wrong params.

## Wave-1 ALL DONE (PRELIMINARY -- re-verify after param-matching) + anti-cheat CLEAN (no delegation in any)
- gem: already bit-exact at matched rounds (param issue, see above).
- neato: 1.23 -> 7.3e-3. Ported graphviz srand48/drand48 init. Residual = convergence-termination
  (graphviz stops ~152 iters on path8, dagua runs 200). Changed neato default to "graphviz". WAVE-2 iterate.
- davidson_harel: still ~0.36 (rounds50/100/200, worst grid3x3 seed3). NO improvement -- BUT igraph
  ref may be running default params (maxiter) -> RE-VERIFY after param-matching before judging.
- drl: still ~1.0 (34/42 exact). NO improvement -- same caveat (igraph param-matching).
- sfdp: made code changes (no clean number extracted) -- re-verify.
- umap: made code changes -- re-verify.
>> NEXT (when param-matching codex 2186380 DONE): re-run full harness with MATCHED PARAMS ->
   authoritative per-engine numbers. THEN: commit clean ports (grep diff for delegation first),
   classify done/iterate/stuck, dispatch wave 2. HOLD all commits until then.
>> IMPORTANT: davidson_harel/drl "no improvement" is likely the igraph adapter not passing
   maxiter/rounds (gem-style param mismatch). Don't conclude port-failure pre-param-matching.

## OGDF BUILD DONE (2026-06-02 14:54) -- gem FIXED
- OGDF installed ~/tools/ogdf (official tag foxglove-202510, static libs). Runner rebuilt
  (scripts/ogdf_runner) -- now honors --gem-rounds/--fmmm-fixed-iterations/--iterations/
  --number-of-pivots. Build script scripts/rng_match/build_ogdf_runner.sh.
- gem: NOW BIT_EXACT (iters100=7.97e-8, iters500=8.25e-8). param-matching + rebuild fixed it.
  SQUEEZE later: ~8e-8 not ~1e-13 because used official OGDF tag, not the exact 2026 ref commit.
- fmmm: STILL DIVERGENT 1.39 at matched iterations -> genuine PORT needed (wave 2).
- pivot_mds / maxent_stress / stress_maj: runner now honors params -> RE-VERIFY (post-rebuild).
- CAVEAT: harness re-verify (pid 2371127) started 14:47, runner rebuilt 14:54 MID-RUN -> its
  OGDF rows are inconsistent (binary swapped mid-run). MUST re-run OGDF engines cleanly after.
- UNCOMMITTED so far (commit after clean re-verify): param-matching (variants.py + graphviz/
  fcose/ogdf competitors + ogdf_runner.cpp), OGDF build scripts, + wave-1 port pipeline edits.

## DEP/RUNTIME FIX WAVE (JMT: "fix the dependency and runtime errors too" 2026-06-02)
Make blocked engines actually evaluable on small graphs -- FIX, don't just document:
- sgd2_multi (8 variants UNAVAILABLE): s_gd2 IS installed and `sgd2` ref works, but
  `sgd2_multi_ref.available()` returns False for another reason -> diagnose its available()
  gate + fix so it runs on small graphs.
- neulay (6 variants ERROR): ref is available() + imports fine -> errors are RUNTIME (likely
  GNN-train timeout or exception on some graphs) -> diagnose actual error + fix.
- Plus any GENUINE ERROR remaining after the clean post-param-matching re-verify (sfdp showed
  ERROR in the racy mid-flight STATUS -- likely transient; confirm on clean run).
DISPATCH AFTER param-matching lands (it edits adapters incl. sgd2 -> conflict if parallel).
fcose (cytoscape, 2): genuinely NO python port -> that's a missing-port, not a runtime error;
separate (would need a from-scratch fcose port -- flag for JMT, not part of dep-fix).

## Work list for later waves (after wave 1 validates the loop)
DIVERGENT engines not yet dispatched: sugiyama (igraph/graphviz_dot), classical_mds (igraph_mds),
fmmm/fdp (ogdf+graphviz), maxent_stress + stress_maj (ogdf_stress), pivot_mds (if divergent),
+ SQUEEZE the BIT_EXACT-but-3e-8 ones (fa2, graphopt, lgl) toward 1e-15.
NO_REFERENCE (10) + UNAVAILABLE (8: sgd2_multi ref too slow) + ERROR (6: neulay timeout) =
likely cannot be matched on small graphs; document in final SUMMARY.
