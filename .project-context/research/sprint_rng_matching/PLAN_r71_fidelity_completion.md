# PLAN r71: Fidelity Completion (goal: 100% fidelity, honestly defined)

Version: 4 -- **APPROVED** (round 3 PASS 2026-06-12; rounds 1-2 -- 18 findings total, all incorporated;
resolutions in Appendices A-B). Author: CC. Max 3 Fable rounds per JMT.
JMT directives (2026-06-12): Tier 1b formalized (DONE e9e91c7); verify-then-run seeded
originals; plan + fix the failures; bigger toolkit budget (W4 RUNNING); sugiyama stale
rung-0 reclass. "The goal is 100% fidelity."

## 0. What "100% fidelity" means (pre-registered, no threshold games)

Success = every one of the 118 variants ends in exactly one of:
- **Tier 1** (bit-exact) / **Tier 1b** (invariance-exact)
- **DISTRIBUTIONALLY_MATCHED** (optionally SEED_FAITHFUL)
- **REF_COMPATIBLE** ONLY where the reference is unseedable under the sec.-2b EVIDENCE
  STANDARD (probe + positive control + upstream-source evidence)
- **REF_COMPATIBLE_POWER_LIMITED** (round-1 finding 6; trigger PINNED per round-2
  finding 3): permitted ONLY when (a) full failing-map coverage at max seeds, (b) the
  observed effect size is below the OC-grid minimum detectable effect at n=100
  (oc_simulation.json + per-combo expected_false_atypicality), (c) the cited OC row is
  published in the report appendix. No other path to this label.
- **DOCUMENTED-IRREDUCIBLE**: every remaining divergent combo carries a root-cause label
  from the closed set {MEASUREMENT_ARTIFACT (then fixed), PARAM_MISMATCH (then fixed),
  PORT_BUG (then fixed), CHAOTIC_BASIN (sec.-3b ensemble-level evidence attached),
  STRUCTURAL_NA (e.g. umap nn30 > n_nodes), NO_REFERENCE, NO_PORT}
- **ZERO UNDETERMINED engines.**
Fix levers are code and data ONLY. r70 thresholds (q95, 90%, BH 5%, 1e-3) NOT revisited.

## 1. Failure inventory (r70, verified; Tier-4 mode split per round-1 audit)

| Bucket | Count | r71 disposition |
|---|---:|---|
| Tier-4 escalation combos | 705 = 72 Mode A + 633 Mode B | P2 (Mode A now; Mode B AFTER P1d) |
| of the 633 Mode B, seedable-ref TODAY | ~364 (gem 123, sfdp 106, fmmm 98, pivot 16, neato 11, maxent 6, ogdf_stress 4) | P1 re-verdicts them under Mode A first |
| Deterministic DIFFERENT / QUALITY-only | 71 / 40 | W4 (RUNNING) then P2 |
| UNDETERMINED engines | 22 | P1 (most), P3 (coverage), sec.-0 labels (residual) |
| INSUFFICIENT_DATA combos | 247 (78 matched<30, 70 reimpl<30, 14 ref<30, 85 no_ref_rows) | P3 repair or STRUCTURAL_NA |
| Sugiyama stale rung-0 (residual_block) | 5 | W5 reclass (mechanical) |

## 2. P1 -- Seeded-reference upgrade (REVISED: the plumbing already exists)

**Round-1 ground truth (reviewer-verified, live-probed):** graphviz adapter passes
`-Gseed/-Gstart` (graphviz_competitor.py:413-415, commit 58359b2); the OGDF runner
consumes seed via `ogdf::setSeed`/srand (scripts/ogdf_runner.cpp:307-322, commit 52930fe;
binary built Jun 2); run_benchmark already forwards seeds (run_benchmark.py:1475).
sfdp/neato/fdp/gem/fmmm/ogdf_stress VARY by seed and are stable within seed TODAY.
The ONLY blockers: `_BASE_ENGINE_STOCHASTICITY` (variants.py:~2110) marks these refs
non-stochastic so `seeds_for_engine()` returns [None]; and igraph_sugiyama's adapter
DROPS the seed (`uses_igraph_rng` unset -- a bug, not unseedability).

a. **Harness task (codex; SPLIT into 2 sequenced dispatches per the 1-3-file sizing rule: (i) seeds override + original_engine routing + igraph adapter fix + regression locks; (ii) provenance stamp + merge source_dir tags + report assertion):**
   - run_benchmark CLI override `--seed-refs <comma engines>`: run-scoped stochasticity
     override; the GLOBAL table is untouched (flipping it would silently change record
     keys `::deterministic` -> `::seedN` for every future run, break --resume on existing
     dirs, and multiply standard-benchmark cost -- round-1 finding 1). The override MUST
     apply at BOTH seeds_for_engine call sites -- job enumeration (run_benchmark.py:~2158)
     AND position-recovery enumeration (~1115) -- and must match `__for__` synthetic
     names via original_engine routing (variants.py:~263) (round-2 finding 6).
   - **git-SHA provenance (round-2 finding 2, HIGH):** stamp `git_sha` into run_benchmark
     run metadata at start; merge_benchmark_datasets.py tags every merged row with
     `source_dir`; report v2 gains a hard assertion (same pattern as
     check_no_mixed_modes) that for every code-fixed engine ZERO rows originate from
     pre-fix source dirs -- report build FAILS otherwise. Without this the per-engine
     single-code-state claim is unverifiable prose.
   - igraph adapter: wire `uses_igraph_rng=True` for IgraphSugiyama (and audit siblings)
     so seeds actually reach igraph's RNG; whether output then varies is the PROBE's
     question, not assumed.
   - Regression locks: seed=None output byte-identical to a stored r70 reference layout
     on 3 graphs per family; same-seed determinism; different-seed variation asserted
     only for the probe-verified-seedable set.
b. **Seedability probe (gate before mass compute; HARDENED per round-1 finding 2):**
   per reference adapter x failing-map MID-SIZE graphs (>=2, drawn from that engine's own
   failing map -- tiny graphs do not exercise RNG paths like crossing-min tie-breaks) x 3
   seeds. "PROVABLY unseedable" requires ALL of: (i) no variation in this probe, (ii) a
   POSITIVE CONTROL in the same harness run (known-seedable engine varies), (iii)
   upstream-source evidence the algorithm consumes no RNG (igraph C source / graphviz attr
   docs), recorded per engine in the probe table (published in report v2). Current
   evidence-track candidates: ogdf_pivot_mds (no variation in round-1 probe, plausibly
   deterministic pivots), igraph_mds, igraph_rt, igraph_sugiyama-after-bug-fix.
c. **Disk pre-step (round-1 finding 8; disk at 35G = 7.6% free, BELOW the 10% global
   threshold):** archive old benchmark stores to /mnt/locker/jt3295/dagua_archives
   (benchmark_100seed_final ~14G+, stale round dirs) BEFORE P1d; EXECUTE IN THE NOW BLOCK; target >=50G free;
   stall-killer invocation gains a disk floor check (abort new writes < 15G).
d. **Reference benchmark:** the exact `<ref>__for__<variant>` SYNTHETIC engine names from
   failing_map (param-mirrored -- running base refs would repeat the gem default-30000-
   rounds incident; round-1 finding 7), x their failing-map graphs x seeds 42-141, via
   `--seed-refs`, fresh dir `eval_output/benchmark_100seed_seeded_refs`, --resume,
   stall-killer. Runner asserts seed AND variant_params are BOTH forwarded (one
   smoke assertion per family before the full run). Measured cost: ~10-13 CPU-h total
   (ref runtimes 0.02-1.24s) -- NOT machine-days.
e. **Re-analysis (P1e):** upgraded combos re-run under Mode A against the union store
   (classify_mode already prefers Mode A when >=30 seeded ref rows exist -- verified).
   PRE-REGISTERED partition change (round-1 finding 9; round-2 finding 4): combos of
   upgraded engines with <30 seeded ok ref rows take the EXISTING `ref_seeds_lt_30`
   Mode-B disposition (sanctioned, enumerated in the report) -- the new assertion is
   "every upgraded engine's scored Mode-A combos have >=30 seeded ok ref rows; <30
   combos carry ref_seeds_lt_30", and the assertion rewrite MUST land BEFORE the first
   post-P1e report regeneration (check_no_mixed_modes fires on any A/B mixture
   otherwise). Union-store merges are SERIALIZED under one merge owner (CC). Verification
   for upgraded engines uses definitive_fidelity_analysis + check_engine.py ONLY --
   fast_fidelity_report's `_resolve_pos` seedN->deterministic fallback is FORBIDDEN for
   them (silent mislabeling risk; round-2 finding 5).
   EXPECTATION SET UPFRONT: seed-tracking will usually FAIL (dagua's ports were never
   RNG-matched to these binaries' streams) -- NOT a fidelity failure; the claim sought is
   DIST_EQUIVALENT.
f. Mode B typicality retires for upgraded combos (EXCEPT the sanctioned
   ref_seeds_lt_30 disposition, sec. 2e); survives only for sec.-2b-proven
   unseedable refs (or becomes REF_COMPATIBLE_POWER_LIMITED per sec. 0).

## 3. P2 -- Tier-4 root-cause loop (GATED on P1 for Mode B; round-1 finding 3)

a. **Cluster table (CC, plain python; build NOW, mark Mode-B clusters PROVISIONAL):**
   all rung-4 + post-W4 deterministic-DIFFERENT combos, grouped by (engine family x graph
   class x flags) with r70 diagnostics (e_rel, disp, d_R vs W_D, stress deltas,
   disconnected/degenerate/TRACKING_BUT_SHIFTED, size bin, spot-check would-flip).
b. **Diagnosis ladder (per cluster, evidence recorded):**
   1. MEASUREMENT_ARTIFACT: invariance flip / disconnected / degenerate / scale -> fix
      measurement, re-verdict.
   2. PARAM_MISMATCH: re-verify the variant->ref param mirror (the gem lesson) -> fix,
      re-benchmark cluster.
   3. INIT/RNG MISMATCH: matched-seed iteration-0 diff -> port init/stream per
      PORTING_PROTOCOL.md.
   4. PORT_BUG: iteration-trace divergence point on a small failing graph (R40-65
      protocol) -> codex fix; verification = per-seed Procrustes vs reference via
      scripts/rng_match/check_engine.py (NEVER aggregate tiers -- the false-bit-exact
      lesson); no-delegation guard.
   5. CHAOTIC_BASIN (RE-REDEFINED, round-2 finding 1 -- the eps-perturbed-init control
      is UNEXECUTABLE for binary refs: graphviz adapter emits no pos attrs, ogdf_runner
      has no init-position input). The label may attach ONLY if BOTH:
      (i) SEED-SPLIT SELF-ENSEMBLE control (zero new compute; machinery = the existing
      E_self split calibration): the reference's own 100 seeded rows split 50/50 must FAIL
      the SAME distributional test against itself (SPEC_definitive_fidelity_analysis.md
      sec. 4.3 machinery; FAIL := median(E_cross between the two 50-seed halves) >
      q95(E_self within-half); no separate margin) -- i.e., the reference cannot reproduce its own distribution
      at this graph. Conservative direction: lower power makes chaos HARDER to claim.
      Optional strengthener for Python/igraph refs only (which accept init): the
      eps-perturbed-init ensemble. AND
      (ii) the reimpl passes the EXISTING stress TOST + quality battery vs the reference
      (no new metrics/margins -- round-2 finding 7; "r70 thresholds not revisited").
      A faithful chaotic port still reproduces ensemble statistics.
      **Combos whose reference is sec-2b-proven unseedable can NEVER receive
      CHAOTIC_BASIN** (no ensemble exists for any control) -- they route to PORT_BUG /
      REF_COMPATIBLE / REF_COMPATIBLE_POWER_LIMITED.
c. **Fix dispatches (SEQUENCED):** START NOW only on Mode-A-derived clusters -- actual
   composition (round-1 audit): umap 34, drl 20, sgd2_multi 18 Tier-4 combos. ALL
   Mode-B-derived clusters (gem/sfdp/fmmm/neato/maxent/ogdf_stress/sugiyama...) WAIT for
   P1e re-verdicts (re-cluster after upgrade; chasing typicality verdicts P1 is about to
   recompute is a machine-day-class waste). Codex per family, 1-3 files, max 3 rounds
   then accept residual.
d. **Fixed-engine re-run scope (round-1 finding 5):** ANY engine receiving a code fix
   re-runs its ENTIRE failing-map graph set at 100 seeds (not just the fixed combos) --
   its previously-passing rows are invalidated by the fix; report v2 records the
   data-generating git SHA PER ENGINE so the 100% claim describes a single code state per
   engine. Measured worst-family cost: drl ~54 CPU-h (~3-5 wall-h at 12-20 workers);
   davidson 14 CPU-h; umap 1.4 CPU-h -- bounded.

## 4. P3 -- Data-gap repair

- Timeout-heavy combos (fr_steps500 coverage + the 78+70 seed-shortfall combos): re-run
  ONLY those cells at --timeout 900; merged runtimes footnoted in v2 (mixed-timeout
  runtime_ratio is diagnostic-only -- verified; round-1 finding 11).
- no_reference_rows (85, mostly sgd2_multi_ref gaps): re-run those reference cells.
- umap nn30 where n_neighbors > n_nodes: STRUCTURAL_NA.
- Residual ref_seeds_lt_30 combos P3 cannot repair: STRUCTURAL_NA-class terminal
  disposition, enumerated (round-3 finding 6).
- Re-analysis of all repaired combos.

## 5. P4 -- Final assembly

- Archive r70 report as DEFINITIVE_FIDELITY_REPORT_r70.md (immutable baseline).
- Report v2: Tier 1b live; P1 probe table + upgrade deltas (r70 -> r71 per-engine
  headline changes shown EXPLICITLY); root-cause appendix; per-engine data SHA;
  updated assertion set (sec. 2e); strict render; supersession + revision log.
- Gate check (sec. 0), commit, file-for-review, text JMT.

## 6. Execution discipline

- State r71_fidelity_completion_STATE.md + autonomous_gate_r71.json (exist).
- Benchmarks via stall-killer (+ disk floor) + Monitor watchers; liveness kill -0.
- Codex quota fallback chain; pause sentinels per dispatch; BLISS subprocess pattern for
  any toolkit call.
- Anti-flail: 3 rounds per family; sec.-0 terminal labels are legitimate with evidence.
- P1a audit list TRIMMED to refs with failing-map combos (drop ogdf_davidson_harel,
  ogdf_sugiyama base, igraph_rt from the seeding track; keep on the evidence track only
  if needed for probe positive controls) (round-1 finding 10).

## 7. Sequencing summary

NOW (parallel): W4 (running) -> W5; P1c disk pre-step (round-3 finding 7: disk below
floor NOW); P1a codex dispatch (i); P2a provisional cluster table; P2c Mode-A fix loop
(umap/drl/sgd2). THEN: P1a dispatch (ii) -> P1b probe -> P1d seeded-ref bench ->
P1e re-analysis -> re-cluster -> P2c Mode-B fix loop -> P3 -> P4.

## Appendix B -- round-2 resolutions (2026-06-12)

7 findings (2 HIGH, 3 MED, 2 LOW), all accepted: (1) CHAOTIC_BASIN control replaced with
seed-split reference self-ensemble (eps-init injection unexecutable for binary refs);
unseedable-ref combos barred from the label [HIGH] -> sec. 3b.5. (2) git-SHA provenance
made machine-checkable (run metadata stamp + source_dir tags + hard report assertion)
[HIGH] -> sec. 2a. (3) POWER_LIMITED trigger pinned to OC-grid MDE [MED] -> sec. 0.
(4) ref_seeds_lt_30 sanctioned disposition + assertion-rewrite ordering [MED] -> sec. 2e.
(5) fast-report fallback forbidden for upgraded engines [MED] -> sec. 2e. (6) --seed-refs
at both call sites + original_engine routing [LOW] -> sec. 2a. (7) edge-length TOST
dropped; existing stress/quality battery only [LOW] -> sec. 3b.5(ii).
Round-2 verified bonus: Mode-A Tier-4s concentrate on weighted/multiedge graphs (umap
e_rel 0.56-1.02 weighted vs 0.03-0.11 unweighted; drl weighted_clusters e_rel 1.0-1.33)
-- suspected systematic edge-weight/multiedge handling mismatch (CODEABLE); sgd2_multi is
13/18 TRACKING_BUT_SHIFTED at tiny e_rel -> ladder step 1 (measurement/transform triage)
before any port surgery.

## Appendix A -- round-1 resolutions (2026-06-12)

11 findings (5 HIGH, 4 MED, 2 LOW), all accepted: (1) P1a re-scoped to harness override
+ igraph bug (plumbing exists; global-table flip forbidden) [HIGH] -> sec. 2a.
(2) "provably unseedable" evidence standard: positive control + upstream source + mid-size
failing-map probe graphs [HIGH] -> sec. 2b. (3) P2c gated: Mode-A clusters now, Mode-B
after P1e; priority list synced to actual Tier-4 composition (stress_sgd/davidson have
ZERO Tier-4) [HIGH] -> secs. 3c/7. (4) CHAOTIC_BASIN redefined at ensemble level +
aggregate checks [HIGH] -> sec. 3b.5. (5) fixed-engine full failing-map re-run + per-engine
data SHA [HIGH] -> sec. 3d. (6) REF_COMPATIBLE_POWER_LIMITED terminal label [MED] ->
sec. 0. (7) P1d uses `__for__` synthetic names + param/seed forwarding assertion [MED] ->
sec. 2d. (8) disk pre-step + stall-killer floor (35G < 10% threshold) [MED] -> sec. 2c.
(9) pre-registered partition/assertion change for the union store [MED] -> sec. 2e.
(10) audit list trimmed; ogdf_pivot_mds to evidence track [LOW] -> secs. 2b/6.
(11) runtime footnote [LOW] -> sec. 4.
