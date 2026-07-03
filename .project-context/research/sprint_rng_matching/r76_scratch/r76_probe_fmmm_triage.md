<task>
r76-B1: triage dagua's 33 remaining divergent fmmm combos against the now-HONEST references
(the stale ogdf_runner binary ignored fmmmFixedIterations; it was rebuilt in r75 commit 0817427
and all ogdf_fmmm references regenerated -- read r75_RESULTS.md "What r75 discovered" first).
RESEARCH/PROBE ONLY: no repo changes; scratch in /tmp; write only the results file.

Repo: /home/jtaylor/projects/dagua (develop, read-only). OGDF source (runner's actual version):
/home/jtaylor/tools/ogdf-src (foxglove-202510).

DATA: eval_output/fidelity_definitive/r75_final.jsonl -- rows with engine containing
classic_fmmm and quality_identical_raw=false and no_canonical_reference!=true. For each: which
legs fail (battery_stress/cross/np *_direct_equivalent), D vs R vs margin, disconnected flag.
Also eval_output/benchmark_100seed_r75_fixes has fresh positions for dagua fmmm AND the
regenerated ogdf_fmmm__for__* references (per-combo freshest overlay applies; topup dirs too).

QUESTIONS (answer ALL with numbers):
1. LEG BREAKDOWN: cluster the 33 by failing-leg pattern and gap size (relative for stress, ABS
   count for crossings -- integer discreteness matters; note cross_margin/ref_self_spread).
2. HAIRLINE vs STRUCTURAL: r75 probes showed dagua-vs-honest-runner RMSD ~1e-3 at steps10 on
   probe graphs. Compute Procrustes RMSD dagua-vs-reference per combo (5 seeds sample) from saved
   positions: how many are <0.01 (near-match, likely crossings-discreteness or margin-power
   fails) vs genuinely apart?
3. steps100/steps200 PARITY: is divergence concentrated at higher step counts? (Chaotic
   divergence growth with iterations vs a systematic defect -- compare RMSD at steps10 vs 100 vs
   200 on 3 shared graphs.)
4. deep_chain_20::classic_fmmm_steps200 regressed (was identical under stale accounting) --
   root-cause this one specifically (positions + legs).
5. BIT-EXACT FEASIBILITY: dagua fmmm consumes an _OgdfMt19937 stream (pipelines/fmmm.py); the
   honest runner uses OGDF's randomNumber seeded via setSeed (scripts/ogdf_runner.cpp:320-328).
   For ONE small graph/seed, dump dagua's first 20 RNG draws + OGDF's (instrument via a tiny
   patched local build to /tmp ONLY if cheap -- else infer from source) and state whether full
   per-seed bit-exactness is a stream-alignment fix (like r71's wins) or blocked by float/order
   differences. RECOMMEND: the minimal path to close each cluster (fix / floor-evidence /
   aggregate-tier) with effort estimates.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_fmmm_triage.md --
commands, tables, per-cluster verdicts + recommendations. ASCII. Budget ~45 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one question blocks, document and continue.
</default_follow_through_policy>
