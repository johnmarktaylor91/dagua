# r78 FINAL-PUSH: neato residual bucket (54+5 rows) -- forensic verdict (2026-07-08)

## Headline
The r77 ledger's 54-row "Graphviz neato exact CG/drand48/component packing residual"
bucket is a STALE-LEDGER ARTIFACT, already retired by r78 (commit 00bd57a + fresh
both-sides bench). Independently re-verified here with per-seed probes vs live
neato 7.0.5. Zero fixable residual remains in classic_neato. Of the 5
"Graphviz FDP/neato packing residual" rows, 4 flipped EQUIVALENT in r78_fdp.jsonl;
1 (parallel_cycles_4x5::classic_fmmm_graphviz_fdp_fidelity) is a VERIFIED FLOOR
where dagua's stress is BETTER than reference (0.082 vs 0.134).

## Task 1: cluster analysis (connected-graph anomaly CONFIRMED)
- 51/59 rows are CONNECTED (47/54 neato + 4/5 fdp). Packing never runs for them,
  so the named cause was wrong for 86% of the bucket. Real story: Mode-B seed-42-era
  references (all 54 neato rows flagged SEED_42_ERA) scored a distribution against
  one stale deterministic draw.
- 8 disconnected: dependency_graph_100(2), disconnected_encoder_residual(2),
  disconnected_label_cycle_collage(3), kitchen_sink_platform_graph(2),
  multi_component_80(7), parallel_cycles_4x5(4, both engines), random_bipartite_60(4).

## Task 2: bisection probes (fresh, this session, live neato 7.0.5, matched params
##          -Gmaxiter=200 -Gepsilon=0.0001 -Gstart=seed, dagua node_sizes/72)
- Connected classic_neato, 20 graphs x 3-5 seeds (seeds 100-104), incl. all 18 combos
  absent from r78_neato.jsonl: per-seed scale-normalized Procrustes rel-RMSD
  <= 6.2e-5 on EVERY seed; many exactly 1e-16..1e-8. First divergent stage: NONE --
  init(maxiter=0) also matches (petersen 3.8e-15).
  The 1e-5-class residual on non-exact seeds is the %.5g JSON-export quantization +
  C-vs-torch float accumulation boundary flips; dagua already models the %.5g rounding
  (neato.py:727-731). This is the measured floor: ~5th significant digit.
- Disconnected classic_neato (parallel_cycles_4x5, multi_component_80,
  random_bipartite_60): per-COMPONENT rel-RMSD <= 1.9e-4 (bit-exact class);
  only inter-component pack arrangement differs vs live binary (global rel 0.04-0.49).
  FIRST DIVERGENT STAGE = component packing placement, NOT init/CG/drand48.
  The official scorer's stress metric uses finite-graph-distance pairs only
  (within-component), so pack arrangement is outside the fidelity construct;
  r78 rescore scores these IDENTICAL (stress_D == stress_R to 6-7 digits).
  True positional pack match would need graphviz packGraphs doSplines=1
  spline-occupancy polyomino rasterization (documented r78 floor; unchanged).
- FDP row parallel_cycles_4x5 (param-matched steps=200 / -Gmaxiter=200, 10 seeds):
  dagua mean stress 0.0820, ref 0.1338; ref at equilibrium (maxiter=600 -> 0.1301).
  Near-init (dagua steps=1 vs ref maxiter=1) rel ~0.05-0.10, so init is seed-aligned;
  divergence accumulates in fdp xLayout grid force loop. Whole fdp family is
  distribution-matched only (IDENT rows have W_D 0.6-0.9; no per-seed matching
  anywhere), so this row's TOST miss -- caused by dagua being BETTER -- is the
  family floor, not a fixable defect.

## Task 3/4: fix vs floor
- NO code changes made (repo clean). Nothing fixable found: connected rows already
  at export-precision; disconnected components bit-exact.
- Floors: (a) %.5g export + C double accumulation order in CG (connected, <=6e-5 rel);
  (b) packGraphs spline-occupancy arrangement (disconnected, metric-exempt);
  (c) fdp grid force-loop equilibrium distortion on 5-cycles (dagua superior stress).

## Closeable-row estimate for the definitive re-ledger
- 54/54 neato rows flip (36 already evidenced in r78_neato.jsonl as 42 IDENT/5 EQUIV
  incl. rd200; the 18 absent combos verified here at the same floor).
  ACTION for re-ledger: benchmark_100seed_r78_neato has fresh positions for 16 of the
  18 absent combos; tl_resnet_2block + tl_transformer_1layer need fresh classic_neato
  bench rows (not in any r78 fresh dir).
- 4/5 fdp rows flip (already in r78_fdp.jsonl). 1 fdp row stays, with floor evidence
  above (candidate disposition: superior-distinct or named-cause floor).
Net: 58/59 close; 1 verified floor.

## Scratch
Probes: /tmp/r78_neato/probe*.py, classify.py; graph cache graphs.pkl.
