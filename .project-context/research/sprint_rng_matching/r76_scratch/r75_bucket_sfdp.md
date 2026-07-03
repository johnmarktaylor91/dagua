# BUCKET: sfdp (126 divergent combos -- largest bucket, likely cheapest big win)

Target list: r75_targets_sfdp.json (BUCKET=sfdp). Reference: graphviz sfdp (seedable base
graphviz_sfdp; adapters in dagua/eval/competitors/graphviz_competitor.py). Dagua side:
dagua/layout/ops/pipelines/sfdp.py + dagua/layout/ops/sfdp.py.

Cluster structure you must explain (from the rescore rows):
- 83/126 combos fail stress by <=1% relative -- HAIRLINE, systematic, sign varies.
- Median stress excess is NEGATIVE (-8.2%): dagua is BETTER than reference on the median failing
  combo. 51 combos are dagua-better on every failing leg; 34 all-worse; 41 mixed.
- Crossings co-fail on 74 combos (45 of those hairline <=1%).
- A tail of large gaps (7 combos >2x stress excess, 5 combos >5x crossings excess).

Ranked hypotheses to CONFIRM or KILL (add your own):
H1. REFERENCE POST-PROCESSING IN EXTRACTED POSITIONS. graphviz sfdp by default may apply overlap
    removal (prism / -x behavior) and other post-passes (see
    _references/graphviz/lib/sfdpgen/{sfdpinit.c,post_process.c,spring_electrical.c},
    lib/neatogen/overlap.c). If the adapter extracts positions AFTER overlap removal while dagua's
    pipeline stops at the spring-electrical solution, dagua would systematically show LOWER stress
    (overlap removal distorts the energy optimum). Check exactly which flags/attrs the adapter
    passes (overlap=?, maxiter, K, smoothing), and what post-processing runs before positions are
    read back. This is the leading suspect for the dagua-better cluster.
H2. RNG STREAM MISMATCH. graphviz uses rejection-sampling for random doubles (gv_random /
    drand48-style paths) while dagua ops/sfdp.py:247-253 uses a raw modulo LCG mapping. If the
    stream diverges, initial placements differ per seed -> distributional-but-not-bit-exact.
    Determine: does graphviz sfdp's default pipeline even consume RNG beyond initial placement?
    Could matching the stream make combos BIT-EXACT (the ideal outcome)?
H3. Iteration/termination mismatch (maxiter, convergence epsilon, cooling schedule) between the
    adapter invocation and dagua's pipeline params.
H4. Multilevel coarsening differences (Multilevel.c) -- coarsening graph match, prolongation.
For each hypothesis give the DECISIVE cheap experiment (e.g., rerun ONE combo through the
benchmark path with overlap removal disabled in the reference and see if stress gap closes).

Also answer explicitly:
- Which of the 126 would a single fix (H1-style adapter/params correction, if confirmed) convert
  to quality-identical or bit-exact? Adapter-side fixes change the COMPARISON, not dagua --
  flag them clearly as such (they need the sprint lead's sign-off on whether the reference
  invocation or the dagua pipeline is the thing to change).
- Is the r74 p_neg2 clamp (commit 6f8cff5) interacting with any of these combos?
