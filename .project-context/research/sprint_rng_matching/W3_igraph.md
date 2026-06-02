Finish the igraph-RNG family to bit-exact. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md FULLY. You OWN: dagua/layout/ops/_igraph_rng.py,
dagua/layout/ops/pipelines/drl.py, davidson_harel.py, sugiyama.py (do all three; they share igraph RNG).
Current (matched seeds+params): drl ~1.0 but 35/42 fixtures already bit-exact (RNG mostly matched --
finish the 7 diverging cases); davidson_harel ~0.36 (simulated annealing -- match igraph's exact RNG
draw order for move proposals + acceptance); sugiyama ~0.93 (igraph layered -- deterministic-ish, match
ordering/coordinate assignment; graphviz_dot variant separate). Reimplement igraph's RNG GENERATOR from
igraph C source (rng.c + each layout's draws) -- NO replay, NO delegation. Verify each:
python scripts/rng_match/check_engine.py classic_drl_default (+ davidson_harel_rounds100, sugiyama_default).
Drive <1e-7; if a specific case is genuinely chaotic even on small graphs, DOCUMENT it precisely.
