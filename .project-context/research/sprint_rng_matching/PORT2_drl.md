RE-PORT drl (wave-1 didn't crack it). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. You OWN dagua/layout/ops/pipelines/drl.py
AND dagua/layout/ops/_igraph_rng.py. At 1.0 (matched params). Reference: igraph_drl. Needs
igraph's EXACT RNG generator (igraph uses its own RNG -- find which: Mersenne/glibc; see igraph
C source rng.c + layout_drl) + DrL's exact multilevel anneal algorithm + draw order. Reimplement
the GENERATOR from source (NO replay). Verify check_engine.py classic_drl_default. Drive <1e-7.
