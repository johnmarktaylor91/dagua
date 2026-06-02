Finish fr + fa2_linlog to bit-exact (both CLOSE). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. You OWN dagua/layout/ops/pipelines/fr.py AND fa2.py.
Current: fr_steps100 CLOSE 1.86e-7 (tiny push -- likely one arithmetic-order/last-step detail), fr_steps200
1.9e-3, fr_steps500 2.7e-2 (longer runs diverge -- check if chaotic or a step-count param), fa2_linlog 2.1e-3
(the linlog mode of fa2 -- match the linlog attraction/repulsion exponents + draw order). Verify
check_engine.py classic_fr_steps100 (+ steps200/500, classic_fa2_linlog). Drive <1e-7; document chaotic walls.
