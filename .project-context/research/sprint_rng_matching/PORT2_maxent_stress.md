PORT maxent_stress to bit-exact. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline: dagua/layout/ops/pipelines/maxent_stress.py.
At 1.16 (matched params). Reference: ogdf_stress (OGDF StressMinimization). OGDF src ~/tools/ogdf/
(src/ogdf/energybased/StressMinimization). Match its init + majorization arithmetic order.
Verify check_engine.py classic_maxent_stress_default. Drive <1e-7.
