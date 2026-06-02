Finish fmmm + maxent_stress variants to bit-exact. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. You OWN dagua/layout/ops/pipelines/fmmm.py
AND maxent_stress.py. Current: fmmm_steps10/100/200 close (~0.011-0.021; residual = 1-unit integer-coord
drift after OGDF component packing/final floor -- match OGDF's final coordinate rounding/packing order);
fmmm_graphviz_fdp_fidelity 1.39 (separate, graphviz-fdp path); maxent_stress_default already BIT_EXACT but
alpha2/entropy variants ~1.16 (handle their alpha/use_entropy params correctly). OGDF src ~/tools/ogdf/.
Verify check_engine.py classic_fmmm_steps100 + classic_maxent_stress_alpha2. Drive <1e-7. Document any wall.
