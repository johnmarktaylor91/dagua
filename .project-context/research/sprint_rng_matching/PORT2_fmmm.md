PORT fmmm to bit-exact (real algorithm match -- params already matched, runner honors fmmm-fixed-iterations). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md.
Pipeline: dagua/layout/ops/pipelines/fmmm.py. fmmm at 1.39. Reference: ogdf_fmmm (OGDF FM^3).
OGDF source now at ~/tools/ogdf/ (read src/ogdf/energybased/FMMMLayout). Match OGDF FMMM's
multilevel + glibc rand init + force model. Verify check_engine.py classic_fmmm_steps100. Drive <1e-7.
