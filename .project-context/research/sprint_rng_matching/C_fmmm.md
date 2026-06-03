Close fmmm (deterministic packing residual). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md. Pipeline: dagua/layout/ops/pipelines/fmmm.py.
Current fmmm_steps10/100/200 ~0.011-0.021 (force trajectory matches 35/42; residual = 1-unit
integer-coordinate drift after OGDF component packing/final floor on symmetric layouts). Read OGDF
~/tools/ogdf/ FMMMLayout final packing/postprocessing (component arrangement order, coordinate
flooring/rounding, orientation) and replicate the EXACT final-coordinate step. (fmmm_graphviz_fdp_fidelity
1.39 is a separate graphviz-fdp path -- lower priority, attempt only if time.) Verify check_engine.py
classic_fmmm_steps100. Drive <1e-7; document if the orientation ambiguity is irreducible.
