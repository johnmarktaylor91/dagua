Add MISSING reference adapters so currently-NO_REFERENCE variants become measurable. Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md for context.
NO_REFERENCE variants: classic_spectral_random_walk, classic_spectral_unnormalized (networkx supports
random-walk + unnormalized spectral/laplacian layouts -- add a reference adapter using networkx),
classic_rt_horizontal (reingold-tilford horizontal -- pair with the appropriate igraph/reference rt
variant if one exists). Edit dagua/eval/competitors/* (add adapters) + dagua/eval/variants.py (set
each variant's original_engine + MATCHED original_params). Do NOT edit dagua/layout/ops/. Then verify
they produce comparisons: python scripts/rng_match/check_engine.py classic_spectral_random_walk (etc).
Report which now have a reference + their RMSD. The fr_kk/kk_fr CHAINS have no single reference -- leave
them NO_REFERENCE (document). MATCHED PARAMS required (see [[feedback_always_parameter_match_comparisons]]).
