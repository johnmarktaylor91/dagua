FIX neulay ERROR (6 variants). Read .project-context/research/sprint_rng_matching/PORTING_PROTOCOL.md for context. neulay ref available()=True and imports
fine, but layout RUNTIME-errors on all graphs (likely GNN-train timeout or an exception). Run
`python scripts/rng_match/check_engine.py classic_neulay_default`, read the actual traceback/
timeout, and FIX so it produces comparisons on SMALL graphs (small graphs should train fast;
install any missing user-space dep e.g. torch-geometric if needed; fix the exception). Edit
dagua/eval/competitors/neulay_* and/or dagua/layout/ops/pipelines/neulay.py as needed (NOT
variants.py). Report root cause + fix + resulting RMSD.
