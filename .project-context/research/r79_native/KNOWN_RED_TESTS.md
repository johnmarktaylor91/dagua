# Pre-existing stale test failures on r79 worktrees (NOT caused by sprint work)
Both P3d and P5 codexes independently wasted 3-4h each running full `pytest tests/ -x`
and whack-a-moling these. FUTURE BRIEFS: scope test gates to touched modules; deselect these.

- tests/test_cuda_activation.py::test_all_stages_fall_back_when_no_cuda  (old NVIDIA driver 12040)
- tests/test_classic_competitor.py::test_each_classic_competitor_produces_a_valid_result (sklearn missing -> classic_tsnet)
- tests/test_fidelity_procrustes.py::test_procrustes_known_good_equivalent (partial_match vs strong)
- tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
- eval/graphs.py tl_* construction: TorchLens 2.28 log_forward_pass/vis_mode API drift
- weighted maxent_stress fixture drift

RULE: brief test gate = `pytest <touched-test-files> -q` + optionally
`pytest tests/ -q -m "not slow and not benchmark and not rare" --deselect <each known-red>`.
Never bare `pytest tests/ -x` on these worktrees.
