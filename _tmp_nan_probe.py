import sys

sys.path.insert(0, "outputs/ruler_v4/p3/gate/probes_fable")
import torch
from traced_baseline import build_scene, complete_table, profiles

from dagua.eval.ruler_v4.surrogate.traced import score_scene_soft

scene = build_scene()
traced = score_scene_soft(scene, complete_table(), profiles())
for subterm_id in sorted(traced.bound_subterms):
    tensor = traced.traced_subterms[subterm_id]
    if not tensor.requires_grad:
        print(f"{subterm_id:16s} DETACHED")
        continue
    (gradient,) = torch.autograd.grad(
        tensor, traced.positions, retain_graph=True, allow_unused=True
    )
    if gradient is None:
        print(f"{subterm_id:16s} NO-GRAPH")
        continue
    norm = float(torch.linalg.vector_norm(gradient))
    has_nan = bool(torch.isnan(gradient).any())
    print(f"{subterm_id:16s} norm={norm:.6g} nan={has_nan} value={float(tensor):.9g}")
