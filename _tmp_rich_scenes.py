"""Scratch: engineer scenes whose cluster rows sit in smooth bands."""

from typing import Mapping, Optional, Tuple

import torch

from dagua.eval.ruler_v4._tracing import trace_subterms
from dagua.eval.ruler_v4.clusters import U25, U26, U27, U28, U29
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)
from dagua.eval.ruler_v4.surrogate.traced import build_traced_scene


def make_scene(positions, clusters, parents=None, edges=None):
    positions = torch.tensor(positions, dtype=torch.float64)
    if edges is None:
        edges = tuple((i, i + 1) for i in range(positions.shape[0] - 1))
    graph = GraphSemantics(
        tuple(f"n{i}" for i in range(positions.shape[0])),
        tuple(edges),
        clusters=clusters,
        cluster_parents=dict(parents or {}),
    )
    routes = tuple(
        Route(i, torch.stack((positions[s], positions[t]))) for i, (s, t) in enumerate(edges)
    )
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene), result
    return result.scene


def probe_facet(scene, facet, args=()):
    exact = facet(scene, *args)
    positions = scene.positions.detach().clone().requires_grad_(True)
    traced_scene = build_traced_scene(scene, positions)
    with trace_subterms() as buffer:
        traced = facet(traced_scene, *args)
    print(f"  exact value={exact.value} subterms={exact.subterms} state={exact.state}")
    for key, tensor in sorted(buffer.items()):
        if not tensor.requires_grad:
            print(f"  {key}: value={float(tensor):.6g} NO-GRAD")
            continue
        (g,) = torch.autograd.grad(tensor, positions, retain_graph=True, allow_unused=True)
        norm = float(torch.linalg.vector_norm(g)) if g is not None else None
        print(f"  {key}: value={float(tensor):.6g} gradnorm={norm}")


print("=== A: U25 spread cluster ===")
compact = [[0.1 * i, 0.13 * ((i * 7) % 5)] for i in range(9)]
spread = [[-40.0, -35.0], [42.0, -38.0], [1.0, 55.0]]
positions_a = spread + compact
scene_a = make_scene(positions_a, {"s": (0, 1, 2)})
probe_facet(scene_a, U25)
print("  intrinsic unit:", scene_a.intrinsic_unit)

print("=== B: U26 boundary margin band ===")
a = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
b = [[2.2, 0.0], [3.2, 0.0], [2.2, 1.0]]
scene_b = make_scene(a + b, {"a": (0, 1, 2), "b": (3, 4, 5)})
probe_facet(scene_b, U26)

print("=== C: U27 exile + grazing route ===")
# cluster m = 0..3 (3 exiled), foreign nodes 4,5 with edge 4-5 grazing region
positions_c = [
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 2.0],
    [7.0, 1.0],  # exiled member
    [10.2, -8.0],
    [10.2, 10.0],  # foreign pair; edge 4-5 vertical at x=10.2 grazes region
]
edges_c = ((0, 1), (1, 2), (2, 3), (4, 5))
scene_c = make_scene(positions_c, {"m": (0, 1, 2, 3)}, edges=edges_c)
probe_facet(scene_c, U27, (1,))
print("  intrinsic unit:", scene_c.intrinsic_unit)

print("=== D: U28 escape + sibling band ===")
# parent 0..9; c1=(0,1,2) spread (big radius), c2=(3,4,5) near c1
positions_d = [
    [0.0, 0.0],
    [3.0, 0.0],
    [0.0, 3.0],  # c1 moderately spread
    [9.4, 0.0],
    [10.4, 0.0],
    [9.4, 1.0],  # c2 compact, slight region overlap with c1
    [2.0, 1.5],
    [4.0, 1.0],
    [5.5, 1.5],
    [7.0, 1.0],
    [8.5, 1.5],
    [3.0, 2.5],  # rest of parent, snug
]
scene_d = make_scene(
    positions_d,
    {"c1": (0, 1, 2), "c2": (3, 4, 5), "p": tuple(range(12))},
    parents={"c1": "p", "c2": "p"},
)
from dagua.eval.ruler_v4.clusters import (
    _region_depth_outside,
    _region_relation_area,
    _regions,
)

regs = _regions(scene_d)
for nm, rg in regs.items():
    print(f"  region {nm}: radius={float(rg.radius):.4g} area={float(_region_relation_area(rg)[0]):.5g}")
a_c1, ov_c1p = _region_relation_area(regs["c1"], regs["p"])
print(f"  c1 escaped={1.0 - float(ov_c1p)/float(a_c1):.5g} depth={float(_region_depth_outside(regs['c1'], regs['p'])):.5g}")
a12, ov12 = _region_relation_area(regs["c1"], regs["c2"])
a2 = _region_relation_area(regs["c2"])[0]
print(f"  sibling overlap={float(ov12):.5g} frac={float(ov12)/min(float(a12), float(a2)):.5g}")
print("  node box half extents:", scene_d.node_boxes[0].half_extents.tolist())
probe_facet(scene_d, U28, (12,))
print("  intrinsic unit:", scene_d.intrinsic_unit)

print("=== E: U29 fragmented dense cluster ===")
hub_edges = tuple((0, i) for i in range(1, 15))
lump = lambda cx, cy: [
    [cx, cy],
    [cx + 1.0, cy],
    [cx, cy + 1.0],
    [cx + 1.0, cy + 1.0],
    [cx + 0.5, cy + 0.5],
]
positions_e = lump(0.0, 0.0) + lump(30.0, 0.0) + lump(15.0, 26.0)
scene_e = make_scene(positions_e, {"c": tuple(range(15))}, edges=hub_edges)
probe_facet(scene_e, U29)
