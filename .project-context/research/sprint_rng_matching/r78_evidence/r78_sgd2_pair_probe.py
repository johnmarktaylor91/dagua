from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("DISABLE_TQDM", "1")

import numpy as np
import torch

ROOT = Path("/home/jtaylor/.claude/worktrees/dagua-r2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.sgd2_multi_competitor import (  # noqa: E402
    _SGD2_REPO,
    SGD2MultiRef,
    _ensure_sgd2_multi_sources,
)
from dagua.eval.graphs import get_test_graphs  # noqa: E402
from dagua.layout.ops.pipelines.sgd2_multi import layout_sgd2_multi_pipeline  # noqa: E402
from dagua.layout.ops.sgd2_multi import (  # noqa: E402
    _CrossingDetector,
    _CyclicSampler,
    _prepare_state,
    _set_seed,
)
from dagua.metrics import count_crossings, sampled_stress  # noqa: E402

GRAPHS = ("real_football_115", "wide_1_100_1")
CRITERIA = {"stress": 1.0, "crossings": 0.5}
REF_VARIANT = {
    "criteria_weights": {"stress": 1.0, "crossings": 0.5},
    "max_iter": 2000,
    "optimizer_kwargs": {"lr": 0.01},
    "grad_clamp": 5.0,
}


def _sha_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=int).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metrics(pos: torch.Tensor, graph: Any, seed: int) -> dict[str, Any]:
    stress = sampled_stress(pos, graph.edge_index, graph.num_nodes, n_sources=200, n_targets=1000)
    return {
        "stress": float(stress["sampled_stress"]),
        "crossings": int(count_crossings(pos, graph.edge_index, seed=seed)),
    }


def _quiet_upstream_modules() -> None:
    repo = str(_SGD2_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import gd2  # type: ignore[import-untyped]

    class _QuietTqdm:
        def __init__(self, iterable, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs):
            return None

    gd2.tqdm = _QuietTqdm


def _make_nx_graph(graph: Any) -> Any:
    import networkx as nx

    ei = graph.edge_index.cpu().numpy()
    gnx = nx.Graph()
    gnx.add_nodes_from(range(graph.num_nodes))
    for i in range(ei.shape[1]):
        s, t = int(ei[0, i]), int(ei[1, i])
        if s != t:
            gnx.add_edge(s, t)
    return gnx


def _ref_first_batches(graph: Any, seed: int) -> dict[str, Any]:
    _ensure_sgd2_multi_sources()
    _quiet_upstream_modules()
    from gd2 import GD2  # type: ignore[import-untyped]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gd2 = GD2(_make_nx_graph(graph))
    gd2.sample_sizes = {"stress": 128, "crossings": 128}
    gd2.init_sampler({"stress": 1.0, "crossings": 0.5})
    stress = gd2.sample("stress")
    crossings = gd2.sample("crossings")
    crossings_stacked = torch.stack(crossings, 1)
    return {
        "init_hash": _sha_tensor(gd2.pos.detach()),
        "stress_shape": list(stress.shape),
        "stress_head": stress[:8].detach().cpu().tolist(),
        "stress_hash": _sha_tensor(stress.to(torch.long)),
        "crossings_shape": list(crossings_stacked.shape),
        "crossings_head": crossings_stacked[:8].detach().cpu().tolist(),
        "crossings_hash": _sha_tensor(crossings_stacked.to(torch.long)),
        "non_incident_count": len(gd2.non_incident_edge_pairs),
    }


def _native_first_batches(graph: Any, seed: int) -> dict[str, Any]:
    device = torch.device("cpu")
    _set_seed(seed)
    prepared = _prepare_state(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        device=device,
        needs_distances=True,
        needs_incident_edge_pairs=False,
        needs_non_incident_edge_pairs=True,
        edge_weights=None,
    )
    pos = torch.randn((graph.num_nodes, 2), device=device, dtype=torch.float32) * (
        float(graph.num_nodes) ** 0.5
    )
    _ = _CrossingDetector().to(device=device)
    stress_sampler = _CyclicSampler(prepared.stress_pairs.shape[1], device)  # type: ignore[union-attr]
    crossing_sampler = _CyclicSampler(prepared.non_incident_edge_pairs.shape[1], device)  # type: ignore[union-attr]
    stress_idx = stress_sampler.sample(128)
    crossing_idx = crossing_sampler.sample(128)
    stress = prepared.stress_pairs[:, stress_idx].transpose(0, 1)  # type: ignore[union-attr]
    crossings = prepared.non_incident_edge_pairs[:, crossing_idx].transpose(0, 1)  # type: ignore[union-attr]
    return {
        "init_hash": _sha_tensor(pos.detach()),
        "stress_shape": list(stress.shape),
        "stress_head": stress[:8].detach().cpu().tolist(),
        "stress_hash": _sha_tensor(stress.to(torch.long)),
        "crossings_shape": list(crossings.shape),
        "crossings_head": crossings[:8].detach().cpu().tolist(),
        "crossings_hash": _sha_tensor(crossings.to(torch.long)),
        "non_incident_count": int(prepared.non_incident_edge_pairs.shape[1]),  # type: ignore[union-attr]
    }


def _run_reference(graph: Any, seed: int) -> torch.Tensor:
    _quiet_upstream_modules()
    result = SGD2MultiRef().layout_with_variant(graph, seed=seed, variant_params=REF_VARIANT)
    if result.error is not None or result.pos is None:
        raise RuntimeError(f"reference failed: {result.error}")
    return result.pos.detach().cpu().to(torch.float32)


def _run_native(graph: Any, seed: int) -> torch.Tensor:
    return (
        layout_sgd2_multi_pipeline(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            seed=seed,
            steps=2000,
            criteria=CRITERIA,
            lr=0.01,
            grad_clamp=5.0,
            fidelity_mode=True,
        )
        .detach()
        .cpu()
        .to(torch.float32)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/r78_r2/sgd2_pair_results.jsonl")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--stop", type=int, default=200)
    parser.add_argument("--max-seeds", type=int, default=None)
    args = parser.parse_args()

    graph_map = {item.name: item.graph for item in get_test_graphs() if item.name in GRAPHS}
    missing = sorted(set(GRAPHS) - set(graph_map))
    if missing:
        raise RuntimeError(f"missing graphs: {missing}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, int]] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            completed.add((item["graph"], int(item["seed"])))

    seeds = list(range(args.start, args.stop))
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]

    with out.open("a", encoding="utf-8") as fh:
        for graph_name in GRAPHS:
            graph = graph_map[graph_name]
            for seed in seeds:
                if (graph_name, seed) in completed:
                    continue
                t0 = time.perf_counter()
                print(
                    json.dumps({"graph": graph_name, "seed": seed, "phase": "first_batches_start"}),
                    flush=True,
                )
                ref_batches = _ref_first_batches(graph, seed)
                native_batches = _native_first_batches(graph, seed)
                print(
                    json.dumps({"graph": graph_name, "seed": seed, "phase": "native_start"}),
                    flush=True,
                )
                native_pos = _run_native(graph, seed)
                print(
                    json.dumps({"graph": graph_name, "seed": seed, "phase": "reference_start"}),
                    flush=True,
                )
                ref_pos = _run_reference(graph, seed)
                print(
                    json.dumps({"graph": graph_name, "seed": seed, "phase": "metrics_start"}),
                    flush=True,
                )
                row = {
                    "graph": graph_name,
                    "seed": seed,
                    "native_hash": _sha_tensor(native_pos),
                    "reference_hash": _sha_tensor(ref_pos),
                    "hash_equal": _sha_tensor(native_pos) == _sha_tensor(ref_pos),
                    "max_abs_delta": float((native_pos - ref_pos).abs().max().item()),
                    "rms_delta": float(torch.sqrt((native_pos - ref_pos).square().mean()).item()),
                    "native_metrics": _metrics(native_pos, graph, seed),
                    "reference_metrics": _metrics(ref_pos, graph, seed),
                    "first_batches": {"native": native_batches, "reference": ref_batches},
                    "first_batch_equal": {
                        "init": native_batches["init_hash"] == ref_batches["init_hash"],
                        "stress": native_batches["stress_hash"] == ref_batches["stress_hash"],
                        "crossings": native_batches["crossings_hash"]
                        == ref_batches["crossings_hash"],
                    },
                    "elapsed_seconds": time.perf_counter() - t0,
                }
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "graph",
                                "seed",
                                "hash_equal",
                                "max_abs_delta",
                                "rms_delta",
                                "first_batch_equal",
                                "elapsed_seconds",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
