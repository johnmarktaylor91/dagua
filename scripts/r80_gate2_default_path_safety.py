"""r80-S4 Stage 4 gate 2: default-path safety for directed graphs.

For five DIRECTED corpus graphs (including the transformer_layer landmine
and dependency_graph_100), compute layout positions with the benchmark's
exact configuration and assert they are BIT-IDENTICAL to positions computed
by the pre-branch code (r79/native at the branch point, imported from the
read-only main worktree via a child process with PYTHONPATH). The portfolio
route must not fire on any of them.

Also proves transformer_layer now classifies as semantically DIRECTED
(it was inferred undirected by the old deep-layering rule).

Usage
-----
    .venv/bin/python scripts/r80_gate2_default_path_safety.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

P2_ROOT = Path(__file__).resolve().parents[1]
MAIN_WORKTREE = Path("/home/jtaylor/.claude/worktrees/dagua-native")

DIRECTED_GATE_GRAPHS = [
    "transformer_layer",
    "dependency_graph_100",
    "asymmetric_hourglass_hub",
    "org_chart_deep",
    "random_dag_50",
]

_CHILD_SCRIPT = r"""
import sys
import torch

out_dir = sys.argv[1]
names = sys.argv[2].split(",")

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.layout import layout

graphs = {tg.name: tg for tg in get_test_graphs()}
for name in names:
    tg = graphs[name]
    tg.graph.compute_node_sizes()
    config = LayoutConfig(device="cpu", verbose=False, seed=42)
    pos = layout(tg.graph, config).detach().cpu().to(dtype=torch.float32)
    torch.save(pos, f"{out_dir}/{name}.pt")
    print(f"saved {name} {tuple(pos.shape)}", flush=True)
print("CHILD_OK", flush=True)
"""


def _compute_positions(source_root: Path, out_dir: Path) -> None:
    """Compute gate-graph positions using the code at ``source_root``.

    Parameters
    ----------
    source_root : Path
        Repo root whose ``dagua`` package should be imported.
    out_dir : Path
        Directory receiving one ``<graph>.pt`` per gate graph.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # never dirty the read-only worktree
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, str(out_dir), ",".join(DIRECTED_GATE_GRAPHS)],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0 or "CHILD_OK" not in result.stdout:
        raise RuntimeError(
            f"position child failed for {source_root}:\n"
            f"stdout tail: {result.stdout[-2000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


def main() -> None:
    # Part A: transformer_layer must classify directed and route non-portfolio.
    from dagua.config import LayoutConfig
    from dagua.eval.graphs import get_test_graphs
    from dagua.layout.graph_classify import classify_graph
    from dagua.layout.ops.pipelines.dagua_native import _choose_native_pipeline

    corpus = {tg.name: tg for tg in get_test_graphs()}
    failures: list[str] = []

    for name in DIRECTED_GATE_GRAPHS:
        if name not in corpus:
            failures.append(f"{name}: missing from corpus")
            continue
        graph = corpus[name].graph
        structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
        route = _choose_native_pipeline(
            structure=structure, config=LayoutConfig(seed=42, device="cpu")
        )
        directed = structure.is_semantically_directed
        print(f"{name:30s} directed={directed} route={route}", flush=True)
        if directed is not True:
            failures.append(f"{name}: classified undirected (must be directed)")
        if route == "undirected_portfolio":
            failures.append(f"{name}: routed to undirected_portfolio (must not fire)")

    # Part B: bit-identical positions before (main worktree) vs after (p2).
    with tempfile.TemporaryDirectory(prefix="r80_gate2_") as tmp:
        before_dir = Path(tmp) / "before"
        after_dir = Path(tmp) / "after"
        before_dir.mkdir()
        after_dir.mkdir()
        print("computing BEFORE positions (main worktree code)...", flush=True)
        _compute_positions(MAIN_WORKTREE, before_dir)
        print("computing AFTER positions (this branch)...", flush=True)
        _compute_positions(P2_ROOT, after_dir)

        for name in DIRECTED_GATE_GRAPHS:
            before = torch.load(before_dir / f"{name}.pt", map_location="cpu")
            after = torch.load(after_dir / f"{name}.pt", map_location="cpu")
            identical = before.shape == after.shape and bool(torch.equal(before, after))
            max_delta = (
                float((before - after).abs().max().item())
                if before.shape == after.shape
                else float("nan")
            )
            print(f"{name:30s} bit-identical={identical} max_delta={max_delta}", flush=True)
            if not identical:
                failures.append(f"{name}: positions changed (max_delta={max_delta})")

    if failures:
        print("\nGATE 2 FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nGATE 2 PASSED: all directed gate graphs bit-identical and non-portfolio.")


if __name__ == "__main__":
    main()
