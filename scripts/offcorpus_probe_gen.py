"""R0.6 off-corpus probe generator for the behind-row families.

Generates seeded, regenerable probe graphs family-matched to the current
behind rows (nested-directed-cluster, clustered-medium, deep-skinny-DAG) plus
the recently-flipped geometric-random family, for validating borrow-best
structural predicates OFF-corpus before banking a change.

NON-HOLDOUT DECLARATION
-----------------------
These probes are explicitly NOT a holdout. They may be inspected, scored,
iterated against, and regenerated freely during tuning. GLaDOS remains the
only sacred holdout; nothing here touches it. Probes exist to answer "does
this structural predicate generalize beyond the exact corpus row?" -- a probe
pass is necessary-but-not-sufficient evidence, never a substitute for the
corpus gate.

Probe names are prefixed ``probe_`` so they can never collide with corpus
graph names, and every probe JSON embeds its family, seed, parameters, and
``non_holdout: true``.

Usage
-----
python scripts/offcorpus_probe_gen.py [--family all] [--count 3] \\
    [--seed-start 1000] [--out-dir .../roundloop/offcorpus_probes]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import roundloop_common as rl  # noqa: E402

from dagua.eval.graphs import (  # noqa: E402
    TestGraph,
    make_clustered_medium,
    make_dependency_graph,
    make_random_geometric,
)
from dagua.graph import DaguaGraph  # noqa: E402

PROBE_SCHEMA = "offcorpus-probe-v1"
DEFAULT_PROBE_DIR = rl.ROUNDLOOP_DIR / "offcorpus_probes"
FAMILIES: Tuple[str, ...] = (
    "nested_directed_cluster",
    "clustered_medium",
    "deep_skinny_dag",
    "geometric_random",
)


def _nested_directed_cluster(seed: int) -> Tuple[DaguaGraph, Dict[str, Any], set]:
    """Build a nested directed-cluster probe (mixed_direct_leaf family).

    A parent cluster mixes direct leaves with child clusters; cross edges tie
    direct leaves to child-cluster members and an external tail cluster
    receives the merged flow -- the structural signature of
    ``r8_nested_mixed_direct_leaf`` / ``nested_shallow_enc_dec``, with sizes
    and cross-wiring randomized per seed.

    Parameters
    ----------
    seed : int
        Probe seed.

    Returns
    -------
    Tuple[DaguaGraph, Dict[str, Any], set]
        Graph, generation params, and tags.
    """
    rng = random.Random(seed)
    n_direct = rng.randint(4, 8)
    n_children = rng.randint(2, 3)
    child_sizes = [rng.randint(4, 8) for _ in range(n_children)]
    n_tail = rng.randint(3, 5)
    graph = DaguaGraph()
    prefix = f"probe_ndc_{seed}"

    direct = [f"direct_{index}" for index in range(n_direct)]
    children: List[List[str]] = [
        [f"child{child}_{index}" for index in range(size)] for child, size in enumerate(child_sizes)
    ]
    tail = [f"tail_{index}" for index in range(n_tail)]
    for node in direct + [node for child in children for node in child] + tail:
        graph.add_node(node, label=node)
    for nodes in (direct, *children, tail):
        for source, target in zip(nodes, nodes[1:]):
            graph.add_edge(source, target)
    # Cross edges: each direct leaf feeds a random member of a random child.
    for node in direct:
        child_pick = children[rng.randrange(n_children)]
        graph.add_edge(node, child_pick[rng.randrange(len(child_pick))])
    # Child tails merge into the external tail head.
    for child in children:
        graph.add_edge(child[-1], tail[0])
    members = direct + [node for child in children for node in child]
    graph.add_cluster(f"{prefix}_parent", members, label="Mixed Parent")
    graph.add_cluster(
        f"{prefix}_direct_band", direct, label="Direct Leaves", parent=f"{prefix}_parent"
    )
    for index, child in enumerate(children):
        graph.add_cluster(
            f"{prefix}_child_{index}", child, label=f"Child {index}", parent=f"{prefix}_parent"
        )
    graph.add_cluster(f"{prefix}_tail", tail, label="External Tail")
    params = {
        "n_direct": n_direct,
        "child_sizes": child_sizes,
        "n_tail": n_tail,
    }
    return graph, params, {"nested-depth", "fanout", "directed"}


def _clustered_medium(seed: int) -> Tuple[DaguaGraph, Dict[str, Any], set]:
    """Build a clustered-medium probe (clustered_medium_5x20 family).

    Parameters
    ----------
    seed : int
        Probe seed.

    Returns
    -------
    Tuple[DaguaGraph, Dict[str, Any], set]
        Graph, generation params, and tags.
    """
    rng = random.Random(seed)
    n_clusters = rng.randint(4, 6)
    nodes_per_cluster = rng.randint(15, 25)
    inter_density = round(rng.uniform(0.04, 0.08), 3)
    graph = make_clustered_medium(
        n_clusters=n_clusters,
        nodes_per_cluster=nodes_per_cluster,
        inter_density=inter_density,
        seed=seed,
    )
    params: Dict[str, Any] = {
        "n_clusters": n_clusters,
        "nodes_per_cluster": nodes_per_cluster,
        "inter_density": inter_density,
        "seed": seed,
    }
    return graph, params, {"clustered", "nested-shallow"}


def _deep_skinny_dag(seed: int) -> Tuple[DaguaGraph, Dict[str, Any], set]:
    """Build a deep-skinny dependency-DAG probe (dependency_500 family).

    Parameters
    ----------
    seed : int
        Probe seed.

    Returns
    -------
    Tuple[DaguaGraph, Dict[str, Any], set]
        Graph, generation params, and tags.
    """
    rng = random.Random(seed)
    n = rng.randint(400, 600)
    n_core = rng.randint(8, 12)
    graph = make_dependency_graph(n=n, n_core=n_core, seed=seed)
    params: Dict[str, Any] = {"n": n, "n_core": n_core, "seed": seed}
    return graph, params, {"dependency", "scale-free", "clustered"}


def _geometric_random(seed: int) -> Tuple[DaguaGraph, Dict[str, Any], set]:
    """Build a geometric-random probe (rgg family).

    Parameters
    ----------
    seed : int
        Probe seed.

    Returns
    -------
    Tuple[DaguaGraph, Dict[str, Any], set]
        Graph, generation params, and tags.
    """
    rng = random.Random(seed)
    n = rng.randint(80, 500)
    radius = round(rng.uniform(0.10, 0.20), 3)
    test_graph = make_random_geometric(n=n, radius=radius, seed=seed)
    params: Dict[str, Any] = {"n": n, "radius": radius, "seed": seed}
    return test_graph.graph, params, {"geometric", "spatial", "random", "undirected"}


_BUILDERS: Dict[str, Callable[[int], Tuple[DaguaGraph, Dict[str, Any], set]]] = {
    "nested_directed_cluster": _nested_directed_cluster,
    "clustered_medium": _clustered_medium,
    "deep_skinny_dag": _deep_skinny_dag,
    "geometric_random": _geometric_random,
}


def generate_probe(family: str, seed: int) -> TestGraph:
    """Generate one seeded probe graph.

    Parameters
    ----------
    family : str
        One of ``FAMILIES``.
    seed : int
        Probe seed (determines every random choice).

    Returns
    -------
    TestGraph
        Probe graph named ``probe_<family>_<seed>`` with family tags plus
        ``offcorpus_probe``, and the semantic-direction flag set from tags.
    """
    if family not in _BUILDERS:
        raise ValueError(f"unknown probe family: {family!r}; expected one of {FAMILIES}")
    graph, params, tags = _BUILDERS[family](seed)
    test_graph = TestGraph(
        name=f"probe_{family}_{seed}",
        graph=graph,
        tags={*tags, "offcorpus_probe"},
        description=(
            f"Off-corpus {family} probe (seed {seed}, params {params}). "
            "NON-HOLDOUT: free to inspect and tune against."
        ),
        source="offcorpus_probe_gen",
        expected_challenges="Family-matched off-corpus generalization check",
    )
    test_graph.graph.is_semantically_directed = "undirected" not in test_graph.tags
    return test_graph


def generate_probes(family: str, count: int, seed_start: int) -> List[TestGraph]:
    """Generate a batch of probes with consecutive seeds.

    Parameters
    ----------
    family : str
        Probe family.
    count : int
        Number of probes.
    seed_start : int
        First seed.

    Returns
    -------
    List[TestGraph]
        Generated probes.
    """
    return [generate_probe(family, seed_start + index) for index in range(count)]


def probe_payload(test_graph: TestGraph, family: str, seed: int) -> Dict[str, Any]:
    """Serialize a probe with provenance.

    Parameters
    ----------
    test_graph : TestGraph
        Probe graph.
    family : str
        Probe family.
    seed : int
        Probe seed.

    Returns
    -------
    Dict[str, Any]
        JSON payload (schema, provenance, non-holdout marker, graph JSON).
    """
    return {
        "schema": PROBE_SCHEMA,
        "name": test_graph.name,
        "family": family,
        "seed": seed,
        "tags": sorted(test_graph.tags),
        "is_semantically_directed": bool(test_graph.graph.is_semantically_directed),
        "description": test_graph.description,
        "non_holdout": True,
        "generated_at": rl.utc_now_iso(),
        "git_sha": rl.git_sha(SCRIPTS_DIR.parent),
        "graph": test_graph.graph.to_json(),
    }


def load_probe(path: Path) -> TestGraph:
    """Load a probe JSON back into a scoreable TestGraph.

    Node sizes are computed on load (the ``build_graph_map`` contract), so
    ``score_position``'s sizeless-graph tripwire passes.

    Parameters
    ----------
    path : Path
        Probe JSON path.

    Returns
    -------
    TestGraph
        Reconstructed probe.
    """
    payload = json.loads(path.read_text())
    if payload.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"unexpected probe schema in {path}: {payload.get('schema')}")
    graph = DaguaGraph.from_json(
        payload["graph"],
        is_semantically_directed=bool(payload["is_semantically_directed"]),
    )
    graph.compute_node_sizes()
    return TestGraph(
        name=str(payload["name"]),
        graph=graph,
        tags=set(payload["tags"]),
        description=str(payload.get("description", "")),
        source="offcorpus_probe_gen",
        expected_challenges="Family-matched off-corpus generalization check",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    parser.add_argument("--count", type=int, default=3, help="Probes per family.")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROBE_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate and write probe JSON files.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    families = list(FAMILIES) if args.family == "all" else [args.family]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for family in families:
        for index in range(args.count):
            seed = args.seed_start + index
            probe = generate_probe(family, seed)
            payload = probe_payload(probe, family, seed)
            path = args.out_dir / f"{probe.name}.json"
            path.write_text(json.dumps(payload, indent=1))
            written.append(f"{probe.name} (n={probe.graph.num_nodes}, e={probe.graph.num_edges})")
    print(f"[probes] wrote {len(written)} probes to {args.out_dir}:", flush=True)
    for line in written:
        print(f"  {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
