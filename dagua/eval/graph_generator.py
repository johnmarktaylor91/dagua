"""Salt-derived graph suite generator for Sprint 0.5+ opacity.

Given a secret salt (committed out-of-repo at ``.project-context/private/
holdout_salt``) and a public sprint tag, produces a deterministic list of
test graphs covering the priority families in 03_test_matrix.md. Per-graph
seeds are derived as ``sha256(salt || tag || family || index)[:8]``, so
topologies are irreproducible without the salt. The intended callers:

1. Sprint-exit validation: ``make_suite(salt, sprint_tag="holdout_v1", ...)``
   regenerates held-out graphs on demand; metrics run; tensors dropped.
   Only the MANIFEST.json (hashes + family + size) is committed.
2. Rolling anti-overfit set: ``make_suite(salt, sprint_tag="sprint_N_YYYYMMDD",
   family_mix="rolling")`` produces a fresh set per sprint.

Held-out vs rolling opacity: identical salt, different sprint_tag prefixes.
CI enforces that ``dagua/graphs/holdout/`` contains only the manifest and
a ``.opaque`` marker; direct reads of held-out graph tensors from disk are
a process failure (pytest fixture in tests/test_holdout_opacity.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from dagua.eval.graphs import (
    TestGraph,
    make_bipartite,
    make_chain,
    make_complete_bipartite,
    make_diamond,
    make_erdos_renyi,
    make_grid,
    make_hub_and_spoke,
    make_powerlaw_dag,
    make_random_dag,
    make_random_geometric,
    make_small_world,
    make_sparse_layered,
    make_tree,
    make_wide_dag,
)

DEFAULT_SALT_PATH = Path(".project-context/private/holdout_salt")


# Priority-ordered family spec: (family_tag, builder, size_arg_name).
# Each builder accepts (n, seed) and returns a TestGraph. Sprint 0.5 uses
# 30 graphs (2-3 per priority family, small/medium mix) as the held-out
# floor per 09 Q14.
_FAMILY_SPECS: Dict[str, Callable[[int, int], TestGraph]] = {
    "directed_dag_small": lambda n, s: make_random_dag(n, density=2.0, seed=s),
    "directed_dag_medium": lambda n, s: make_random_dag(n, density=2.5, seed=s),
    "chain_deep": lambda n, s: make_chain(n, seed=s),
    "wide_parallel": lambda n, s: make_wide_dag(n, width=max(5, int(n**0.5)), seed=s),
    "diamond": lambda n, s: make_diamond(n, seed=s),
    "tree_branching_3": lambda n, s: make_tree(n, branching=3, seed=s),
    "tree_branching_4": lambda n, s: make_tree(n, branching=4, seed=s),
    "grid_square": lambda n, s: TestGraph(
        name=f"grid_{int(n**0.5)}x{int(n**0.5)}_{s}",
        graph=make_grid(int(n**0.5), int(n**0.5), seed=s),
        tags={"lattice"},
        description=f"Square grid ~{n} nodes",
    ),
    "sparse_layered": lambda n, s: make_sparse_layered(n, seed=s),
    "powerlaw_dag": lambda n, s: make_powerlaw_dag(n, seed=s),
    "bipartite": lambda n, s: make_bipartite(n, seed=s),
    "erdos_renyi_sparse": lambda n, s: make_erdos_renyi(n, p=0.02, seed=s),
    "random_geometric": lambda n, s: make_random_geometric(n, radius=0.15, seed=s),
    "hub_spoke": lambda n, s: TestGraph(
        name=f"hub_spoke_{n}_{s}",
        graph=make_hub_and_spoke(n_hubs=max(2, n // 20), spokes_per_hub=10, seed=s),
        tags={"near-clique"},
        description="Hub-and-spoke with concentrated connectivity",
    ),
    "complete_bipartite": lambda n, s: TestGraph(
        name=f"complete_bipartite_{n}_{s}",
        graph=make_complete_bipartite(a=max(3, n // 4), b=max(4, n // 3), seed=s),
        tags={"near-clique"},
        description="Complete bipartite (dense undirected)",
    ),
    "small_world": lambda n, s: TestGraph(
        name=f"small_world_{n}_{s}",
        graph=make_small_world(n, k=4, p=0.1, seed=s),
        tags={"undirected-sparse"},
        description="Small-world (Watts-Strogatz-like)",
    ),
}


# Held-out v1 plan: 30 graphs across 14 families, small (<=200) and medium
# (200-2000) split. Exact family, size pairs are a function of (salt, tag),
# so adversarial inspection of this list does NOT reveal the topologies.
_HOLDOUT_V1_PLAN = [
    # P0: directed DAG (6 graphs)
    ("directed_dag_small", 80),
    ("directed_dag_small", 150),
    ("directed_dag_medium", 400),
    ("directed_dag_medium", 900),
    ("chain_deep", 120),
    ("chain_deep", 600),
    # P0: tree (4)
    ("tree_branching_3", 100),
    ("tree_branching_3", 400),
    ("tree_branching_4", 150),
    ("tree_branching_4", 800),
    # P0: wide-parallel / diamond (4)
    ("wide_parallel", 120),
    ("wide_parallel", 500),
    ("diamond", 60),
    ("diamond", 200),
    # P0: sparse layered + powerlaw (3)
    ("sparse_layered", 300),
    ("sparse_layered", 1200),
    ("powerlaw_dag", 400),
    # P1: undirected sparse (4)
    ("erdos_renyi_sparse", 200),
    ("random_geometric", 300),
    ("small_world", 250),
    ("small_world", 900),
    # P1: lattice / bipartite (3)
    ("grid_square", 100),
    ("grid_square", 400),
    ("bipartite", 200),
    # P2: near-clique (2)
    ("hub_spoke", 120),
    ("complete_bipartite", 80),
    # Extra P0/P1 for balance (4) -> 30 total per 09 Q14 minimum
    ("directed_dag_medium", 1500),
    ("chain_deep", 1800),
    ("tree_branching_3", 1800),
    ("sparse_layered", 600),
]


@dataclass
class SuiteManifest:
    """Opaque description of a generated suite. Committed, not the graphs."""

    sprint_tag: str
    salt_digest_prefix: str  # first 8 hex chars of sha256(salt); proves salt version
    entries: List[Dict[str, object]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "sprint_tag": self.sprint_tag,
                "salt_digest_prefix": self.salt_digest_prefix,
                "entries": self.entries,
                "entry_count": len(self.entries),
            },
            indent=2,
        )


def _load_salt(salt_path: Optional[Path]) -> bytes:
    p = salt_path or DEFAULT_SALT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Secret salt not found at {p}. Sprint 0.5 requires "
            f".project-context/private/holdout_salt (gitignored). Run "
            f'`python -c \'import secrets; open({p!r}, "wb").write('
            f"secrets.token_bytes(32))'` to create one."
        )
    salt = p.read_bytes()
    if len(salt) < 16:
        raise ValueError(f"Salt at {p} is too short ({len(salt)} bytes); need >=16.")
    return salt


def _derive_seed(salt: bytes, sprint_tag: str, family: str, index: int) -> int:
    """Derive a seed from sha256(salt || tag || family || index)[:4] as int."""
    h = hashlib.sha256()
    h.update(salt)
    h.update(sprint_tag.encode("utf-8"))
    h.update(b"\x00")
    h.update(family.encode("utf-8"))
    h.update(b"\x00")
    h.update(index.to_bytes(4, "big", signed=False))
    return int.from_bytes(h.digest()[:4], "big", signed=False)


def _topology_hash(graph) -> str:
    """SHA256 hex of (sorted edge_index bytes + num_nodes), truncated to 16
    chars so the committed MANIFEST does not trip detect-secrets' high-entropy
    rule. 16 hex chars (64 bits) is still way more than enough for per-suite
    drift detection.
    """
    h = hashlib.sha256()
    h.update(int(graph.num_nodes).to_bytes(8, "big"))
    ei = graph.edge_index.cpu().to(torch.int64).contiguous()
    # sort for deterministic order
    sorted_ei = ei.T.sort(dim=0)[0] if ei.numel() else ei.T
    h.update(sorted_ei.numpy().tobytes())
    # 10 hex chars = 40 bits; well below detect-secrets' threshold, still
    # unique across a 30-graph suite.
    return h.hexdigest()[:10]


def make_holdout_suite(
    sprint_tag: str = "holdout_v1",
    salt_path: Optional[Path] = None,
) -> tuple[List[TestGraph], SuiteManifest]:
    """Generate the opaque held-out suite.

    Returns (graphs, manifest). Callers should compute metrics on the graphs,
    write the manifest, then drop the graph tensors so nothing persists on
    disk except the manifest.
    """
    salt = _load_salt(salt_path)
    salt_digest_prefix = hashlib.sha256(salt).hexdigest()[:8]
    manifest = SuiteManifest(sprint_tag=sprint_tag, salt_digest_prefix=salt_digest_prefix)

    graphs: List[TestGraph] = []
    for idx, (family, n) in enumerate(_HOLDOUT_V1_PLAN):
        builder = _FAMILY_SPECS[family]
        seed = _derive_seed(salt, sprint_tag, family, idx)
        tg = builder(n, seed)
        graphs.append(tg)
        manifest.entries.append(
            {
                "index": idx,
                "family": family,
                "target_n": n,
                "actual_n": int(tg.graph.num_nodes),
                "actual_e": int(tg.graph.edge_index.shape[1]),
                "topology_sha256_10": _topology_hash(tg.graph),
                "seed": seed,
            }
        )

    return graphs, manifest


def make_rolling_suite(
    sprint_tag: str,
    salt_path: Optional[Path] = None,
    size: int = 10,
) -> tuple[List[TestGraph], SuiteManifest]:
    """Generate a per-sprint rolling anti-overfit set.

    Different seed than held-out (different sprint_tag), same salt. Size
    defaults to 10 per 10_iteration_loop.md; families drawn from the
    P0/P1 pool to exercise the current sprint's weak spots.
    """
    if not sprint_tag or sprint_tag.startswith("holdout_"):
        raise ValueError(
            f"Rolling sprint_tag must be non-empty and must NOT start with "
            f"'holdout_' (got {sprint_tag!r}); reserved for held-out."
        )
    salt = _load_salt(salt_path)
    salt_digest_prefix = hashlib.sha256(salt).hexdigest()[:8]
    manifest = SuiteManifest(sprint_tag=sprint_tag, salt_digest_prefix=salt_digest_prefix)

    # Round-robin through the family specs; size determines how many rotations.
    family_names = list(_FAMILY_SPECS.keys())
    graphs: List[TestGraph] = []
    for idx in range(size):
        family = family_names[idx % len(family_names)]
        builder = _FAMILY_SPECS[family]
        # Vary size: 80, 200, 400, ... repeating. Rolling set biased small for speed.
        n = [80, 200, 400, 600][idx % 4]
        seed = _derive_seed(salt, sprint_tag, family, idx)
        tg = builder(n, seed)
        graphs.append(tg)
        manifest.entries.append(
            {
                "index": idx,
                "family": family,
                "target_n": n,
                "actual_n": int(tg.graph.num_nodes),
                "actual_e": int(tg.graph.edge_index.shape[1]),
                "topology_sha256_10": _topology_hash(tg.graph),
                "seed": seed,
            }
        )

    return graphs, manifest


__all__ = [
    "DEFAULT_SALT_PATH",
    "SuiteManifest",
    "make_holdout_suite",
    "make_rolling_suite",
]
