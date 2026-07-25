"""Tripwire for V3 ruler wiring over the real R8-extended corpus."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.ruler_v3_groups import evaluate_conditional_groups  # noqa: E402
from scripts.native_sprint_score import (  # noqa: E402
    build_graph_map,
    graph_meta_for_v3,
    selected_names,
)

REQUIRED_APPLICABLE = ("G1", "G2", "G3", "G4", "G6")
EXPECTED_INAPPLICABLE = ("G5", "G7")
DEFAULT_CORPUS_POSITIONS = ROOT / "eval_output" / "r81_regate2" / "positions"


def _probe_positions(num_nodes: int) -> torch.Tensor:
    """Build deterministic non-degenerate probe positions for group evaluation.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[num_nodes, 2]``.
    """
    side = max(1, int(num_nodes**0.5))
    coords = [[float(index % side), float(index // side)] for index in range(num_nodes)]
    return torch.tensor(coords, dtype=torch.float32)


def _corpus_names() -> List[str]:
    """Return the real 121-row corpus names used by the freeze ceremony.

    Returns
    -------
    List[str]
        Ordered corpus names.
    """
    _old_names, extended_names = selected_names(DEFAULT_CORPUS_POSITIONS, include_r8=True)
    return extended_names


def group_applicability_counts(names: Sequence[str]) -> Dict[str, List[str]]:
    """Evaluate V3 conditional-group applicability over the requested corpus.

    Parameters
    ----------
    names : Sequence[str]
        Corpus row names.

    Returns
    -------
    Dict[str, List[str]]
        Applicable example names keyed by group id.
    """
    graphs = build_graph_map(names)
    applicable: Dict[str, List[str]] = {
        group_key: [] for group_key in (*REQUIRED_APPLICABLE, *EXPECTED_INAPPLICABLE)
    }
    for name in names:
        test_graph = graphs[name]
        graph = test_graph.graph
        if graph.node_sizes is None:
            raise RuntimeError(f"{name}: node sizes missing from build_graph_map")
        groups = evaluate_conditional_groups(
            _probe_positions(graph.num_nodes),
            graph.edge_index,
            graph.node_sizes,
            graph_meta_for_v3(test_graph),
        )
        for group_key in applicable:
            if groups[group_key].applicable:
                applicable[group_key].append(name)
    return applicable


def _format_count(group_key: str, names: Sequence[str]) -> str:
    """Format one tripwire count line.

    Parameters
    ----------
    group_key : str
        Conditional group key.
    names : Sequence[str]
        Applicable row names.

    Returns
    -------
    str
        Human-readable count and examples.
    """
    examples = ", ".join(names[:2]) if names else "-"
    return f"{group_key}: applicable={len(names)} examples={examples}"


def main() -> int:
    """Run the V3 wiring tripwire.

    Returns
    -------
    int
        Process exit status.
    """
    names = _corpus_names()
    if len(names) != 121:
        raise RuntimeError(f"expected 121 corpus rows, found {len(names)}")
    applicable = group_applicability_counts(names)
    for group_key in (*REQUIRED_APPLICABLE, *EXPECTED_INAPPLICABLE):
        print(_format_count(group_key, applicable[group_key]))
    failures: List[str] = []
    for group_key in REQUIRED_APPLICABLE:
        if not applicable[group_key]:
            failures.append(f"{group_key} applicable on zero rows")
    for group_key in EXPECTED_INAPPLICABLE:
        if applicable[group_key]:
            failures.append(
                f"{group_key} spuriously applicable on {len(applicable[group_key])} rows"
            )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
