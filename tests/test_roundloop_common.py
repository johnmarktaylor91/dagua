"""ScoreCache keying tests: byte-identical tensors must not cross engines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import roundloop_common as rl  # noqa: E402

from dagua.eval.graphs import TestGraph
from dagua.graph import DaguaGraph

_GRAPH_NAME = "keying_probe"


def _graph_map() -> Dict[str, TestGraph]:
    """Build a minimal graph map with measured node boxes.

    Returns
    -------
    Dict[str, TestGraph]
        Single 3-node path graph keyed by its probe name.
    """
    graph = DaguaGraph()
    for name in ("a", "b", "c"):
        graph.add_node(name)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.compute_node_sizes()
    return {_GRAPH_NAME: TestGraph(name=_GRAPH_NAME, graph=graph, tags={"directed"})}


def test_score_cache_key_discriminates_engines() -> None:
    """Same graph/sha/signature under two engines must map to distinct keys."""
    key_x1 = rl.ScoreCache.key(_GRAPH_NAME, "webcola", "f" * 64, "sig")
    key_x72 = rl.ScoreCache.key(_GRAPH_NAME, "sparse_stress", "f" * 64, "sig")
    assert key_x1 != key_x72


def test_cross_engine_tensor_does_not_share_cached_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One tensor under an x1 then an x72 engine gets two correct fresh scores.

    The finder's exact reproduction shape (drywell R3-B3-F3): score_position
    is engine-parameterized by the x72/x1 store-unit multiplier, so a
    webcola-cached row served for sparse_stress scored 6.69 with a spurious
    DEGENERATE_SCALE instead of 36.56. The cache key must include the engine;
    same-engine reuse must still hit the cache.
    """
    graphs = _graph_map()
    pos_path = tmp_path / "pos.pt"
    torch.save(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32), pos_path)
    cache_path = tmp_path / "scores_cache.json"
    signature = "probe-sig"

    scored_engines: list[str] = []
    real_score_position = rl.score_position

    def counting_score_position(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Record the engine of every fresh scoring call."""
        scored_engines.append(str(args[2]))
        return real_score_position(*args, **kwargs)

    monkeypatch.setattr(rl, "score_position", counting_score_position)

    task_x1 = (_GRAPH_NAME, "webcola", str(pos_path))
    cache_first = rl.ScoreCache(cache_path)
    row_x1 = rl.score_positions_cached([task_x1], graphs, cache_first, signature, workers=1)[
        task_x1
    ]

    # Reload from disk, as a second tool run sharing the default cache would.
    task_x72 = (_GRAPH_NAME, "sparse_stress", str(pos_path))
    cache_second = rl.ScoreCache(cache_path)
    row_x72 = rl.score_positions_cached([task_x72], graphs, cache_second, signature, workers=1)[
        task_x72
    ]

    # The x72 engine was scored fresh, NOT served the x1 engine's row.
    assert scored_engines == ["webcola", "sparse_stress"]
    assert row_x1["engine"] == "webcola"
    assert row_x72["engine"] == "sparse_stress"
    assert row_x1["scoring_units"]["position_scale"] == 1.0
    assert row_x72["scoring_units"]["position_scale"] == 72.0
    assert "DEGENERATE_SCALE" in row_x1["scoring_units"]["flags"]
    assert "DEGENERATE_SCALE" not in row_x72["scoring_units"]["flags"]
    assert row_x1["extended_composite"] != row_x72["extended_composite"]

    # Same-engine reuse still hits the cache: no third fresh scoring call.
    row_again = rl.score_positions_cached([task_x72], graphs, cache_second, signature, workers=1)[
        task_x72
    ]
    assert scored_engines == ["webcola", "sparse_stress"]
    assert row_again["extended_composite"] == row_x72["extended_composite"]

    # Both rows persist under engine-bearing keys.
    persisted = json.loads(cache_path.read_text())
    keys = sorted(persisted["rows"])
    assert len(keys) == 2
    assert any(f"{_GRAPH_NAME}::webcola::" in key for key in keys)
    assert any(f"{_GRAPH_NAME}::sparse_stress::" in key for key in keys)
