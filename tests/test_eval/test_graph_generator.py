"""Targeted tests for dagua.eval.graph_generator (WP07-F09 / WP07-F10)."""

from __future__ import annotations

import string
from pathlib import Path

import dagua
from dagua.eval.graph_generator import DEFAULT_SALT_PATH, _topology_hash
from dagua.graph import DaguaGraph


def test_default_salt_path_is_repo_anchored_not_cwd_relative() -> None:
    """The holdout salt path must resolve independently of the process CWD.

    The old ``Path(".project-context/...")`` value resolved against whatever
    directory the process happened to run from, so ``make_holdout_suite()``
    raised FileNotFoundError from any CWD but the repo root (WP07-F10).
    """
    assert DEFAULT_SALT_PATH.is_absolute()
    repo_root = Path(dagua.__file__).resolve().parents[1]
    assert DEFAULT_SALT_PATH == repo_root / ".project-context" / "private" / "holdout_salt"


def test_topology_hash_is_ten_hex_chars() -> None:
    """Pins the 10-hex-char (40-bit) truncation the docstring now documents.

    The docstring used to claim 16 chars while the code returned 10
    (WP07-F09); this keeps code and documentation from drifting apart again.
    """
    graph = DaguaGraph()
    for index in range(3):
        graph.add_node(index)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    digest = _topology_hash(graph)
    assert len(digest) == 10
    assert all(char in string.hexdigits for char in digest)
    # Deterministic for identical topology.
    assert _topology_hash(graph) == digest
