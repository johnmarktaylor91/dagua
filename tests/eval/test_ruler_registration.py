"""Registration and V3-isolation tests for the opt-in V4 scorer."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch

from dagua.eval.ruler_registry import (
    DEFAULT_RULER_KEY,
    RULER_CONFIG_KEY,
    get_ruler_registration,
    get_ruler_scorer,
)


def _v3_source_digest() -> str:
    """Hash the frozen V3 source bytes.

    Returns
    -------
    str
        SHA-256 of ``ruler_v3.py``.
    """

    import dagua.eval.ruler_v3 as ruler_v3

    path = Path(ruler_v3.__file__ or "")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registration_defaults_to_v3_without_importing_v4() -> None:
    """The absent opt-in key preserves the V3 callable and source bytes."""

    before = _v3_source_digest()
    registration = get_ruler_registration({})
    scorer = get_ruler_scorer({})
    after = _v3_source_digest()
    fresh_process = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import sys; from dagua.eval.ruler_registry import get_ruler_scorer; "
                "scorer=get_ruler_scorer({}); "
                "print(scorer.__module__); "
                "print('dagua.eval.ruler_v4.score' in sys.modules)"
            ),
        ],
        text=True,
    ).splitlines()

    assert RULER_CONFIG_KEY == "ruler"
    assert DEFAULT_RULER_KEY == "ruler_v3"
    assert registration.key == "ruler_v3"
    assert scorer.__module__ == "dagua.eval.ruler_v3"
    assert before == after
    assert fresh_process[-2:] == ["dagua.eval.ruler_v3", "False"]


def test_v4_registration_requires_explicit_opt_in() -> None:
    """The new key resolves V4 while retaining probation metadata."""

    registration = get_ruler_registration({"ruler": "ruler_v4"})
    scorer = get_ruler_scorer({"ruler": "ruler_v4"})

    assert registration.experimental
    assert registration.required_arguments == ("scene", "weight_table", "profiles")
    assert scorer.__module__ == "dagua.eval.ruler_v4.score"
    assert scorer.__name__ == "score"


def test_v3_result_is_identical_with_v4_installed_and_unused() -> None:
    """Registry-default V3 behavior equals a direct V3 call exactly."""

    from dagua.eval.ruler_v3 import score_core_v3

    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, 0.0]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    parameters = {
        "stress_sources": 4,
        "stress_targets": 4,
        "crossing_samples": 20,
        "neighborhood_samples": 4,
        "seed": 17,
    }

    direct = score_core_v3(positions, edge_index, **parameters)
    registered = get_ruler_scorer({})(positions, edge_index, **parameters)

    assert registered == direct
