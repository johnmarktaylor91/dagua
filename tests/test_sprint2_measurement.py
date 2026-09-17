"""Focused tests for the sprint-2 W1-D measurement scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import sprint2_proxy_audit as proxy
from scripts import sprint2_tie_headroom as headroom


def test_facet_headroom_uses_effective_weight_share() -> None:
    """Movable headroom should be expressed in exact headline score points."""
    facets = {
        "C4": {"applicable": True, "score": 0.5, "effective_weight": 2.0},
        "C5": {"applicable": True, "score": 1.0, "effective_weight": 1.0},
        "C6": {"applicable": False, "score": None, "effective_weight": 1.0},
        "C9": {"applicable": True, "score": 1.0, "effective_weight": 1.0},
        "C10": {"applicable": True, "score": 1.0, "effective_weight": 1.0},
        "C1": {"applicable": True, "score": 0.0, "effective_weight": 5.0},
    }

    contributions, aggregate = headroom.facet_headroom(facets)

    assert contributions["C4"] == pytest.approx(10.0)
    assert aggregate == pytest.approx(10.0)


def test_parse_tied_rows_accepts_padded_numeric_columns(tmp_path: Path) -> None:
    """The canonical table's spaces after equals signs should parse."""
    tied_rows = tmp_path / "rows.txt"
    tied_rows.write_text(
        "north/g.10.4 native= 99.35 field_best= 99.29 (sgd2) delta=+0.061 directed=True\n"
    )

    rows = headroom.parse_tied_rows(tied_rows)

    assert rows == [
        {
            "graph": "north/g.10.4",
            "native": 99.35,
            "field_best": 99.29,
            "field_engine": "sgd2",
            "delta": 0.061,
            "directed": True,
        }
    ]


def test_proxy_summary_flags_winner_outside_cut() -> None:
    """A low-proxy honest winner should be reported when all arms were scored."""
    arms = [
        {"name": f"arm_{index}", "raw_score": float(10 - index), "full_score": float(index)}
        for index in range(10)
    ]
    event = {
        "event": "native_candidate_marketplace",
        "route": "undirected",
        "top_k": 8,
        "_graph": "rome/example",
        "arms": arms,
    }

    overall = next(row for row in proxy.summarize([event]) if row["class"] == "overall")

    assert overall["observed_drops"] == ["rome/example:arm_9"]
    assert overall["spearman"] == pytest.approx(-1.0)


def test_verify_cousins_detects_name_collision_without_opening_graphs(tmp_path: Path) -> None:
    """Cousin verification should compare manifest identity, not graph paths."""
    train_dir = tmp_path / "train"
    corpus_dir = tmp_path / "corpus"
    train_dir.mkdir()
    corpus_dir.mkdir()
    (train_dir / "MANIFEST.json").write_text(
        json.dumps({"files": [{"relpath": "rome/shared.graphml", "sha256": "abc"}]})
    )
    (corpus_dir / "SUBSET.json").write_text(
        json.dumps(
            {
                "selected": [
                    {"relpath": "rome/shared.graphml"},
                    *({"relpath": f"rome/dev_{index}.graphml"} for index in range(62)),
                ]
            }
        )
    )
    (corpus_dir / "SEALED_REMAINDER.json").write_text(
        json.dumps(
            {"sealed": [{"relpath": f"north/sealed_{index}.graphml"} for index in range(77)]}
        )
    )

    _files, collisions = headroom.verify_cousins(train_dir / "MANIFEST.json", corpus_dir)

    assert collisions == ["name:rome/shared"]
