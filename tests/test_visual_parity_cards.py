"""Tests for the visual parity v2 card manifest generator."""

from __future__ import annotations

from pathlib import Path

from scripts.visual_parity import cards
from scripts.visual_parity.io import read_card_manifest, write_card_manifest


def test_v2_calibration_rows_match_catalog() -> None:
    """Every v2 row should carry a case_id known to the calibration catalog."""

    import scripts.generate_calibration_suite as calibration

    catalog_ids = {case.case_id for case in calibration.build_case_catalog()}
    rows = cards._v2_calibration_rows()

    assert rows
    assert {row["case_id"] for row in rows} == catalog_ids
    assert all(row["source"] == cards.SOURCE_V2 for row in rows)
    assert all(row["size_policy"] in {"auto", "fixed", "density", "stress"} for row in rows)


def test_export_legacy_cards_covers_reference_combo_and_evil_tiers() -> None:
    """Legacy export should include Tier A/B/C ids across all three card kinds."""

    rows = cards.export_legacy_cards()

    assert rows
    kinds = {row["kind"] for row in rows}
    assert kinds == {"reference", "combo", "evil"}

    tiers = {row["tier"] for row in rows}
    assert tiers == {"A", "B", "C"}

    case_ids = {row["case_id"] for row in rows}
    assert "evil_huge_arrows" in case_ids
    assert all(row["source"] == cards.SOURCE_LEGACY for row in rows)


def test_build_card_manifest_round_trips_through_io(tmp_path: Path) -> None:
    """The generated manifest should validate and round-trip byte-stably."""

    manifest = cards.build_card_manifest()
    out_path = tmp_path / "card_manifest.json"
    write_card_manifest(out_path, manifest)

    reloaded = read_card_manifest(out_path)
    assert reloaded["schema_version"] == manifest["schema_version"]
    assert len(reloaded["cards"]) == len(manifest["cards"])

    out_path_2 = tmp_path / "card_manifest_2.json"
    write_card_manifest(out_path_2, reloaded)
    assert out_path.read_text(encoding="utf-8") == out_path_2.read_text(encoding="utf-8")


def test_build_card_manifest_filters_v2_and_legacy_independently() -> None:
    """The include_v2 / include_legacy flags should be independently honored."""

    v2_only = cards.build_card_manifest(include_v2=True, include_legacy=False)
    legacy_only = cards.build_card_manifest(include_v2=False, include_legacy=True)

    assert all(card["source"] == cards.SOURCE_V2 for card in v2_only["cards"])
    assert all(card["source"] == cards.SOURCE_LEGACY for card in legacy_only["cards"])
