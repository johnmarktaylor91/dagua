# ruff: noqa: E402
"""Card manifest generation and legacy album export (Lane D).

This module owns the *generated content* of ``card_manifest.json`` (Lane E0
creates a schema-valid stub only; Lane C reads the file; Lane D writes it --
see IMPLEMENTATION_PLAN.md's file-ownership map).

Two sources feed the manifest:

1. **v2 native rows** -- one row per
   ``scripts.generate_calibration_suite.CalibrationCase``. These case_ids are
   exactly what ``generate_calibration_suite.py --manifest`` resolves against
   when rendering two-panel comparisons.
2. **Legacy album export** -- a *one-time* extraction of Tier A/B/C ids and
   evil-combo ids from the FROZEN album zoo
   (``scripts/build_gallery_audit.py``, which itself imports catalog data
   from ``generate_cosmetic_album.py`` and ``generate_combo_album.py``).
   Per FINAL_DESIGN.md section 2 ("FREEZE") and correction F18: the zoo
   (~17.7k lines across build_gallery_audit.py / generate_cosmetic_album.py /
   generate_combo_album.py / per_card_pixel_diff.py) is left runnable for
   historical comparison but is not extended. Only its *card definitions*
   (stable ids, tiers, and category/kind labels) are captured here so the
   catalog is not silently lost when the zoo stops being touched. Legacy rows
   are informational (``source="legacy_album_zoo"``): their case_ids do not
   necessarily match ``generate_calibration_suite``'s catalog and they are
   not directly renderable through the v2 two-panel path.

Card row schema (this module's own -- not part of
``scripts.visual_parity.types``, since Lane D owns card_manifest.json's
generated content per the file-ownership map)::

    {
      "case_id": str,
      "category": str,
      "description": str,
      "source": "v2_calibration_suite" | "legacy_album_zoo",
      "kind": "reference" | "combo" | "evil" | None,
      "tier": "A" | "B" | "C" | None,
      "reference_tool": str,
      "reference_attr": str,
      "reference_value": str,
      "coverage_cell_id": str | None,
      "target_kind": str,
      "size_policy": str,
    }
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visual_parity.io import CARD_MANIFEST_SCHEMA_VERSION, write_card_manifest

SOURCE_V2 = "v2_calibration_suite"
SOURCE_LEGACY = "legacy_album_zoo"


def _v2_calibration_rows() -> List[Dict[str, Any]]:
    """Build one manifest row per v2 ``CalibrationCase``.

    Returns
    -------
    list[dict[str, Any]]
        Renderable rows whose ``case_id`` matches
        ``scripts.generate_calibration_suite.build_case_catalog()``.
    """

    import scripts.generate_calibration_suite as calibration

    rows: List[Dict[str, Any]] = []
    for case in calibration.build_case_catalog():
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "description": case.description,
                "source": SOURCE_V2,
                "kind": None,
                "tier": None,
                "reference_tool": case.reference_tool,
                "reference_attr": case.reference_attr,
                "reference_value": case.reference_value,
                "coverage_cell_id": case.coverage_cell_id,
                "target_kind": case.target_kind,
                "size_policy": case.size_policy,
            }
        )
    return rows


def _legacy_row(
    *,
    case_id: str,
    category: str,
    description: str,
    kind: str,
    tier: str,
) -> Dict[str, Any]:
    """Build one legacy-export manifest row.

    Parameters
    ----------
    case_id
        Stable legacy identifier (card_id / case_id from the album zoo).
    category
        Human-readable category/feature grouping from the zoo.
    description
        Human-readable description.
    kind
        One of ``"reference"``, ``"combo"``, ``"evil"``.
    tier
        Audit tier: ``"A"``, ``"B"``, or ``"C"``.

    Returns
    -------
    dict[str, Any]
        A card_manifest.json row.
    """

    return {
        "case_id": case_id,
        "category": category,
        "description": description,
        "source": SOURCE_LEGACY,
        "kind": kind,
        "tier": tier,
        "reference_tool": "graphviz",
        "reference_attr": "",
        "reference_value": "",
        "coverage_cell_id": None,
        "target_kind": "heuristic",
        "size_policy": "stress" if kind == "evil" else "density",
    }


def export_legacy_cards() -> List[Dict[str, Any]]:
    """Export Tier A/B/C ids and evil combos from the frozen album zoo.

    This is the one-time extraction described by FINAL_DESIGN.md section 2
    ("Cut ... Card DEFINITIONS worth keeping (Tier A/B/C ids, evil combos)
    are exported once into card_manifest.json by export_legacy_cards; then
    the zoo is legacy."). It imports ``scripts.build_gallery_audit`` for its
    catalog-building functions only (no rendering is triggered) and never
    modifies the zoo.

    Returns
    -------
    list[dict[str, Any]]
        Legacy-sourced manifest rows covering atomic reference cards, combo
        cards, and evil stress-test cards, each carrying its resolved
        Tier A/B/C classification.
    """

    import scripts.build_gallery_audit as gallery

    rows: List[Dict[str, Any]] = []

    for item in gallery.build_reference_items():
        rows.append(
            _legacy_row(
                case_id=item.card_id,
                category=item.spec.category,
                description=f"{item.spec.feature}={item.value.slug}",
                kind="reference",
                tier=item.spec.tier,
            )
        )

    for spec in gallery.build_combo_specs():
        rows.append(
            _legacy_row(
                case_id=spec.case_id,
                category=f"combo_{spec.combo_kind}",
                description=spec.title,
                kind="combo",
                tier=spec.tier,
            )
        )

    for spec in gallery.build_evil_specs():
        rows.append(
            _legacy_row(
                case_id=spec.case_id,
                category="evil_combos",
                description=spec.title,
                kind="evil",
                tier=spec.tier,
            )
        )

    return rows


def build_card_manifest(
    *,
    include_v2: bool = True,
    include_legacy: bool = True,
) -> Dict[str, Any]:
    """Build the full card_manifest.json payload.

    Parameters
    ----------
    include_v2
        Whether to include native v2 ``CalibrationCase`` rows.
    include_legacy
        Whether to include the one-time legacy album-zoo export.

    Returns
    -------
    dict[str, Any]
        A schema-version-checked card_manifest.json payload ready for
        ``scripts.visual_parity.io.write_card_manifest``.
    """

    cards: List[Dict[str, Any]] = []
    if include_v2:
        cards.extend(_v2_calibration_rows())
    if include_legacy:
        cards.extend(export_legacy_cards())

    return {
        "schema_version": CARD_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cards": cards,
    }


def _summarize(manifest: Dict[str, Any]) -> str:
    """Return a short human-readable summary of a card manifest.

    Parameters
    ----------
    manifest
        Card manifest payload.

    Returns
    -------
    str
        Multi-line summary text.
    """

    cards = manifest.get("cards", [])
    by_source = Counter(card.get("source") for card in cards)
    by_tier = Counter(card.get("tier") for card in cards if card.get("tier"))
    lines = [f"cards: {len(cards)}"]
    for source, count in sorted(by_source.items()):
        lines.append(f"  {source}: {count}")
    if by_tier:
        lines.append(f"  tiers: {dict(sorted(by_tier.items()))}")
    return "\n".join(lines)


def main() -> int:
    """Parse CLI arguments and export the card manifest.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=".project-context/research/sprint_visual_parity_v2/card_manifest.json",
        help="Destination card_manifest.json path.",
    )
    parser.add_argument(
        "--v2-only",
        action="store_true",
        help="Emit only native v2 CalibrationCase rows (skip legacy export).",
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Emit only the legacy album-zoo export (skip v2 rows).",
    )
    args = parser.parse_args()

    manifest = build_card_manifest(
        include_v2=not args.legacy_only,
        include_legacy=not args.v2_only,
    )
    write_card_manifest(args.out, manifest)
    print(f"Wrote {args.out}")
    print(_summarize(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
