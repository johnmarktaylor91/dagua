#!/usr/bin/env python
"""Build Agg-vs-cairo-vs-Graphviz comparison triptychs for Tier A cards."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.build_gallery_audit as gallery  # noqa: E402
import scripts.per_card_pixel_diff as pixel_diff  # noqa: E402
from scripts.competitor_renderers import render_competitor  # noqa: E402
from scripts.competitor_renderers.utils import graph_spec_from_dagua  # noqa: E402

DEFAULT_OUT_DIR = Path("eval_output/backend_comparison")
DEFAULT_AGG_SUMMARY = Path("eval_output/gallery_audit/per_card_pixel_diff/summary.json")
DEFAULT_CAIRO_SUMMARY = Path("eval_output/gallery_audit_cairo/per_card_pixel_diff/summary.json")
PANEL_LABELS: Tuple[str, str, str] = ("dagua-agg", "dagua-cairo", "graphviz")
ROUND9_WIN_IDS: Tuple[str, ...] = (
    "combo_pie_bold",
    "combo_donut_shadow",
    "evil_donut_diamond",
    "clusters_opacity_1_0",
)


def _tier_a_contexts(
    contexts: Mapping[str, pixel_diff.CardRenderContext],
) -> Dict[str, pixel_diff.CardRenderContext]:
    """Filter contexts down to Graphviz-backed Tier A cards.

    Parameters
    ----------
    contexts : Mapping[str, pixel_diff.CardRenderContext]
        All per-card render contexts.

    Returns
    -------
    dict[str, pixel_diff.CardRenderContext]
        Tier A contexts keyed by card ID.
    """

    return {
        card_id: context
        for card_id, context in contexts.items()
        if context.tier == "A" and "graphviz" in context.competitor_tools
    }


def _category_from_context(context: pixel_diff.CardRenderContext) -> Path:
    """Return the backend-comparison category path for a card.

    Parameters
    ----------
    context : pixel_diff.CardRenderContext
        Card render context.

    Returns
    -------
    pathlib.Path
        Relative category path below the comparison output root.
    """

    parts = Path(context.relative_path).parts
    if len(parts) >= 4 and parts[0] == "cards":
        return Path(*parts[2:-1])
    if len(parts) >= 3 and parts[0] == "cards":
        return Path(parts[1])
    return Path(context.kind)


def _render_dagua_panel(
    context: pixel_diff.CardRenderContext,
    backend: str,
    temp_dir: Path,
) -> Image.Image:
    """Render one Dagua backend panel.

    Parameters
    ----------
    context : pixel_diff.CardRenderContext
        Card render context.
    backend : str
        Dagua Matplotlib backend name.
    temp_dir : pathlib.Path
        Temporary directory for raw PNG output.

    Returns
    -------
    PIL.Image.Image
        Normalized panel image.
    """

    raw_path = temp_dir / f"{backend}.png"
    gallery._render_dagua_png(
        context.graph,
        context.positions,
        raw_path,
        gallery.PANEL_SIZE,
        backend=backend,
    )
    return gallery._place_render_on_canvas(
        raw_path,
        gallery.PANEL_SIZE,
        gallery.PANEL_CONTENT_INSET,
        canvas_color=gallery._graph_background_color(context.graph),
    )


def _render_graphviz_panel(
    context: pixel_diff.CardRenderContext,
    temp_dir: Path,
) -> Optional[Image.Image]:
    """Render the Graphviz reference panel.

    Parameters
    ----------
    context : pixel_diff.CardRenderContext
        Card render context.
    temp_dir : pathlib.Path
        Temporary directory for raw PNG output.

    Returns
    -------
    PIL.Image.Image | None
        Normalized Graphviz panel, or ``None`` if Graphviz could not render.
    """

    raw_path = temp_dir / "graphviz.png"
    graph_spec = graph_spec_from_dagua(context.graph)
    positions = [(float(x), float(y)) for x, y in context.positions.detach().cpu().tolist()]
    rendered: Optional[Path]
    try:
        rendered = render_competitor(
            "graphviz",
            graph_spec,
            positions,
            raw_path,
            gallery.PANEL_SIZE,
        )
    except Exception:
        rendered = None
    if rendered is None or not rendered.exists():
        cached = (
            pixel_diff.DEFAULT_GALLERY_ROOT
            / "per_card_pixel_diff"
            / "competitors"
            / "graphviz"
            / f"{context.card_id}.png"
        )
        if not cached.exists():
            return None
        rendered = cached
    return gallery._place_render_on_canvas(
        rendered,
        gallery.PANEL_SIZE,
        gallery.PANEL_CONTENT_INSET,
        downscale_overflow=True,
    )


def _compose_triptych(
    panels: Sequence[Image.Image],
    context: pixel_diff.CardRenderContext,
    output_path: Path,
) -> None:
    """Compose and write a three-panel comparison image.

    Parameters
    ----------
    panels : Sequence[PIL.Image.Image]
        Agg, cairo, and Graphviz panels.
    context : pixel_diff.CardRenderContext
        Card render context.
    output_path : pathlib.Path
        Destination PNG path.

    Returns
    -------
    None
        The triptych PNG is written to ``output_path``.
    """

    width = gallery.PANEL_SIZE[0] * 3
    height = gallery.PANEL_SIZE[1]
    canvas = Image.new("RGB", (width, height), gallery.WHITE)
    for index, panel in enumerate(panels):
        x_offset = index * gallery.PANEL_SIZE[0]
        canvas.paste(panel.convert("RGB"), (x_offset, 0))
    draw = ImageDraw.Draw(canvas)
    label_font = gallery._load_font(22, bold=True)
    for index, label in enumerate(PANEL_LABELS):
        x_offset = index * gallery.PANEL_SIZE[0]
        draw.text((x_offset + 42, 96), label, fill=gallery.TEXT_COLOR, font=label_font)
        if index:
            draw.line((x_offset, 0, x_offset, height), fill="#D7DDE3", width=2)
    gallery._draw_header(
        canvas,
        title=context.card_id,
        subtitle="dagua-agg | dagua-cairo | Graphviz",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_backend_comparison_gallery(
    output_dir: Path = DEFAULT_OUT_DIR,
    card_ids: Optional[Sequence[str]] = None,
) -> Tuple[int, List[str]]:
    """Build backend triptychs for Tier A cards.

    Parameters
    ----------
    output_dir : pathlib.Path, default=DEFAULT_OUT_DIR
        Output root for triptych PNGs.
    card_ids : Sequence[str] | None, optional
        Optional card ID subset.

    Returns
    -------
    tuple[int, list[str]]
        Number of written triptychs and warning messages for skipped cards.
    """

    contexts = _tier_a_contexts(pixel_diff._build_contexts(card_ids))
    warnings: List[str] = []
    written = 0
    for card_id, context in contexts.items():
        output_path = output_dir / _category_from_context(context) / f"{card_id}.png"
        with tempfile.TemporaryDirectory(prefix="dagua-backend-compare-") as tmp:
            temp_dir = Path(tmp)
            graphviz_panel = _render_graphviz_panel(context, temp_dir)
            if graphviz_panel is None:
                warnings.append(f"{card_id}: graphviz render unavailable")
                continue
            panels = [
                _render_dagua_panel(context, "agg", temp_dir),
                _render_dagua_panel(context, "cairo", temp_dir),
                graphviz_panel,
            ]
            _compose_triptych(panels, context, output_path)
            written += 1
    return written, warnings


def _load_l1_by_card(summary_path: Path) -> Dict[str, float]:
    """Load non-skipped card L1 values from a pixel-diff summary.

    Parameters
    ----------
    summary_path : pathlib.Path
        Path to ``summary.json``.

    Returns
    -------
    dict[str, float]
        L1 values keyed by card slug.
    """

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        str(card["slug"]): float(card["l1"])
        for card in data.get("cards", [])
        if not card.get("skipped") and card.get("tier") == "A" and "l1" in card
    }


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean for a non-empty sequence.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values.

    Returns
    -------
    float
        Arithmetic mean.
    """

    if not values:
        return 0.0
    return sum(values) / len(values)


def write_summary_report(
    output_dir: Path = DEFAULT_OUT_DIR,
    agg_summary: Path = DEFAULT_AGG_SUMMARY,
    cairo_summary: Path = DEFAULT_CAIRO_SUMMARY,
    warnings: Optional[Sequence[str]] = None,
) -> Path:
    """Write the backend comparison metric report.

    Parameters
    ----------
    output_dir : pathlib.Path, default=DEFAULT_OUT_DIR
        Output root for ``SUMMARY.md``.
    agg_summary : pathlib.Path, default=DEFAULT_AGG_SUMMARY
        Agg pixel-diff summary JSON.
    cairo_summary : pathlib.Path, default=DEFAULT_CAIRO_SUMMARY
        Cairo pixel-diff summary JSON.
    warnings : Sequence[str] | None, optional
        Triptych generation warnings to include.

    Returns
    -------
    pathlib.Path
        Written Markdown report path.
    """

    agg_l1 = _load_l1_by_card(agg_summary)
    cairo_l1 = _load_l1_by_card(cairo_summary)
    common_ids = sorted(set(agg_l1) & set(cairo_l1))
    drops = sorted(
        ((card_id, agg_l1[card_id] - cairo_l1[card_id]) for card_id in common_ids),
        key=lambda item: item[1],
        reverse=True,
    )
    regressions = [item for item in reversed(drops) if item[1] < 0.0]

    lines = [
        "# Backend Comparison Summary",
        "",
        f"- Agg Tier A cards: {len(agg_l1)}",
        f"- Cairo Tier A cards: {len(cairo_l1)}",
        f"- Common Tier A cards: {len(common_ids)}",
        f"- Mean Tier A L1 under Agg: {_mean(list(agg_l1.values())):.6f}",
        f"- Mean Tier A L1 under cairo: {_mean(list(cairo_l1.values())):.6f}",
        "",
        "## Top 20 Cairo L1 Drops",
        "",
        "| Card | Agg L1 | Cairo L1 | Drop |",
        "| --- | ---: | ---: | ---: |",
    ]
    for card_id, drop in drops[:20]:
        lines.append(
            f"| {card_id} | {agg_l1[card_id]:.6f} | {cairo_l1[card_id]:.6f} | {drop:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Top 5 Cairo Regressions",
            "",
            "| Card | Agg L1 | Cairo L1 | Increase |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for card_id, drop in regressions[:5]:
        lines.append(
            f"| {card_id} | {agg_l1[card_id]:.6f} | {cairo_l1[card_id]:.6f} | {-drop:.6f} |"
        )
    if not regressions:
        lines.append("| None | - | - | - |")

    lines.extend(
        [
            "",
            "## Round-9 Wins",
            "",
            "| Card | Agg L1 | Cairo L1 | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for card_id in ROUND9_WIN_IDS:
        agg_value = agg_l1.get(card_id)
        cairo_value = cairo_l1.get(card_id)
        if agg_value is None or cairo_value is None:
            lines.append(f"| {card_id} | missing | missing | missing |")
            continue
        lines.append(
            f"| {card_id} | {agg_value:.6f} | {cairo_value:.6f} | {agg_value - cairo_value:.6f} |"
        )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_csv(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Parse an optional comma-separated card list.

    Parameters
    ----------
    value : str | None
        Raw comma-separated value.

    Returns
    -------
    tuple[str, ...] | None
        Parsed card IDs, or ``None`` when omitted.
    """

    if value is None or not value.strip():
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cards", type=str, default=None, help="Comma-separated card IDs")
    parser.add_argument("--agg-summary", type=Path, default=DEFAULT_AGG_SUMMARY)
    parser.add_argument("--cairo-summary", type=Path, default=DEFAULT_CAIRO_SUMMARY)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write SUMMARY.md from existing metric summaries",
    )
    return parser.parse_args()


def main() -> None:
    """Build triptychs and write the quantitative comparison report.

    Returns
    -------
    None
        Artifacts are written under the requested output directory.
    """

    args = parse_args()
    warnings: List[str] = []
    written = 0
    if not args.summary_only:
        written, warnings = build_backend_comparison_gallery(
            args.output_dir,
            _parse_csv(args.cards),
        )
    summary_path = write_summary_report(
        args.output_dir,
        args.agg_summary,
        args.cairo_summary,
        warnings=warnings,
    )
    print(f"backend_comparison triptychs={written} summary={summary_path}")


if __name__ == "__main__":
    main()
