#!/usr/bin/env python
"""Per-card pixel diffs for gallery-audit cosmetic competitor references."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile
from skimage.metrics import structural_similarity

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.build_gallery_audit as gallery  # noqa: E402
from scripts.competitor_renderers import render_competitor  # noqa: E402
from scripts.competitor_renderers.utils import (  # noqa: E402
    MAX_COMPARISON_SIDE_PX,
    ensure_png_dimensions,
    graph_spec_from_dagua,
    write_json,
)

DEFAULT_GALLERY_ROOT = Path("eval_output/gallery_audit")
DEFAULT_OUT_DIR = DEFAULT_GALLERY_ROOT / "per_card_pixel_diff"
DEFAULT_CARD_DIMS = (gallery.PANEL_HALF_WIDTH, gallery.PANEL_HEIGHT)
HIRES_MAX_SIDE_PX = 1500


@dataclass(frozen=True)
class CardRenderContext:
    """Resolved graph data for a card.

    Parameters
    ----------
    card_id : str
        Stable card ID.
    kind : str
        Card kind: ``reference``, ``combo``, or ``evil``.
    graph : Any
        Dagua graph to render.
    positions : Any
        Node positions with shape ``[N, 2]``.
    relative_path : str
        Gallery-relative Dagua card path.
    competitor_tools : tuple[str, ...]
        Ordered competitor preference.
    tier : str
        Tier A/B/C marker.
    tier_c_reason : str
        Skip reason for Tier C cards.
    """

    card_id: str
    kind: str
    graph: Any
    positions: Any
    relative_path: str
    competitor_tools: Tuple[str, ...]
    tier: str
    tier_c_reason: str = ""


def _parse_csv(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Parse an optional comma-separated CLI argument.

    Parameters
    ----------
    value : str | None
        Raw CLI value.

    Returns
    -------
    tuple[str, ...] | None
        Parsed values, or ``None`` when omitted.
    """

    if value is None or not value.strip():
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _reference_context(item: gallery.ReferenceCardItem) -> CardRenderContext:
    """Resolve render context for one atomic reference card.

    Parameters
    ----------
    item : gallery.ReferenceCardItem
        Reference card item.

    Returns
    -------
    CardRenderContext
        Resolved graph, positions, and tier metadata.
    """

    graph, positions = gallery._prepare_reference_render(item)
    tools = gallery._card_competitor_tools(item)
    return CardRenderContext(
        card_id=item.card_id,
        kind="reference",
        graph=graph,
        positions=positions,
        relative_path=item.relative_path,
        competitor_tools=tools,
        tier=gallery._classify_tier(tools),
        tier_c_reason=gallery._tier_c_reason(item) if not tools else "",
    )


def _combo_context(item: gallery.ComboCardItem) -> CardRenderContext:
    """Resolve render context for one combo card.

    Parameters
    ----------
    item : gallery.ComboCardItem
        Combo card item.

    Returns
    -------
    CardRenderContext
        Resolved graph, positions, and tier metadata.
    """

    fixture = gallery._choose_combo_fixture(item.spec.settings)
    direction = str(item.spec.settings.get("direction", "TB"))
    graph, positions = gallery._build_fixture(fixture, direction=direction)
    positions = gallery._apply_reference_params(
        graph,
        positions,
        gallery._combo_params(item.spec.settings, fixture),
        fixture,
    )
    tools = gallery._card_competitor_tools(item)
    return CardRenderContext(
        card_id=item.card_id,
        kind="combo",
        graph=graph,
        positions=positions,
        relative_path=item.relative_path,
        competitor_tools=tools,
        tier=gallery._classify_tier(tools),
        tier_c_reason=gallery._tier_c_reason(item) if not tools else "",
    )


def _evil_context(item: gallery.EvilCardItem) -> CardRenderContext:
    """Resolve render context for one evil card.

    Parameters
    ----------
    item : gallery.EvilCardItem
        Evil card item.

    Returns
    -------
    CardRenderContext
        Resolved graph, positions, and tier metadata.
    """

    tools = gallery._card_competitor_tools(item)
    return CardRenderContext(
        card_id=item.card_id,
        kind="evil",
        graph=item.spec.graph,
        positions=item.spec.positions,
        relative_path=item.relative_path,
        competitor_tools=tools,
        tier=gallery._classify_tier(tools),
        tier_c_reason=gallery._tier_c_reason(item) if not tools else "",
    )


def _build_contexts(card_filter: Optional[Sequence[str]]) -> Dict[str, CardRenderContext]:
    """Build all card render contexts, optionally filtered by ID.

    Parameters
    ----------
    card_filter : Sequence[str] | None
        Optional card IDs to include.

    Returns
    -------
    dict[str, CardRenderContext]
        Contexts keyed by card ID.
    """

    requested = set(card_filter or [])
    contexts: Dict[str, CardRenderContext] = {}
    for item in gallery.build_reference_items():
        if not requested or item.card_id in requested:
            contexts[item.card_id] = _reference_context(item)
    for item in gallery.build_combo_items():
        if not requested or item.card_id in requested:
            contexts[item.card_id] = _combo_context(item)
    for item in gallery.build_evil_items():
        if not requested or item.card_id in requested:
            contexts[item.card_id] = _evil_context(item)
    missing = sorted(requested - set(contexts))
    if missing:
        raise ValueError(f"Unknown card IDs: {missing}")
    return contexts


def _save_dagua_render(
    context: CardRenderContext,
    output_path: Path,
    dimensions: Tuple[int, int],
    backend: Optional[str] = None,
) -> Path:
    """Render the Dagua side of a comparison.

    Parameters
    ----------
    context : CardRenderContext
        Resolved card context.
    output_path : pathlib.Path
        PNG destination.
    dimensions : tuple[int, int]
        Requested output dimensions.
    backend : str | None, optional
        Matplotlib backend selector passed through to Dagua rendering.

    Returns
    -------
    pathlib.Path
        Written Dagua PNG path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dagua-card-diff-") as tmp:
        raw_path = Path(tmp) / "dagua.png"
        gallery._render_dagua_png(
            context.graph,
            context.positions,
            raw_path,
            dimensions,
            backend=backend,
        )
        placed = gallery._place_render_on_canvas(raw_path, dimensions, (42, 42, 42, 42))
        placed.save(output_path)
    return ensure_png_dimensions(output_path, dimensions)


def _fit_hires_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Return dimensions capped to the hi-res longest-side limit.

    Parameters
    ----------
    width : int
        Source width.
    height : int
        Source height.

    Returns
    -------
    tuple[int, int]
        Capped dimensions.
    """

    longest = max(width, height)
    if longest <= HIRES_MAX_SIDE_PX:
        return width, height
    scale = HIRES_MAX_SIDE_PX / float(longest)
    return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)


def _image_arrays(dagua_path: Path, competitor_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load two RGB images as arrays.

    Parameters
    ----------
    dagua_path : pathlib.Path
        Dagua PNG.
    competitor_path : pathlib.Path
        Competitor PNG.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        RGB arrays with shape ``[H, W, 3]``.
    """

    with Image.open(dagua_path) as dagua_img, Image.open(competitor_path) as comp_img:
        dagua_rgb = dagua_img.convert("RGB")
        comp_rgb = comp_img.convert("RGB")
        if dagua_rgb.size != comp_rgb.size:
            raise ValueError(f"Image dimensions differ: {dagua_rgb.size} vs {comp_rgb.size}")
        return np.asarray(dagua_rgb), np.asarray(comp_rgb)


def _write_heatmap(diff: np.ndarray, output_path: Path) -> Path:
    """Write a red transparent heatmap from per-pixel RGB error.

    Parameters
    ----------
    diff : numpy.ndarray
        Absolute RGB diff with shape ``[H, W, 3]``.
    output_path : pathlib.Path
        Heatmap PNG path.

    Returns
    -------
    pathlib.Path
        Written heatmap path.
    """

    error = diff.mean(axis=2)
    alpha = np.clip(error * 2.0, 0, 255).astype(np.uint8)
    rgba = np.zeros((*error.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path)
    ensure_png_dimensions(output_path, (rgba.shape[1], rgba.shape[0]))
    return output_path


def _metrics(dagua_path: Path, competitor_path: Path, heatmap_path: Path) -> Dict[str, object]:
    """Compute per-card pixel metrics.

    Parameters
    ----------
    dagua_path : pathlib.Path
        Dagua PNG.
    competitor_path : pathlib.Path
        Competitor PNG.
    heatmap_path : pathlib.Path
        Heatmap destination.

    Returns
    -------
    dict[str, object]
        Metric payload.
    """

    dagua_rgb, comp_rgb = _image_arrays(dagua_path, competitor_path)
    diff = np.abs(dagua_rgb.astype(np.int16) - comp_rgb.astype(np.int16)).astype(np.uint8)
    _write_heatmap(diff, heatmap_path)
    return {
        "l1": float(diff.mean()),
        "ssim": float(structural_similarity(dagua_rgb, comp_rgb, channel_axis=2)),
        "dims": [int(dagua_rgb.shape[1]), int(dagua_rgb.shape[0])],
    }


def _write_side_by_side(dagua_path: Path, competitor_path: Path, output_path: Path) -> Path:
    """Write a capped side-by-side comparison panel.

    Parameters
    ----------
    dagua_path : pathlib.Path
        Dagua PNG.
    competitor_path : pathlib.Path
        Competitor PNG.
    output_path : pathlib.Path
        Side-by-side destination.

    Returns
    -------
    pathlib.Path
        Written comparison panel path.
    """

    with Image.open(dagua_path) as dagua_img, Image.open(competitor_path) as comp_img:
        dagua_rgb = dagua_img.convert("RGB")
        comp_rgb = comp_img.convert("RGB")
        panel_width = dagua_rgb.width + comp_rgb.width
        panel_height = max(dagua_rgb.height, comp_rgb.height)
        if max(panel_width, panel_height) > MAX_COMPARISON_SIDE_PX:
            scale = MAX_COMPARISON_SIDE_PX / float(max(panel_width, panel_height))
            dagua_rgb = dagua_rgb.resize(
                (max(int(dagua_rgb.width * scale), 1), max(int(dagua_rgb.height * scale), 1)),
                Image.LANCZOS,
            )
            comp_rgb = comp_rgb.resize(
                (max(int(comp_rgb.width * scale), 1), max(int(comp_rgb.height * scale), 1)),
                Image.LANCZOS,
            )
            panel_width = dagua_rgb.width + comp_rgb.width
            panel_height = max(dagua_rgb.height, comp_rgb.height)
        canvas = Image.new("RGB", (panel_width, panel_height), "#FFFFFF")
        canvas.paste(dagua_rgb, (0, (panel_height - dagua_rgb.height) // 2))
        canvas.paste(comp_rgb, (dagua_rgb.width, (panel_height - comp_rgb.height) // 2))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    return output_path


def _render_competitor_first_available(
    context: CardRenderContext,
    output_root: Path,
    dimensions: Tuple[int, int],
    reference_cache_root: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """Render the first available competitor for a card.

    Parameters
    ----------
    context : CardRenderContext
        Resolved card context.
    output_root : pathlib.Path
        Pixel-diff output root.
    dimensions : tuple[int, int]
        Requested image dimensions.
    reference_cache_root : pathlib.Path | None, optional
        Existing gallery root whose competitor renders can be reused when a
        fresh competitor invocation fails. This keeps cairo-vs-Agg comparisons
        anchored to the same reference set.

    Returns
    -------
    tuple[pathlib.Path | None, str | None]
        Competitor PNG path and tool name, or ``(None, None)``.
    """

    graph_spec = graph_spec_from_dagua(context.graph)
    position_pairs = [(float(x), float(y)) for x, y in context.positions.detach().cpu().tolist()]
    for tool in context.competitor_tools:
        candidate = output_root / "competitors" / tool / f"{context.card_id}.png"
        try:
            rendered = render_competitor(tool, graph_spec, position_pairs, candidate, dimensions)
        except Exception as exc:
            print(
                f"warning: {context.card_id}: competitor {tool} failed: {exc}",
                file=sys.stderr,
            )
            if reference_cache_root is not None:
                cached = (
                    reference_cache_root
                    / "per_card_pixel_diff"
                    / "competitors"
                    / tool
                    / f"{context.card_id}.png"
                )
                if cached.exists():
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    if cached.resolve() != candidate.resolve():
                        shutil.copy2(cached, candidate)
                    return ensure_png_dimensions(candidate, dimensions), tool
            continue
        if rendered is not None and rendered.exists():
            return rendered, tool
        print(
            f"warning: {context.card_id}: competitor {tool} unavailable or unsupported",
            file=sys.stderr,
        )
    return None, None


def _process_card(
    context: CardRenderContext,
    gallery_root: Path,
    output_root: Path,
    backend: Optional[str] = None,
) -> Dict[str, object]:
    """Process one card and write its JSON/heatmap artifacts.

    Parameters
    ----------
    context : CardRenderContext
        Resolved card context.
    gallery_root : pathlib.Path
        Gallery root path.
    output_root : pathlib.Path
        Pixel-diff output root.
    backend : str | None, optional
        Matplotlib backend selector passed through to Dagua rendering.

    Returns
    -------
    dict[str, object]
        Per-card result payload.
    """

    del gallery_root
    if context.tier == "C" or not context.competitor_tools:
        payload = {
            "slug": context.card_id,
            "kind": context.kind,
            "tier": "C",
            "reason": context.tier_c_reason or "no competitor",
            "skipped": True,
        }
        write_json(output_root / "cards" / f"{context.card_id}.json", payload)
        return payload

    dimensions = DEFAULT_CARD_DIMS
    if max(dimensions) > MAX_COMPARISON_SIDE_PX:
        raise ValueError(f"Card dimensions exceed cap: {dimensions}")
    dagua_path = output_root / "dagua" / f"{context.card_id}.png"
    _save_dagua_render(context, dagua_path, dimensions, backend=backend)
    competitor_path, tool_used = _render_competitor_first_available(
        context,
        output_root,
        dimensions,
        reference_cache_root=DEFAULT_GALLERY_ROOT,
    )
    if competitor_path is None or tool_used is None:
        payload = {
            "slug": context.card_id,
            "kind": context.kind,
            "tier": "C",
            "reason": "no installed competitor rendered this card",
            "requested_tools": list(context.competitor_tools),
            "skipped": True,
        }
        write_json(output_root / "cards" / f"{context.card_id}.json", payload)
        return payload

    heatmap_path = output_root / "heatmaps" / f"{context.card_id}_heatmap.png"
    payload = {
        "slug": context.card_id,
        "kind": context.kind,
        "tier": context.tier,
        "tool_used": tool_used,
        "dagua_png": str(dagua_path),
        "competitor_png": str(competitor_path),
        "heatmap_png": str(heatmap_path),
        "comparison_png": str(
            output_root / "comparisons" / f"{context.card_id}_vs_{tool_used}.png"
        ),
        "skipped": False,
    }
    payload.update(_metrics(dagua_path, competitor_path, heatmap_path))
    _write_side_by_side(dagua_path, competitor_path, Path(str(payload["comparison_png"])))
    write_json(output_root / "cards" / f"{context.card_id}.json", payload)
    return payload


def _write_summary(results: Sequence[Mapping[str, object]], output_root: Path) -> Path:
    """Write aggregate JSON and Markdown summaries.

    Parameters
    ----------
    results : Sequence[Mapping[str, object]]
        Per-card result payloads.
    output_root : pathlib.Path
        Pixel-diff output root.

    Returns
    -------
    pathlib.Path
        Markdown summary path.
    """

    counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for result in results:
        tier = str(result.get("tier", "C"))
        counts[tier] = counts.get(tier, 0) + 1
    summary = {
        "counts": counts,
        "cards": list(results),
    }
    write_json(output_root / "summary.json", summary)
    lines = [
        "# Per-Card Pixel Diff Summary",
        "",
        f"- Tier A: {counts.get('A', 0)}",
        f"- Tier B: {counts.get('B', 0)}",
        f"- Tier C skipped: {counts.get('C', 0)}",
        "",
        "| Card | Tier | Tool | L1 | SSIM | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for result in results:
        if result.get("skipped"):
            lines.append(
                f"| {result['slug']} | {result.get('tier', 'C')} | - | - | - | "
                f"{result.get('reason', 'skipped')} |"
            )
        else:
            lines.append(
                f"| {result['slug']} | {result.get('tier')} | {result.get('tool_used')} | "
                f"{float(result.get('l1', 0.0)):.3f} | "
                f"{float(result.get('ssim', 0.0)):.4f} | heatmap written |"
            )
    path = output_root.parent / "per_card_pixel_diff_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _emit_hires(
    contexts: Mapping[str, CardRenderContext],
    card_ids: Sequence[str],
    gallery_root: Path,
    output_root: Path,
    backend: Optional[str] = None,
) -> None:
    """Emit separate hi-res Dagua and competitor PNGs for requested cards.

    Parameters
    ----------
    contexts : Mapping[str, CardRenderContext]
        Contexts keyed by card ID.
    card_ids : Sequence[str]
        Requested card IDs.
    gallery_root : pathlib.Path
        Gallery root path.
    output_root : pathlib.Path
        Pixel-diff output root.
    backend : str | None, optional
        Matplotlib backend selector passed through to Dagua rendering.

    Returns
    -------
    None
        Hi-res files are written under ``gallery_root/hires``.
    """

    del output_root
    dimensions = _fit_hires_dimensions(*DEFAULT_CARD_DIMS)
    for card_id in card_ids:
        context = contexts[card_id]
        destination = gallery_root / "hires" / card_id
        _save_dagua_render(context, destination / "dagua.png", dimensions, backend=backend)
        if context.tier == "C":
            continue
        graph_spec = graph_spec_from_dagua(context.graph)
        positions = [(float(x), float(y)) for x, y in context.positions.detach().cpu().tolist()]
        for tool in context.competitor_tools:
            rendered = render_competitor(
                tool,
                graph_spec,
                positions,
                destination / "competitor.png",
                dimensions,
            )
            if rendered is not None:
                break


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gallery-dir",
        "--gallery-root",
        dest="gallery_root",
        type=Path,
        default=DEFAULT_GALLERY_ROOT,
        help="Gallery audit root to evaluate",
    )
    parser.add_argument("--cards", type=str, default=None, help="Comma-separated card IDs")
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for per-card pixel-diff artifacts",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=[None, "agg", "cairo"],
        help="Matplotlib backend (default: auto-detect cairo if installed, else agg)",
    )
    parser.add_argument(
        "--hires",
        type=str,
        default=None,
        help="Comma-separated card IDs for hi-res output",
    )
    return parser.parse_args()


def main() -> None:
    """Run per-card pixel diffs.

    Returns
    -------
    None
        Artifacts are written to disk and a short summary is printed.
    """

    args = parse_args()
    card_filter = _parse_csv(args.cards)
    hires_filter = _parse_csv(args.hires)
    contexts = _build_contexts(card_filter)
    args.out.mkdir(parents=True, exist_ok=True)
    results = [
        _process_card(context, args.gallery_root, args.out, backend=args.backend)
        for context in contexts.values()
    ]
    summary_path = _write_summary(results, args.out)
    if hires_filter:
        _emit_hires(contexts, hires_filter, args.gallery_root, args.out, backend=args.backend)
    counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for result in results:
        tier = str(result.get("tier", "C"))
        counts[tier] = counts.get(tier, 0) + 1
    print(
        "per_card_pixel_diff "
        f"cards={len(results)} tier_a={counts.get('A', 0)} "
        f"tier_b={counts.get('B', 0)} tier_c={counts.get('C', 0)} "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
