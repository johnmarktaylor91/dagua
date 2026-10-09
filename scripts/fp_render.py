"""Render fresh-pairs drawings and A|B composites for the pairwise rejudge.

Neutral instrument-style rendering after A10 ``FRAME-1`` / ``ordinary-unlabeled``:
equal-size white candidate canvases (950 x 950 px), robust display window
(trimmed span expanded 20% each side, aspect preserved), constant 1 px
strokes, real node boxes with no labels, straight edges, arrowheads iff the
graph is directed, no cluster boundaries, only ``A`` / ``B`` captions, a thin
black divider, composite long side <= 2000 px. A locator inset (approximating
``LOCATOR-1``) is drawn when node ink falls outside the window.

Input ``pairs.json``: list of
``{"pair_id", "graph", "x": {"layout": "<engine>@<tag>"}, "y": {...}}``.
Writes ``<out>/composites/<pair_id>__<xy|yx>.png`` (first letter is the A side)
and ``<out>/drawings/<pair_id>__<x|y>.png`` plus ``render_manifest.json`` with
sha256 for every file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

CANVAS_PX = 950  # A10 FRAME-1: each candidate column is at most 950 px wide
CAPTION_PX = 40
DIVIDER_PX = 4
SS = 2  # supersampling factor for antialiasing
WINDOW_MARGIN = 0.20  # A10 FRAME-1: expand robust span by 0.20 on each side
TRIM_FRACTION = 0.02  # robust-core trim per side when N >= N_SMALL
N_SMALL = 50
MIN_BOX_PX = 3.0
ARROW_PX = 9.0
INK = (0, 0, 0)
FILL = (255, 255, 255)
LOCATOR_FRACTION = 0.15


def _quantile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolated quantile of a sorted list."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def display_window(pos: List[List[float]], sizes: List[List[float]]) -> Tuple[float, float, float, float]:
    """Return the robust display window ``(x0, x1, y0, y1)`` in world units.

    Parameters
    ----------
    pos : list of [x, y]
        Node centers.
    sizes : list of [w, h]
        Node box sizes in the same units.
    """
    n = len(pos)
    trim = 0.0 if n < N_SMALL else TRIM_FRACTION
    out = []
    for axis in (0, 1):
        lo_edges = sorted(p[axis] - s[axis] / 2 for p, s in zip(pos, sizes))
        hi_edges = sorted(p[axis] + s[axis] / 2 for p, s in zip(pos, sizes))
        lo = _quantile(lo_edges, trim)
        hi = _quantile(hi_edges, 1.0 - trim)
        span = max(hi - lo, 1e-6)
        out.extend([lo - WINDOW_MARGIN * span, hi + WINDOW_MARGIN * span])
    return out[0], out[1], out[2], out[3]


def _clip_to_box(cx: float, cy: float, hw: float, hh: float, dx: float, dy: float) -> Tuple[float, float]:
    """Point where a ray from a box center along (dx, dy) leaves the box."""
    if dx == 0 and dy == 0:
        return cx, cy
    tx = hw / abs(dx) if dx else float("inf")
    ty = hh / abs(dy) if dy else float("inf")
    t = min(tx, ty)
    return cx + dx * t, cy + dy * t


def draw_drawing(
    pos: List[List[float]],
    graph: Dict[str, Any],
    px: int = CANVAS_PX,
) -> Image.Image:
    """Render one candidate canvas as an RGB image of ``px`` x ``px``."""
    sizes = graph["node_sizes"]
    edges = graph["edges"]
    directed = bool(graph["directed"])
    x0, x1, y0, y1 = display_window(pos, sizes)
    ww, wh = x1 - x0, y1 - y0
    scale = min(px / ww, px / wh)
    ox = (px - ww * scale) / 2
    oy = (px - wh * scale) / 2
    big = px * SS
    img = Image.new("RGB", (big, big), FILL)
    d = ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (ox + (x - x0) * scale) * SS, (oy + (y1 - y) * scale) * SS  # y up, like dagua

    half = []
    for p, s in zip(pos, sizes):
        hw = max(s[0] * scale / 2, MIN_BOX_PX / 2)
        hh = max(s[1] * scale / 2, MIN_BOX_PX / 2)
        half.append((hw, hh))

    # edges first (under nodes), in pixel space
    centers = [to_px(p[0], p[1]) for p in pos]
    lw = max(1, SS)
    for u, v in edges:
        if u == v:
            cx, cy = centers[u]
            hw, hh = half[u]
            r = max(hw, hh, 6 * SS) * 0.9
            d.ellipse([cx + hw - r * 0.3, cy - hh - r * 1.6, cx + hw + r * 1.3, cy - hh + r * 0.4], outline=INK, width=lw)
            continue
        (ux, uy), (vx, vy) = centers[u], centers[v]
        dx, dy = vx - ux, vy - uy
        if dx == 0 and dy == 0:
            continue
        sx, sy = _clip_to_box(ux, uy, half[u][0] * SS, half[u][1] * SS, dx, dy)
        ex, ey = _clip_to_box(vx, vy, half[v][0] * SS, half[v][1] * SS, -dx, -dy)
        d.line([(sx, sy), (ex, ey)], fill=INK, width=lw)
        if directed:
            ln = math.hypot(ex - sx, ey - sy)
            if ln > 1e-6:
                ux_, uy_ = (ex - sx) / ln, (ey - sy) / ln
                a = ARROW_PX * SS
                bx, by = ex - ux_ * a, ey - uy_ * a
                nx_, ny_ = -uy_, ux_
                d.polygon([(ex, ey), (bx + nx_ * a * 0.35, by + ny_ * a * 0.35), (bx - nx_ * a * 0.35, by - ny_ * a * 0.35)], fill=INK)
    for (cx, cy), (hw, hh) in zip(centers, half):
        d.rectangle([cx - hw * SS, cy - hh * SS, cx + hw * SS, cy + hh * SS], fill=FILL, outline=INK, width=lw)

    out = img.resize((px, px), Image.LANCZOS)

    # LOCATOR-1 approximation: inset showing every center when ink escapes the window
    escaped = [i for i, p in enumerate(pos) if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1)]
    if escaped:
        iw = int(px * LOCATOR_FRACTION * 1.0)
        pad = 8
        od = ImageDraw.Draw(out)
        bx0, by0 = px - iw - pad, pad
        od.rectangle([bx0, by0, bx0 + iw, by0 + iw], fill=FILL, outline=INK, width=1)
        xs = [p[0] for p in pos] + [x0, x1]
        ys = [p[1] for p in pos] + [y0, y1]
        lx0, lx1, ly0, ly1 = min(xs), max(xs), min(ys), max(ys)
        ls = min((iw - 6) / max(lx1 - lx0, 1e-6), (iw - 6) / max(ly1 - ly0, 1e-6))
        def lp(x: float, y: float) -> Tuple[float, float]:
            return bx0 + 3 + (x - lx0) * ls, by0 + iw - 3 - (y - ly0) * ls
        (wx0, wy0), (wx1, wy1) = lp(x0, y0), lp(x1, y1)
        od.rectangle([min(wx0, wx1), min(wy0, wy1), max(wx0, wx1), max(wy0, wy1)], outline=INK, width=1)
        for i, p in enumerate(pos):
            qx, qy = lp(p[0], p[1])
            od.rectangle([qx - 1, qy - 1, qx + 1, qy + 1], fill=INK)
        od.text((bx0, by0 + iw + 2), "box = main view; dots = content beyond", fill=INK, font=_font(11))
    return out


_FONT_CACHE: Dict[int, Any] = {}


def _font(size: int) -> Any:
    """Return a bold-ish TrueType font if available, else PIL's default."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(cand, size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def composite(a: Image.Image, b: Image.Image) -> Image.Image:
    """Place two canvases side by side with ``A`` / ``B`` captions and a divider."""
    w = a.width + DIVIDER_PX + b.width
    h = CAPTION_PX + a.height
    img = Image.new("RGB", (w, h), FILL)
    img.paste(a, (0, CAPTION_PX))
    img.paste(b, (a.width + DIVIDER_PX, CAPTION_PX))
    d = ImageDraw.Draw(img)
    d.rectangle([a.width, 0, a.width + DIVIDER_PX - 1, h], fill=INK)
    d.text((12, 6), "A", fill=INK, font=_font(28))
    d.text((a.width + DIVIDER_PX + 12, 6), "B", fill=INK, font=_font(28))
    return img


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_layouts(paths: Sequence[Path]) -> Dict[Tuple[str, str], List[List[float]]]:
    """Index positions by ``(graph, "<engine>@<tag>")``."""
    out: Dict[Tuple[str, str], List[List[float]]] = {}
    for path in paths:
        tag = path.stem.split("layouts-", 1)[1]
        for line in path.open():
            row = json.loads(line)
            if row.get("pos") is not None:
                out[(row["graph"], f"{row['engine']}@{tag}")] = row["pos"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render all requested pairs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", type=Path, required=True)
    ap.add_argument("--layouts", type=Path, nargs="+", required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    graphs = json.loads(args.graphs.read_text())
    layouts = load_layouts(args.layouts)
    pairs = json.loads(args.pairs.read_text())
    (args.out / "composites").mkdir(parents=True, exist_ok=True)
    (args.out / "drawings").mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    for i, pair in enumerate(pairs):
        g = graphs[pair["graph"]]
        canv = {}
        for side in ("x", "y"):
            canv[side] = draw_drawing(layouts[(pair["graph"], pair[side]["layout"])], g)
            p = args.out / "drawings" / f"{pair['pair_id']}__{side}.png"
            canv[side].save(p, optimize=True)
            manifest[f"drawings/{p.name}"] = sha256_file(p)
        for order in ("xy", "yx"):
            p = args.out / "composites" / f"{pair['pair_id']}__{order}.png"
            composite(canv[order[0]], canv[order[1]]).save(p, optimize=True)
            manifest[f"composites/{p.name}"] = sha256_file(p)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(pairs)}", flush=True)
    (args.out / "render_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
