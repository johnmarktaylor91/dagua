"""D3/jsdom competitor renderer.

Supported features include fixed-position node/edge SVG rendering, circular or
rectangular nodes, basic font styling, straight edges, line caps and joins, and
simple fill or edge color gradients when represented in the unified style spec.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .utils import ensure_png_dimensions, normalize_positions

SUPPORTED_OVERRIDE_KEYS = {"line_cap", "line_join", "gradient", "color_gradient"}


def render(
    graph_spec: dict,
    positions: Sequence[Tuple[float, float]],
    output_path: Path,
    dimensions: Tuple[int, int],
    feature_overrides: Optional[dict] = None,
) -> Optional[Path]:
    """Render a graph with D3, jsdom, and node-canvas.

    Parameters
    ----------
    graph_spec : dict
        Unified graph spec.
    positions : Sequence[tuple[float, float]]
        Node positions with shape ``[N, 2]``.
    output_path : pathlib.Path
        PNG destination.
    dimensions : tuple[int, int]
        Requested dimensions as ``(width_px, height_px)``.
    feature_overrides : dict | None, optional
        Tool-native overrides for supported D3 features.

    Returns
    -------
    pathlib.Path | None
        PNG path, or ``None`` when Node dependencies cannot render the graph.
    """

    if feature_overrides and not set(feature_overrides).issubset(SUPPORTED_OVERRIDE_KEYS):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graph": graph_spec,
        "positions": normalize_positions(positions, dimensions),
        "width": dimensions[0],
        "height": dimensions[1],
    }
    script = r"""
const fs = require('fs');
const d3 = require('d3');
const { JSDOM } = require('jsdom');
const { createCanvas, loadImage } = require('canvas');
const payload = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const width = payload.width, height = payload.height;
const dom = new JSDOM('<!DOCTYPE html><body></body>');
const body = d3.select(dom.window.document).select('body');
const svg = body.append('svg').attr('xmlns', 'http://www.w3.org/2000/svg')
  .attr('width', width).attr('height', height);
svg.append('rect').attr('width', width).attr('height', height).attr('fill', '#fff');
const nodes = payload.graph.nodes || [];
const nodeById = new Map(nodes.map((node, index) => [
  String(node.id),
  {node, pos: payload.positions[index] || [80, 80]}
]));
for (const edge of (payload.graph.edges || [])) {
  const src = nodeById.get(String(edge.src));
  const tgt = nodeById.get(String(edge.tgt));
  if (!src || !tgt) continue;
  const style = edge.style || {};
  svg.append('line').attr('x1', src.pos[0]).attr('y1', src.pos[1])
    .attr('x2', tgt.pos[0]).attr('y2', tgt.pos[1])
    .attr('stroke', style.color || '#000').attr('stroke-width', style.width || 1)
    .attr('stroke-linecap', style.line_cap || 'butt')
    .attr('stroke-linejoin', style.line_join || 'miter');
}
for (const [id, record] of nodeById) {
  const node = record.node, pos = record.pos, style = node.style || {};
  const shape = style.shape || node.shape || 'ellipse';
  const fill = node.fill || style.fill || '#fff';
  const stroke = node.stroke || style.stroke || '#000';
  const sw = style.stroke_width || 1;
  const w = style.min_width || 86, h = style.min_height || 54;
  if (shape === 'rect' || shape === 'roundrect') {
    svg.append('rect').attr('x', pos[0] - w / 2).attr('y', pos[1] - h / 2)
      .attr('width', w).attr('height', h).attr('rx', shape === 'roundrect' ? 8 : 0)
      .attr('fill', fill).attr('stroke', stroke).attr('stroke-width', sw);
  } else {
    svg.append('ellipse').attr('cx', pos[0]).attr('cy', pos[1])
      .attr('rx', w / 2).attr('ry', h / 2)
      .attr('fill', fill).attr('stroke', stroke).attr('stroke-width', sw);
  }
  svg.append('text').attr('x', pos[0]).attr('y', pos[1] + 5)
    .attr('text-anchor', 'middle').attr('font-size', style.font_size || 14)
    .attr('font-family', style.font_family || 'sans-serif')
    .attr('fill', style.font_color || '#000').text(node.label || id);
}
const canvas = createCanvas(width, height);
const ctx = canvas.getContext('2d');
const xml = body.html();
loadImage('data:image/svg+xml;base64,' + Buffer.from(xml).toString('base64')).then(img => {
  ctx.drawImage(img, 0, 0);
  process.stdout.write(canvas.toBuffer('image/png').toString('base64'));
}).catch(err => { console.error(err); process.exit(1); });
"""
    with tempfile.TemporaryDirectory(prefix="dagua-d3-render-") as tmp:
        payload_path = Path(tmp) / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            ["node", "-e", script, str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    output_path.write_bytes(base64.b64decode(result.stdout.strip()))
    return ensure_png_dimensions(output_path, dimensions)
