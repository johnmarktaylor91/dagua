"""Unit pins for d3-force-compatible ops."""

from __future__ import annotations

import json
import subprocess

import torch

from dagua.layout.ops.d3force import (
    _d3force_quadtree_accumulation_rows,
    d3force_lcg_values,
    d3force_phyllotaxis_positions,
)


def test_d3force_lcg_matches_reference_first_20_values() -> None:
    """Pin d3-force's LCG sequence.

    Returns
    -------
    None
        The first 20 values must match a Node reference script bit-for-bit.
    """
    expected = [
        0.23645552527159452,
        0.3692706737201661,
        0.5042420323006809,
        0.7048832636792213,
        0.05054362863302231,
        0.3695183543022722,
        0.7747629624791443,
        0.556188570568338,
        0.0164932357147336,
        0.6392460397910327,
        0.2504511415027082,
        0.4223777682054788,
        0.5906901974231005,
        0.8369336591567844,
        0.23507591942325234,
        0.980845961952582,
        0.8608870944008231,
        0.32687550294212997,
        0.6826027217321098,
        0.5314591128844768,
    ]
    assert d3force_lcg_values(seed=1, count=20) == expected


def test_d3force_phyllotaxis_matches_reference_initial_nodes() -> None:
    """Pin d3-force's missing-position initialization spiral.

    Returns
    -------
    None
        First six phyllotaxis positions must match d3-force.
    """
    expected = torch.tensor(
        [
            [7.0710678118654755, 0.0],
            [-9.03088751750192, 8.273032735715967],
            [1.3823220809823638, -15.750847141167634],
            [11.382848792909423, 14.846910566099618],
            [-20.88892748977138, -3.694957148205299],
            [19.78781566111266, -12.587388583889217],
        ],
        dtype=torch.float64,
    )
    actual = d3force_phyllotaxis_positions(6)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_d3force_quadtree_accumulation_matches_d3_quadtree_grid_probe() -> None:
    """Pin d3-quadtree topology and accumulated charge centroids.

    Returns
    -------
    None
        The local quadtree dump must match upstream d3-quadtree after the
        forceManyBody accumulation callback on a 25-node probe.
    """
    positions = [tuple(row.tolist()) for row in d3force_phyllotaxis_positions(25)]
    actual = [list(row) for row in _d3force_quadtree_accumulation_rows(positions)]
    script = f"""
import {{quadtree}} from "d3-quadtree";
const positions = {json.dumps(positions)};
const nodes = positions.map(([x, y], index) => ({{index, x, y}}));
const strengths = new Array(nodes.length).fill(-30);
function accumulate(quad) {{
  let strength = 0, q, c, weight = 0, x, y, i;
  if (quad.length) {{
    for (x = y = i = 0; i < 4; ++i) {{
      if ((q = quad[i]) && (c = Math.abs(q.value))) {{
        strength += q.value, weight += c, x += c * q.x, y += c * q.y;
      }}
    }}
    quad.x = x / weight;
    quad.y = y / weight;
  }} else {{
    q = quad;
    q.x = q.data.x;
    q.y = q.data.y;
    do strength += strengths[q.data.index];
    while (q = q.next);
  }}
  quad.value = strength;
}}
const tree = quadtree(nodes, d => d.x, d => d.y).visitAfter(accumulate);
const rows = [];
function collect(node, x0, y0, x1, y1) {{
  if (node.length) {{
    rows.push(["internal", -1, node.x, node.y, node.value, x0, y0, x1, y1]);
    const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
    if (node[0]) collect(node[0], x0, y0, xm, ym);
    if (node[1]) collect(node[1], xm, y0, x1, ym);
    if (node[2]) collect(node[2], x0, ym, xm, y1);
    if (node[3]) collect(node[3], xm, ym, x1, y1);
  }} else {{
    let leaf = node;
    do {{
      rows.push(["leaf", leaf.data.index, node.x, node.y, node.value, x0, y0, x1, y1]);
    }} while (leaf = leaf.next);
  }}
}}
const extent = tree.extent();
collect(tree.root(), extent[0][0], extent[0][1], extent[1][0], extent[1][1]);
process.stdout.write(JSON.stringify(rows));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = json.loads(result.stdout)
    assert actual == expected
