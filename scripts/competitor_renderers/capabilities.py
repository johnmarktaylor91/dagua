# ruff: noqa: E402
"""Adapter capability self-report and version preflight (Lane D hardening).

FINAL_DESIGN.md correction F3: competitor adapters GATE NOTHING until their
fixed-geometry and per-element-style capability is proven, cell by cell.
This module is the single machine-readable source of truth for each
adapter's verified capability facts -- it does not change any adapter's
rendering behavior (no behavioral rewrites this sprint; capability honesty,
not new features). ``scripts.visual_parity.coverage`` (Lane C) is expected
to seed its ``adapter_capabilities`` rows from ``ADAPTER_CAPABILITIES``
below rather than re-deriving the same facts a second time.

Verified facts (re-checked against the adapters in this package):
- gephi: ``gephi_renderer.render()`` always returns ``None`` -- the Gephi
  Toolkit is a library, not a CLI, and no automated preview exporter exists.
- mermaid: ``mermaid_renderer.render()`` discards the ``positions`` argument
  (``del positions``) -- Mermaid always chooses its own layout. Per-node
  shape varies, but no per-node fill/stroke/font styling is emitted.
- graphviz (this competitor adapter, distinct from the dot-layout-then-
  inject Track G path): ``graphviz_renderer.render()`` discards
  ``positions`` and maps unknown shapes to ``ellipse`` via
  ``GRAPHVIZ_SHAPES.get(shape, "ellipse")``.
- cytoscape: ``cytoscape_renderer._style()`` reads only ``nodes[0]`` /
  ``edges[0]`` and applies that single style to the whole ``node``/``edge``
  selector -- every element in the graph gets the first element's style.
  Fixed positions ARE honored (``layout: {name: "preset"}``).
- d3: ``d3_renderer.render()`` honors fixed positions and applies per-node
  style individually, but only exposes a minimal line/rect/ellipse feature
  set (no gradients, patterns, or clusters).

``gate_eligible=False`` for every adapter until proven per-cell (F3):
capability facts alone do not authorize gating; a future sprint that
verifies per-cell fidelity may flip individual capability rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.competitor_renderers.utils import command_available, write_json
from scripts.visual_parity.types import AdapterCapability

DEFAULT_VERSIONS_PATH = "refcache/versions.json"

ADAPTER_CAPABILITIES: Dict[str, AdapterCapability] = {
    "graphviz": AdapterCapability(
        adapter="graphviz",
        fixed_positions=False,
        per_element_styles=True,
        deterministic=True,
        gate_eligible=False,
        evidence=(
            "graphviz_renderer.py: render() discards `positions` (`del positions`); "
            "node shapes fall back to ellipse via GRAPHVIZ_SHAPES.get(shape, 'ellipse') "
            "for unmapped shapes. Use the dot-layout-then-inject Track G path for "
            "geometry-matched graphviz cells, not this competitor adapter."
        ),
    ),
    "mermaid": AdapterCapability(
        adapter="mermaid",
        fixed_positions=False,
        per_element_styles=False,
        deterministic=True,
        gate_eligible=False,
        evidence=(
            "mermaid_renderer.py: render() discards `positions` (`del positions`); "
            "Mermaid always chooses its own layout. _build_mermaid() varies node "
            "shape per node but emits no per-node fill/stroke/font styling."
        ),
    ),
    "cytoscape": AdapterCapability(
        adapter="cytoscape",
        fixed_positions=True,
        per_element_styles=False,
        deterministic=True,
        gate_eligible=False,
        evidence=(
            "cytoscape_renderer.py: _style() reads only nodes[0]/edges[0] and "
            "applies that single style globally to the 'node'/'edge' selector -- "
            "every element gets the first element's style. Fixed positions ARE "
            "honored via layout: {name: 'preset'}."
        ),
    ),
    "d3": AdapterCapability(
        adapter="d3",
        fixed_positions=True,
        per_element_styles=True,
        deterministic=True,
        gate_eligible=False,
        evidence=(
            "d3_renderer.py: honors fixed positions (normalize_positions) and "
            "applies per-node style individually, but is a minimal "
            "line/rect/ellipse renderer -- no gradients, patterns, or clusters. "
            "Smoke reference only."
        ),
    ),
    "gephi": AdapterCapability(
        adapter="gephi",
        fixed_positions=False,
        per_element_styles=False,
        deterministic=False,
        gate_eligible=False,
        evidence=(
            "gephi_renderer.py: render() always returns None. Gephi Toolkit is "
            "a library, not a CLI; no automated preview-export wrapper exists."
        ),
    ),
}


def capability_rows() -> list[AdapterCapability]:
    """Return the adapter capability rows in registry order.

    Returns
    -------
    list[AdapterCapability]
        One capability record per registered competitor adapter.
    """

    return list(ADAPTER_CAPABILITIES.values())


def _run_version(command: list[str]) -> Optional[str]:
    """Return a tool's version string, or ``None`` when it cannot be probed.

    Parameters
    ----------
    command
        Full command (executable + args) that prints a version string.

    Returns
    -------
    str or None
        Stripped combined stdout/stderr output, or ``None`` on failure.
    """

    if not command_available(command[0]):
        return None
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip()
    return output or None


def probe_versions() -> Dict[str, Optional[str]]:
    """Probe installed versions for every competitor adapter's toolchain.

    Returns
    -------
    dict[str, str or None]
        Tool name to version string (or ``None`` when unavailable),
        covering ``dot`` (graphviz adapter + Track G reference), ``mmdc``
        (mermaid), ``node`` (cytoscape/d3 runtime), and the Gephi Toolkit
        jar presence.
    """

    from scripts.competitor_renderers.gephi_renderer import GEPHI_JAR

    versions: Dict[str, Optional[str]] = {
        "dot": _run_version(["dot", "-V"]),
        "mmdc": _run_version(["mmdc", "--version"]),
        "node": _run_version(["node", "--version"]),
    }
    versions["gephi_toolkit"] = str(GEPHI_JAR) if GEPHI_JAR.exists() else None
    return versions


def print_versions(output_path: str | Path = DEFAULT_VERSIONS_PATH) -> Dict[str, Optional[str]]:
    """Probe adapter toolchain versions and write ``refcache/versions.json``.

    Parameters
    ----------
    output_path
        Destination JSON path (default ``refcache/versions.json``).

    Returns
    -------
    dict[str, str or None]
        The probed version map, identical to what was written to disk.
    """

    versions = probe_versions()
    write_json(Path(output_path), versions)
    return versions


def main() -> int:
    """Parse CLI arguments and run the requested preflight action.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-versions",
        action="store_true",
        help="Probe adapter toolchain versions and write refcache/versions.json.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_VERSIONS_PATH,
        help="Destination path for --print-versions output.",
    )
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help="Print the adapter capability table as JSON and exit.",
    )
    args = parser.parse_args()

    if args.capabilities:
        print(json.dumps([asdict(row) for row in capability_rows()], indent=2, sort_keys=True))
        return 0

    if args.print_versions:
        versions = print_versions(args.out)
        print(f"Wrote {args.out}")
        for tool, version in sorted(versions.items()):
            print(f"  {tool}: {version or 'not found'}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
