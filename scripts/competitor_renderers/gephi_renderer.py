"""Gephi Toolkit competitor renderer.

Gephi Toolkit support is intentionally thin for this setup pass. The adapter
detects the local Toolkit JAR and returns ``None`` for per-card rendering until
a Java preview-export wrapper is added. This keeps Gephi in the registry without
blocking Tier A/B/C classification or smoke tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

GEPHI_JAR = Path.home() / ".local/share/gephi-toolkit/gephi-toolkit-0.10.0-all.jar"


def render(
    graph_spec: dict,
    positions: Sequence[Tuple[float, float]],
    output_path: Path,
    dimensions: Tuple[int, int],
    feature_overrides: Optional[dict] = None,
) -> Optional[Path]:
    """Return ``None`` until an automated Gephi preview exporter exists.

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
        Tool-native feature overrides.

    Returns
    -------
    pathlib.Path | None
        Always ``None`` for now; Gephi Toolkit is a library, not a CLI.
    """

    del graph_spec, positions, output_path, dimensions, feature_overrides
    if not GEPHI_JAR.exists():
        return None
    return None
