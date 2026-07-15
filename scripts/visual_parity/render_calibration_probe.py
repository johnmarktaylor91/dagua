# ruff: noqa: E402
"""Render the 8-panel VLM model-selection calibration probe (FINAL_DESIGN.md section 6).

The probe package is a SCORED, known-answer test set used to select which
vision-language model(s) audit the visual parity v2 loop -- it is rendered
once per model-selection round and never sent anywhere by this module (no
web access, no API calls; that is the orchestrator's job per
``select_vision_model.py``).

Panels (6 known defects + 2 false-positive controls, all <= 2000px):
  1. wrong_font            -- dagua font size mismatched vs the reference
  2. invisible_edge_stem   -- dagua edge stroke effectively invisible
  3. truncated_label       -- dagua label clipped vs the reference's full text
  4. arrow_fill_mismatch   -- dagua arrowhead fill mode flipped
  5. cluster_border_missing_one_side -- one dagua cluster border disabled
  6. size_normalization_artifact -- dagua panel re-thumbnailed (tw_scalehide)
  7. antialiased_residual  -- clean match, AA/hinting noise only (control)
  8. true_noop_match       -- byte-identical panels, zero diff (control)

A model that misses any of the 6 defects, or flags either control panel, is
rejected for auditing duty (FINAL_DESIGN.md section 6, step 5).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

import scripts.generate_calibration_suite as calibration

MAX_PROBE_SIDE_PX = 2000
PROBE_FIGSIZE: Tuple[float, float] = (9.0, 3.6)
DEFAULT_OUTPUT_DIR = "eval_output/visual_parity_v2/probe"


@dataclass(frozen=True)
class ProbePanel:
    """One calibration probe panel and its known-answer metadata.

    Parameters
    ----------
    panel_id
        Stable panel identifier.
    is_control
        Whether this panel is a false-positive control (no real defect).
    defect_description
        Human-readable description of the injected defect, or the control's
        rationale.
    build_scene
        Zero-argument callable returning a
        ``scripts.generate_calibration_suite.CalibrationScene``.
    """

    panel_id: str
    is_control: bool
    defect_description: str
    build_scene: Any = field(repr=False)


def _wrong_font_scene() -> calibration.CalibrationScene:
    """Build the wrong-font defect scene (dagua font size mismatched).

    Returns
    -------
    calibration.CalibrationScene
        Scene with a dagua-side font-size defect injected.
    """

    style = calibration._base_node_style(font_size=22.0)
    scene = calibration._node_scene([("Probe Node", style)], columns=1)
    scene.graphviz.node_attrs = {0: {"fontsize": "14.0"}}
    return scene


def _invisible_edge_stem_scene() -> calibration.CalibrationScene:
    """Build the invisible-edge-stem defect scene.

    Returns
    -------
    calibration.CalibrationScene
        Scene with a near-invisible dagua edge stroke.
    """

    return calibration._edge_scene(
        [("stem", calibration._base_edge_style(width=0.05, opacity=0.03))],
        columns=1,
    )


def _truncated_label_scene() -> calibration.CalibrationScene:
    """Build the truncated-label defect scene.

    Returns
    -------
    calibration.CalibrationScene
        Scene with a dagua-side clipped label vs the reference's full text.
    """

    style = calibration._base_node_style(min_width=70.0, min_height=40.0)
    scene = calibration._node_scene([("Trunc", style)], columns=1)
    scene.graphviz.node_attrs = {0: {"label": "Truncated Label Full Text"}}
    return scene


def _arrow_fill_mismatch_scene() -> calibration.CalibrationScene:
    """Build the arrow-fill-mismatch defect scene.

    Returns
    -------
    calibration.CalibrationScene
        Scene with a hollow dagua arrowhead vs a filled reference arrowhead.
    """

    return calibration._edge_scene(
        [("fill", calibration._base_edge_style(arrow="normal", arrow_fill="hollow"))],
        columns=1,
    )


def _cluster_border_missing_scene() -> calibration.CalibrationScene:
    """Build the cluster-border-missing-one-side defect scene.

    Returns
    -------
    calibration.CalibrationScene
        Scene with one dagua cluster border disabled.
    """

    no_border_style = calibration._base_cluster_style(stroke_width=0.0)
    scene = calibration._build_graph(
        node_labels=["A", "B"],
        positions=[(-40.0, 0.0), (40.0, 0.0)],
        edges=[(0, 1)],
        clusters=[("probe_cluster", [0, 1], no_border_style, "Probe", None)],
        figsize=(4.0, 3.2),
    )
    return scene


def _size_normalization_artifact_panel(scratch_dir: Path) -> Tuple[Path, Path]:
    """Render the size-normalization-artifact defect panel (tw_scalehide bait).

    Re-introduces the historical bbox-tight-plus-thumbnail defect on the
    dagua side only, so the composited panel shows a disproportionately
    shrunk/padded dagua image relative to the reference.

    Parameters
    ----------
    scratch_dir
        Scratch directory for the raw per-side renders (not the probe's
        output directory -- only the final composed panel belongs there).

    Returns
    -------
    tuple[Path, Path]
        Reference and dagua raw PNG paths.
    """

    scene = calibration._node_scene([("Scale", calibration._base_node_style())], columns=1)
    reference_path = scratch_dir / "reference.png"
    dagua_raw_path = scratch_dir / "dagua_raw.png"
    dagua_path = scratch_dir / "dagua.png"
    calibration._render_graphviz_png(
        calibration._build_graphviz_dot(scene), reference_path, scene.graphviz.engine
    )
    calibration._render_dagua_png(scene, dagua_raw_path)
    # Re-thumbnail onto a mismatched canvas -- the exact tw_scalehide defect:
    # bbox_inches="tight" plus a fixed-canvas thumbnail resize distorts scale
    # relative to the reference instead of a same-box content crop.
    with Image.open(dagua_raw_path) as raw:
        canvas = Image.new("RGBA", (400, 700), "#FFFFFF")
        raw_rgba = raw.convert("RGBA")
        raw_rgba.thumbnail((160, 160), Image.LANCZOS)
        canvas.paste(raw_rgba, (20, 20), raw_rgba)
        canvas.convert("RGB").save(dagua_path)
    return reference_path, dagua_path


def _control_scene() -> calibration.CalibrationScene:
    """Build the shared clean scene used by both false-positive controls.

    Returns
    -------
    calibration.CalibrationScene
        A plain, undefective scene.
    """

    return calibration._edge_scene(
        [("clean", calibration._base_edge_style())],
        columns=1,
    )


def _defect_panels() -> List[ProbePanel]:
    """Return the 6 known-defect probe panel definitions.

    Returns
    -------
    list[ProbePanel]
        Defect panels in fixed order.
    """

    return [
        ProbePanel(
            "wrong_font",
            False,
            "Dagua node font size (22pt) does not match the reference's declared font size (14pt).",
            _wrong_font_scene,
        ),
        ProbePanel(
            "invisible_edge_stem",
            False,
            "Dagua edge stroke is effectively invisible (width 0.05, opacity 0.03) "
            "while the reference draws a normal solid edge.",
            _invisible_edge_stem_scene,
        ),
        ProbePanel(
            "truncated_label",
            False,
            "Dagua node label is clipped to 'Trunc' while the reference shows the full label text.",
            _truncated_label_scene,
        ),
        ProbePanel(
            "arrow_fill_mismatch",
            False,
            "Dagua arrowhead renders hollow while the reference's 'normal' arrowhead is filled.",
            _arrow_fill_mismatch_scene,
        ),
        ProbePanel(
            "cluster_border_missing_one_side",
            False,
            "Dagua cluster border is disabled (stroke_width=0) while the "
            "reference draws a full cluster border.",
            _cluster_border_missing_scene,
        ),
        ProbePanel(
            "size_normalization_artifact",
            False,
            "Dagua panel is re-thumbnailed onto a mismatched canvas (the "
            "historical tw_scalehide defect), producing a disproportionate "
            "scale/crop relative to the reference.",
            None,
        ),
    ]


def _control_panels() -> List[ProbePanel]:
    """Return the 2 false-positive control panel definitions.

    Returns
    -------
    list[ProbePanel]
        Control panels in fixed order.
    """

    return [
        ProbePanel(
            "antialiased_residual",
            True,
            "Clean match: reference and dagua render the same scene through "
            "the normal path. Any residual is anti-aliasing/font-hinting "
            "noise, not a real defect. A model that flags this fails the "
            "false-positive check.",
            _control_scene,
        ),
        ProbePanel(
            "true_noop_match",
            True,
            "Byte-identical panels (the same dagua render used on both "
            "sides). Guaranteed zero diff. A model that flags this fails "
            "the false-positive check.",
            _control_scene,
        ),
    ]


def _ensure_within_cap(path: Path, cap_px: int = MAX_PROBE_SIDE_PX) -> None:
    """Downscale a composed panel in place if it exceeds the probe's pixel cap.

    ``matplotlib``'s ``bbox_inches="tight"`` sizes the saved PNG to the
    rendered content's bounding box, which can exceed the nominal
    ``figsize`` requested from ``_compose_comparison``. This guarantees the
    <= 2000px probe contract regardless of that layout variance.

    Parameters
    ----------
    path
        Composed panel PNG path (overwritten in place if downscaled).
    cap_px
        Maximum allowed side length in pixels.

    Returns
    -------
    None
        The file is rewritten in place when it exceeds the cap.
    """

    with Image.open(path) as image:
        longest = max(image.size)
        if longest <= cap_px:
            return
        scale = cap_px / longest
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        resized = image.convert("RGB").resize(new_size, Image.LANCZOS)
    resized.save(path)


def _render_scene_panel(panel: ProbePanel, output_root: Path) -> Path:
    """Render one scene-backed probe panel as a two-panel comparison.

    Parameters
    ----------
    panel
        Panel definition with a ``build_scene`` callable.
    output_root
        Destination directory for the composed PNG.

    Returns
    -------
    Path
        Path to the composed comparison PNG.
    """

    scene = panel.build_scene()
    output_path = output_root / f"{panel.panel_id}.png"
    with tempfile.TemporaryDirectory(prefix="dagua_probe_") as temp_dir:
        temp_root = Path(temp_dir)
        graphviz_path = temp_root / "graphviz.png"
        dagua_path = temp_root / "dagua.png"
        calibration._render_graphviz_png(
            calibration._build_graphviz_dot(scene), graphviz_path, scene.graphviz.engine
        )
        calibration._render_dagua_png(scene, dagua_path)
        calibration._compose_comparison(
            reference_image=graphviz_path,
            dagua_image=dagua_path,
            output_path=output_path,
            description=panel.defect_description,
            category="calibration_probe",
            case_id=panel.panel_id,
            figsize=PROBE_FIGSIZE,
        )
    return output_path


def _render_true_noop_panel(output_root: Path) -> Path:
    """Render the byte-identical true-noop control panel.

    Parameters
    ----------
    output_root
        Destination directory for the composed PNG.

    Returns
    -------
    Path
        Path to the composed comparison PNG.
    """

    scene = _control_scene()
    output_path = output_root / "true_noop_match.png"
    with tempfile.TemporaryDirectory(prefix="dagua_probe_") as temp_dir:
        temp_root = Path(temp_dir)
        dagua_path = temp_root / "dagua.png"
        calibration._render_dagua_png(scene, dagua_path)
        # Byte-identical: use the SAME raw render for both composited sides.
        calibration._compose_comparison(
            reference_image=dagua_path,
            dagua_image=dagua_path,
            output_path=output_path,
            description="Byte-identical panels (guaranteed zero diff).",
            category="calibration_probe",
            case_id="true_noop_match",
            figsize=PROBE_FIGSIZE,
        )
    return output_path


def render_calibration_probe(output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    """Render the full 8-panel calibration probe and write the defect manifest.

    Parameters
    ----------
    output_dir
        Destination directory for the 8 panels and ``defect_manifest.json``.

    Returns
    -------
    dict[str, Any]
        The defect manifest payload (also written to disk).
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_panels: List[Dict[str, Any]] = []

    for panel in _defect_panels():
        if panel.panel_id == "size_normalization_artifact":
            output_path = output_root / f"{panel.panel_id}.png"
            with tempfile.TemporaryDirectory(prefix="dagua_probe_scalehide_") as scratch:
                reference_path, dagua_path = _size_normalization_artifact_panel(Path(scratch))
                calibration._compose_comparison(
                    reference_image=reference_path,
                    dagua_image=dagua_path,
                    output_path=output_path,
                    description=panel.defect_description,
                    category="calibration_probe",
                    case_id=panel.panel_id,
                    figsize=PROBE_FIGSIZE,
                )
        else:
            output_path = _render_scene_panel(panel, output_root)
        manifest_panels.append(
            {
                "panel_id": panel.panel_id,
                "is_control": panel.is_control,
                "defect_description": panel.defect_description,
                "image_path": str(output_path),
            }
        )

    for panel in _control_panels():
        if panel.panel_id == "true_noop_match":
            output_path = _render_true_noop_panel(output_root)
        else:
            output_path = _render_scene_panel(panel, output_root)
        manifest_panels.append(
            {
                "panel_id": panel.panel_id,
                "is_control": panel.is_control,
                "defect_description": panel.defect_description,
                "image_path": str(output_path),
            }
        )

    for row in manifest_panels:
        _ensure_within_cap(Path(row["image_path"]))
        with Image.open(row["image_path"]) as image:
            assert max(image.size) <= MAX_PROBE_SIDE_PX, (
                f"probe panel {row['panel_id']} still exceeds the {MAX_PROBE_SIDE_PX}px "
                f"cap after downscaling: {image.size}"
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "panel_count": len(manifest_panels),
        "defect_count": sum(1 for row in manifest_panels if not row["is_control"]),
        "control_count": sum(1 for row in manifest_panels if row["is_control"]),
        "panels": manifest_panels,
    }
    manifest_path = output_root / "defect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """Parse CLI arguments and render the calibration probe.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = render_calibration_probe(args.out)
    print(f"Rendered {manifest['panel_count']} probe panels to {manifest['output_dir']}")
    print(f"  defects: {manifest['defect_count']}, controls: {manifest['control_count']}")
    print(f"Manifest: {Path(args.out) / 'defect_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
