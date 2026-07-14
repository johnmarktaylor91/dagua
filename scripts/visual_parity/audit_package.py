"""Bundle visual parity round artifacts for VLM audits.

The audit-visible package contains only the images, metric summaries, and
prompt variables an auditor needs. Canary identity is written to a separate
harness-only sidecar so the audit prompt and manifest cannot leak which panel
was deliberately corrupted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from PIL import Image

MAX_AUDIT_IMAGE_SIDE_PX = 2000
AUDIT_PACKAGE_SCHEMA_VERSION = 1
DEFAULT_PROMPT_PATH = ".project-context/research/sprint_visual_parity_v2/prompts/audit_v2.md"
DEFAULT_OUTPUT_DIR = "eval_output/visual_parity_v2/audit_package"


@dataclass(frozen=True)
class AuditPackageResult:
    """Paths and payloads produced by ``build_audit_package``.

    Parameters
    ----------
    output_dir
        Root directory of the generated package.
    manifest_path
        Audit-visible manifest path.
    sidecar_path
        Harness-only sidecar path.
    manifest
        Audit-visible manifest payload.
    sidecar
        Harness-only sidecar payload.
    """

    output_dir: Path
    manifest_path: Path
    sidecar_path: Path
    manifest: Dict[str, Any]
    sidecar: Dict[str, Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON with a trailing newline.

    Parameters
    ----------
    path
        Destination JSON file path.
    payload
        JSON-compatible mapping to write.

    Returns
    -------
    None
        The file is written to disk.
    """

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_image_under_cap(source: Path, destination: Path, cap_px: int) -> Dict[str, Any]:
    """Copy an image into the package, downscaling if needed.

    Parameters
    ----------
    source
        Source image path.
    destination
        Destination PNG path inside the package.
    cap_px
        Maximum allowed side length.

    Returns
    -------
    dict[str, Any]
        Image metadata with original and packaged dimensions.
    """

    with Image.open(source) as image:
        original_size = [int(image.width), int(image.height)]
        output = image.convert("RGB")
        longest = max(output.size)
        if longest > cap_px:
            scale = cap_px / longest
            new_size = (max(1, round(output.width * scale)), max(1, round(output.height * scale)))
            output = output.resize(new_size, Image.LANCZOS)
        packaged_size = [int(output.width), int(output.height)]
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.save(destination)

    return {
        "original_path": str(source),
        "original_size_px": original_size,
        "packaged_size_px": packaged_size,
        "resized": original_size != packaged_size,
    }


def _copy_artifact(source: Path, destination_dir: Path) -> str:
    """Copy a non-image artifact into the package.

    Parameters
    ----------
    source
        Source artifact path.
    destination_dir
        Destination directory inside the package.

    Returns
    -------
    str
        Relative destination path from the package root.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return str(destination.relative_to(destination_dir.parent))


def _generic_panel_id(index: int) -> str:
    """Return the audit-visible panel id for a 1-based package index.

    Parameters
    ----------
    index
        One-based panel index.

    Returns
    -------
    str
        Generic panel identifier that does not reveal canary status.
    """

    return f"pair_{index:03d}"


def build_audit_package(
    round_image_paths: Sequence[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    metric_summary_paths: Sequence[str | Path] = (),
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    round_id: str = "round_unset",
    geometry_mode: str = "injected",
    roi_crop_paths: Sequence[str | Path] = (),
    pixel_triage_summary_path: str | Path | None = None,
    prior_findings: Sequence[str] = (),
    canary_image_path: str | Path | None = None,
    canary_tripwire_id: str | None = None,
    image_cap_px: int = MAX_AUDIT_IMAGE_SIDE_PX,
) -> AuditPackageResult:
    """Build an audit-visible package plus harness-only canary sidecar.

    Parameters
    ----------
    round_image_paths
        Round comparison images to include before the optional canary.
    output_dir
        Destination package directory.
    metric_summary_paths
        Declarative metric summary files to copy into ``metrics/``.
    prompt_path
        Audit prompt skeleton path copied into ``prompts/``.
    round_id
        Round identifier recorded in the manifest.
    geometry_mode
        Geometry mode label for the prompt variables.
    roi_crop_paths
        Optional ROI crop images copied into ``rois/``.
    pixel_triage_summary_path
        Optional pixel triage summary copied into ``metrics/``.
    prior_findings
        Prior finding ids or notes to re-check.
    canary_image_path
        Optional corrupted tripwire render to inject as one package panel.
    canary_tripwire_id
        Optional tripwire id for the harness sidecar.
    image_cap_px
        Maximum image side length for all packaged images.

    Returns
    -------
    AuditPackageResult
        Generated paths and JSON payloads.
    """

    output_root = Path(output_dir)
    images_dir = output_root / "images"
    metrics_dir = output_root / "metrics"
    prompts_dir = output_root / "prompts"
    rois_dir = output_root / "rois"
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_source = Path(prompt_path)
    prompt_relative = _copy_artifact(prompt_source, prompts_dir)

    metric_relatives = [_copy_artifact(Path(path), metrics_dir) for path in metric_summary_paths]
    pixel_triage_relative = (
        _copy_artifact(Path(pixel_triage_summary_path), metrics_dir)
        if pixel_triage_summary_path is not None
        else None
    )
    roi_relatives = [_copy_artifact(Path(path), rois_dir) for path in roi_crop_paths]

    source_images = [Path(path) for path in round_image_paths]
    canary_panel_id = None
    if canary_image_path is not None:
        source_images.append(Path(canary_image_path))
        canary_panel_id = _generic_panel_id(len(source_images))

    image_rows: List[Dict[str, Any]] = []
    for index, source in enumerate(source_images, start=1):
        panel_id = _generic_panel_id(index)
        destination = images_dir / f"{panel_id}.png"
        metadata = _copy_image_under_cap(source, destination, image_cap_px)
        image_rows.append(
            {
                "panel_id": panel_id,
                "image_path": str(destination.relative_to(output_root)),
                "max_side_px": max(metadata["packaged_size_px"]),
                "size_px": metadata["packaged_size_px"],
            }
        )

    metric_prompt_value = metric_relatives[0] if metric_relatives else None
    prompt_variables = {
        "N": len(image_rows),
        "pair_image_paths": [row["image_path"] for row in image_rows],
        "roi_crops": roi_relatives,
        "declarative_metric_summary": metric_prompt_value,
        "metric_summaries": metric_relatives,
        "pixel_triage_summary": pixel_triage_relative,
        "prior_findings": list(prior_findings) if prior_findings else "none",
        "geometry_mode": geometry_mode,
    }

    manifest: Dict[str, Any] = {
        "schema_version": AUDIT_PACKAGE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "round_id": round_id,
        "image_limits": {"max_side_px": image_cap_px, "max_images_per_call": 10},
        "prompt": {"template_path": prompt_relative, "variables": prompt_variables},
        "images": image_rows,
        "metric_summaries": metric_relatives,
        "pixel_triage_summary": pixel_triage_relative,
        "roi_crops": roi_relatives,
    }
    sidecar: Dict[str, Any] = {
        "schema_version": AUDIT_PACKAGE_SCHEMA_VERSION,
        "generated_at": manifest["generated_at"],
        "round_id": round_id,
        "canary_panel_id": canary_panel_id,
        "canary_tripwire_id": canary_tripwire_id,
        "canary_source_path": str(canary_image_path) if canary_image_path is not None else None,
    }

    manifest_path = output_root / "audit_manifest.json"
    sidecar_path = output_root / "harness_sidecar.json"
    _write_json(manifest_path, manifest)
    _write_json(sidecar_path, sidecar)

    return AuditPackageResult(
        output_dir=output_root,
        manifest_path=manifest_path,
        sidecar_path=sidecar_path,
        manifest=manifest,
        sidecar=sidecar,
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for audit package bundling.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--round-id", default="round_unset")
    parser.add_argument("--geometry-mode", default="injected")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--round-image", action="append", default=[], required=True)
    parser.add_argument("--metric-summary", action="append", default=[])
    parser.add_argument("--roi-crop", action="append", default=[])
    parser.add_argument("--pixel-triage-summary")
    parser.add_argument("--prior-finding", action="append", default=[])
    parser.add_argument("--canary-image")
    parser.add_argument("--canary-tripwire-id")
    parser.add_argument("--image-cap-px", type=int, default=MAX_AUDIT_IMAGE_SIDE_PX)
    return parser.parse_args()


def main() -> int:
    """Build an audit package from CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """

    args = _parse_args()
    result = build_audit_package(
        round_image_paths=args.round_image,
        output_dir=args.out,
        metric_summary_paths=args.metric_summary,
        prompt_path=args.prompt,
        round_id=args.round_id,
        geometry_mode=args.geometry_mode,
        roi_crop_paths=args.roi_crop,
        pixel_triage_summary_path=args.pixel_triage_summary,
        prior_findings=args.prior_finding,
        canary_image_path=args.canary_image,
        canary_tripwire_id=args.canary_tripwire_id,
        image_cap_px=args.image_cap_px,
    )

    print(f"Wrote audit package to {result.output_dir}")
    print(f"  manifest: {result.manifest_path}")
    print(f"  harness sidecar: {result.sidecar_path}")
    print(f"  panels: {len(result.manifest['images'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
