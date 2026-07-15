# ruff: noqa: E402
"""Score and store VLM candidates against the calibration probe (F14).

FINAL_DESIGN.md section 6, step 5 + correction F14: a repo script cannot
web-search. Model discovery (searching for the current best vision models)
is the ORCHESTRATOR's job, run as a runbook step outside this script. This
module only SCORES orchestrator-supplied candidate responses against the
known-answer calibration probe (``render_calibration_probe.py``'s
``defect_manifest.json``) and STORES the result in ``model_selection.json``.
It never calls a model, never fetches a URL, and never performs any network
I/O -- "evidence_urls" are accepted as opaque strings the orchestrator
already gathered, and are only shape-validated (well-formed http(s) URLs),
never dereferenced.

Rejection rule (FINAL_DESIGN.md section 6, step 5): a candidate that misses
the injected stem (``invisible_edge_stem``) or truncation
(``truncated_label``) defect, or that flags either false-positive control
panel (``antialiased_residual`` / ``true_noop_match``), is REJECTED for
auditing duty regardless of its aggregate score.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Defects that MUST be caught; missing either is an automatic rejection.
MANDATORY_DEFECT_PANEL_IDS: frozenset[str] = frozenset({"invisible_edge_stem", "truncated_label"})
DEFAULT_MODEL_SELECTION_PATH = (
    ".project-context/research/sprint_visual_parity_v2/model_selection.json"
)


@dataclass(frozen=True)
class CandidateScore:
    """Scoring result for one candidate model against the calibration probe.

    Parameters
    ----------
    model
        Candidate model identifier.
    recall
        Fraction of the 6 known defects the candidate reported a finding for.
    defects_found
        Defect panel ids the candidate reported a finding for.
    defects_missed
        Defect panel ids the candidate reported no finding for.
    false_positive_panels
        Control panel ids the candidate incorrectly flagged.
    rejected
        Whether the candidate is rejected for auditing duty.
    rejection_reason
        Human-readable rejection reason, if rejected.
    structured_output_compliant
        Whether the response matches the audit prompt's required JSON shape.
    score
        Aggregate score: ``recall - 0.5 * false_positive_count`` (0 when
        rejected).
    cost_latency_notes
        Pass-through cost/latency notes supplied by the orchestrator.
    """

    model: str
    recall: float
    defects_found: List[str] = field(default_factory=list)
    defects_missed: List[str] = field(default_factory=list)
    false_positive_panels: List[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str = ""
    structured_output_compliant: bool = True
    score: float = 0.0
    cost_latency_notes: str = ""


def _load_defect_manifest(probe_manifest_path: str | Path) -> Dict[str, Any]:
    """Load and validate a calibration probe defect manifest.

    Parameters
    ----------
    probe_manifest_path
        Path to ``defect_manifest.json`` (from ``render_calibration_probe``).

    Returns
    -------
    dict[str, Any]
        Parsed manifest.

    Raises
    ------
    ValueError
        If the manifest is missing required panels.
    """

    manifest = json.loads(Path(probe_manifest_path).read_text(encoding="utf-8"))
    panel_ids = {panel["panel_id"] for panel in manifest.get("panels", [])}
    missing = MANDATORY_DEFECT_PANEL_IDS - panel_ids
    if missing:
        raise ValueError(f"probe manifest missing mandatory defect panels: {sorted(missing)}")
    return manifest


def _panel_ids_by_kind(manifest: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    """Split a defect manifest into defect and control panel id lists.

    Parameters
    ----------
    manifest
        Parsed ``defect_manifest.json``.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(defect_panel_ids, control_panel_ids)``.
    """

    defects = [p["panel_id"] for p in manifest.get("panels", []) if not p.get("is_control")]
    controls = [p["panel_id"] for p in manifest.get("panels", []) if p.get("is_control")]
    return defects, controls


def _is_structured_output_compliant(response: Mapping[str, Any]) -> bool:
    """Check a candidate response against the audit prompt's required JSON shape.

    Parameters
    ----------
    response
        Candidate's structured JSON response.

    Returns
    -------
    bool
        ``True`` when ``verdict`` and a list ``findings`` (with required
        per-finding fields) are present.
    """

    if "verdict" not in response or "findings" not in response:
        return False
    findings = response["findings"]
    if not isinstance(findings, list):
        return False
    required_fields = {"pair", "category", "description", "finding_class", "severity"}
    for finding in findings:
        if not isinstance(finding, Mapping):
            return False
        if not required_fields.issubset(finding.keys()):
            return False
    return True


def _referenced_panel_ids(response: Mapping[str, Any]) -> set[str]:
    """Return the set of panel ids a candidate's findings reference.

    Parameters
    ----------
    response
        Candidate's structured JSON response.

    Returns
    -------
    set[str]
        Panel ids appearing in the ``pair`` field of any finding.
    """

    findings = response.get("findings", [])
    referenced: set[str] = set()
    for finding in findings:
        if isinstance(finding, Mapping) and finding.get("pair"):
            referenced.add(str(finding["pair"]))
    return referenced


def score_candidate(
    model: str,
    response: Mapping[str, Any],
    defect_panel_ids: Sequence[str],
    control_panel_ids: Sequence[str],
    cost_latency_notes: str = "",
) -> CandidateScore:
    """Score one candidate model's response against the calibration probe.

    Parameters
    ----------
    model
        Candidate model identifier.
    response
        Candidate's structured JSON response (audit_v2.md's OUTPUT shape).
    defect_panel_ids
        The 6 known-defect panel ids.
    control_panel_ids
        The 2 false-positive control panel ids.
    cost_latency_notes
        Orchestrator-supplied cost/latency notes.

    Returns
    -------
    CandidateScore
        Scoring result, including rejection status.
    """

    compliant = _is_structured_output_compliant(response)
    referenced = _referenced_panel_ids(response) if compliant else set()

    defects_found = [pid for pid in defect_panel_ids if pid in referenced]
    defects_missed = [pid for pid in defect_panel_ids if pid not in referenced]
    false_positive_panels = [pid for pid in control_panel_ids if pid in referenced]
    recall = len(defects_found) / len(defect_panel_ids) if defect_panel_ids else 0.0

    rejected = False
    rejection_reason = ""
    if not compliant:
        rejected = True
        rejection_reason = "response does not match the required structured-output shape"
    else:
        missed_mandatory = MANDATORY_DEFECT_PANEL_IDS.intersection(defects_missed)
        if missed_mandatory:
            rejected = True
            rejection_reason = f"missed mandatory defect(s): {sorted(missed_mandatory)}"
        elif false_positive_panels:
            rejected = True
            rejection_reason = f"flagged control panel(s): {sorted(false_positive_panels)}"

    score = 0.0 if rejected else max(0.0, recall - 0.5 * len(false_positive_panels))

    return CandidateScore(
        model=model,
        recall=recall,
        defects_found=defects_found,
        defects_missed=defects_missed,
        false_positive_panels=false_positive_panels,
        rejected=rejected,
        rejection_reason=rejection_reason,
        structured_output_compliant=compliant,
        score=score,
        cost_latency_notes=cost_latency_notes,
    )


def _validate_evidence_urls(evidence_urls: Sequence[str]) -> List[str]:
    """Shape-validate orchestrator-supplied evidence URLs (never fetched).

    Parameters
    ----------
    evidence_urls
        Candidate evidence URLs gathered by the orchestrator's web search.

    Returns
    -------
    list[str]
        The subset of well-formed http(s) URLs, in input order.

    Raises
    ------
    ValueError
        If any entry is not a well-formed http(s) URL.
    """

    validated: List[str] = []
    for url in evidence_urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"evidence url is not a well-formed http(s) URL: {url!r}")
        validated.append(url)
    return validated


def select_vision_models(
    responses_path: str | Path,
    probe_manifest_path: str | Path,
) -> Dict[str, Any]:
    """Score all candidates in a responses file and select auditor roles.

    Parameters
    ----------
    responses_path
        Path to the orchestrator-supplied candidate responses JSON (see
        module docstring for the expected schema).
    probe_manifest_path
        Path to the calibration probe's ``defect_manifest.json``.

    Returns
    -------
    dict[str, Any]
        A ``model_selection.json``-shaped payload:
        ``{primary_auditor, ceiling_auditor, fallback_auditor, scores,
        evidence_urls, probed_at, image_limits}``.
    """

    manifest = _load_defect_manifest(probe_manifest_path)
    defect_panel_ids, control_panel_ids = _panel_ids_by_kind(manifest)

    payload = json.loads(Path(responses_path).read_text(encoding="utf-8"))
    evidence_urls = _validate_evidence_urls(payload.get("evidence_urls", []))
    image_limits = payload.get("image_limits", {"max_side_px": 2000, "max_images_per_call": 10})

    scores: List[CandidateScore] = []
    for candidate in payload.get("candidates", []):
        scores.append(
            score_candidate(
                model=candidate["model"],
                response=candidate.get("response", {}),
                defect_panel_ids=defect_panel_ids,
                control_panel_ids=control_panel_ids,
                cost_latency_notes=str(candidate.get("cost_latency_notes", "")),
            )
        )

    eligible = sorted((s for s in scores if not s.rejected), key=lambda s: s.score, reverse=True)
    primary_auditor: Optional[str] = eligible[0].model if eligible else None
    ceiling_auditor: Optional[str] = (
        next((s.model for s in eligible if not s.false_positive_panels), primary_auditor)
        if eligible
        else None
    )
    fallback_auditor: Optional[str] = eligible[1].model if len(eligible) > 1 else None

    return {
        "primary_auditor": primary_auditor,
        "ceiling_auditor": ceiling_auditor,
        "fallback_auditor": fallback_auditor,
        "scores": [_score_to_dict(s) for s in scores],
        "evidence_urls": evidence_urls,
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "image_limits": image_limits,
    }


def _score_to_dict(score: CandidateScore) -> Dict[str, Any]:
    """Convert a ``CandidateScore`` to a JSON-serializable dict.

    Parameters
    ----------
    score
        Candidate score record.

    Returns
    -------
    dict[str, Any]
        JSON-serializable representation.
    """

    return {
        "model": score.model,
        "recall": score.recall,
        "defects_found": score.defects_found,
        "defects_missed": score.defects_missed,
        "false_positive_panels": score.false_positive_panels,
        "rejected": score.rejected,
        "rejection_reason": score.rejection_reason,
        "structured_output_compliant": score.structured_output_compliant,
        "score": score.score,
        "cost_latency_notes": score.cost_latency_notes,
    }


def main() -> int:
    """Parse CLI arguments and run the score-and-store command.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        dest="responses_path",
        required=True,
        help="Path to the orchestrator-supplied candidate responses JSON.",
    )
    parser.add_argument(
        "--probe-manifest",
        default="eval_output/visual_parity_v2/probe/defect_manifest.json",
        help="Path to the calibration probe's defect_manifest.json.",
    )
    parser.add_argument("--out", default=DEFAULT_MODEL_SELECTION_PATH)
    args = parser.parse_args()

    result = select_vision_models(args.responses_path, args.probe_manifest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  primary_auditor: {result['primary_auditor']}")
    print(f"  ceiling_auditor: {result['ceiling_auditor']}")
    print(f"  fallback_auditor: {result['fallback_auditor']}")
    for row in result["scores"]:
        status = "REJECTED" if row["rejected"] else "eligible"
        print(f"  {row['model']}: score={row['score']:.2f} recall={row['recall']:.2f} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
