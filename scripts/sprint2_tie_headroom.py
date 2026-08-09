"""Measure sprint-2 tied-row facet headroom and verify training cousins.

The script is deliberately read-only with respect to evaluation inputs. It scores the
already-regenerated native position tensors and writes ``TIED_HEADROOM.md`` plus
``COUSINS.md`` to the requested measurement directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.glados_holdout_run import (  # noqa: E402
    DEFAULT_MAX_EDGES,
    DEFAULT_MAX_NODES,
    build_test_graph_map,
    load_phase,
)
from scripts.native_sprint_score import score_position, scoring_signature  # noqa: E402

RESEARCH_ROOT = Path.home() / ".claude/research/dagua/sprint2_improve"
MAIN_EVAL_ROOT = Path.home() / "projects/dagua/eval_output"
DEFAULT_TIED_ROWS = RESEARCH_ROOT / "DEV63_TIED_ROWS.txt"
DEFAULT_DEV_DIR = MAIN_EVAL_ROOT / "sprint2_dev"
DEFAULT_TRAIN_DIR = MAIN_EVAL_ROOT / "sprint2_train"
DEFAULT_CORPUS_DIR = MAIN_EVAL_ROOT / "stdcorpora"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "measure"
MOVABLE_FACETS = ("C4", "C5", "C6", "C9", "C10")
WINNABLE_HEADROOM = 0.5
STRICT_TIE_BAND = 0.5

WAVE_TARGETS = {
    "C4": "W2-4 winner polish",
    "C5": "W2-3 winsorized sprawl repair",
    "C6": "W2-4 winner polish",
    "C9": "W2-4 winner polish",
    "C10": "W2-4 winner polish",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] or None
        Explicit arguments, or ``None`` to read ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed paths for the read-only inputs and report directory.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tied-rows", type=Path, default=DEFAULT_TIED_ROWS)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def parse_tied_rows(path: Path) -> List[Dict[str, Any]]:
    """Parse the canonical tied-row text table.

    Parameters
    ----------
    path : Path
        ``DEV63_TIED_ROWS.txt`` input path.

    Returns
    -------
    List[Dict[str, Any]]
        Parsed rows in file order.

    Raises
    ------
    ValueError
        If a non-empty row does not match the canonical format.
    """
    rows: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        pattern = re.compile(
            r"^(?P<graph>\S+)\s+native=\s*(?P<native>[+-]?\d+(?:\.\d+)?)\s+"
            r"field_best=\s*(?P<field>[+-]?\d+(?:\.\d+)?)\s+"
            r"\((?P<engine>[^)]+)\)\s+delta=(?P<delta>[+-]?\d+(?:\.\d+)?)\s+"
            r"directed=(?P<directed>True|False)$"
        )
        match = pattern.match(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: malformed tied row")
        values = match.groupdict()
        try:
            rows.append(
                {
                    "graph": values["graph"],
                    "native": float(values["native"]),
                    "field_best": float(values["field"]),
                    "field_engine": values["engine"],
                    "delta": float(values["delta"]),
                    "directed": values["directed"] == "True",
                }
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: malformed tied row") from exc
    return rows


def position_path(dev_dir: Path, graph_name: str) -> Path:
    """Return the regenerated native position path for a graph.

    Parameters
    ----------
    dev_dir : Path
        Sprint-2 development data directory.
    graph_name : str
        Canonical ``corpus/stem`` graph name.

    Returns
    -------
    Path
        Expected deterministic native tensor path.
    """
    flat_name = graph_name.replace("/", "_")
    return dev_dir / "run/positions" / f"{flat_name}__dagua__deterministic.pt"


def facet_headroom(facets: Mapping[str, Mapping[str, Any]]) -> Tuple[Dict[str, float], float]:
    """Compute exact linear-headline uplift available in movable facets.

    The V3 headline is a renormalized weighted linear composite. Each returned
    contribution is therefore the score-point uplift obtained by moving only that
    applicable facet to its arithmetic cap of 1.0 while holding weights fixed.

    Parameters
    ----------
    facets : Mapping[str, Mapping[str, Any]]
        Serialized V3 facet records from ``score_position(ruler="v3")``.

    Returns
    -------
    Tuple[Dict[str, float], float]
        Per-facet headline point uplift and their aggregate.
    """
    total_weight = sum(
        float(facet["effective_weight"])
        for facet in facets.values()
        if facet.get("applicable") and facet.get("score") is not None
    )
    contributions: Dict[str, float] = {}
    for code in MOVABLE_FACETS:
        facet = facets.get(code)
        if facet is None or not facet.get("applicable") or facet.get("score") is None:
            contributions[code] = 0.0
            continue
        score = float(facet["score"])
        weight = float(facet["effective_weight"])
        contributions[code] = 100.0 * max(0.0, 1.0 - score) * weight / total_weight
    return contributions, sum(contributions.values())


def classify_headroom(native_score: float, headroom: float) -> str:
    """Classify a tied row using the packet's arithmetic-headroom rule.

    Parameters
    ----------
    native_score : float
        Native V3 headline score on the 0--100 scale.
    headroom : float
        Aggregate movable-facet headline uplift in score points.

    Returns
    -------
    str
        ``WINNABLE`` at or above 0.5 points, otherwise ``CAPPED``. The native
        score is accepted to keep the threshold decision auditable in callers.
    """
    _ = native_score
    return "WINNABLE" if headroom >= WINNABLE_HEADROOM else "CAPPED"


def choose_wave_target(row: Mapping[str, Any], contributions: Mapping[str, float]) -> str:
    """Choose the planned wave item matching the row's observed mechanism.

    Parameters
    ----------
    row : Mapping[str, Any]
        Tied-row record including graph and field-best engine.
    contributions : Mapping[str, float]
        Per-facet headline point headroom.

    Returns
    -------
    str
        Planned wave item label.
    """
    graph_name = str(row["graph"])
    engine = str(row["field_engine"])
    if graph_name in {"suitesparse/arc130", "suitesparse/ash85"}:
        return "W2-3 winsorized sprawl repair"
    if any(token in engine for token in ("stress", "sgd2", "davidson")):
        return "W2-2 stress-family arms"
    if any(token in engine for token in ("smartgd", "fcose")):
        return "W2-1 deterministic multi-seed"
    leading_facet = max(MOVABLE_FACETS, key=lambda code: (contributions.get(code, 0.0), code))
    return WAVE_TARGETS[leading_facet]


def load_dev_graphs(dev_dir: Path, names: Set[str]) -> Dict[str, Any]:
    """Load only named development graphs from their isolated copied corpus.

    Parameters
    ----------
    dev_dir : Path
        Sprint-2 development directory containing ``graphs/``.
    names : Set[str]
        Canonical graph names to load.

    Returns
    -------
    Dict[str, Any]
        Test graphs keyed by canonical name.
    """
    manifest = json.loads((dev_dir / "field_bests.json").read_text())
    relpaths = {manifest["graphs"][name]["relpath"] for name in names}
    entries, load_rows, _stats = load_phase(
        dev_dir / "graphs", relpaths, DEFAULT_MAX_NODES, DEFAULT_MAX_EDGES
    )
    failures = [row for row in load_rows if row.get("status") not in {None, "OK"}]
    if failures or {entry.name for entry in entries} != names:
        raise RuntimeError(f"development graph load mismatch: {failures}")
    return build_test_graph_map(entries)


def measure_headroom(rows: List[Dict[str, Any]], dev_dir: Path) -> List[Dict[str, Any]]:
    """Score tied native positions and attach facet headroom measurements.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Parsed canonical tied rows.
    dev_dir : Path
        Sprint-2 development directory.

    Returns
    -------
    List[Dict[str, Any]]
        Rows enriched with freshly scored facets and verdicts.
    """
    graph_map = load_dev_graphs(dev_dir, {str(row["graph"]) for row in rows})
    signature = scoring_signature()
    measured: List[Dict[str, Any]] = []
    for row in rows:
        graph_name = str(row["graph"])
        path = position_path(dev_dir, graph_name)
        if not path.is_file():
            raise FileNotFoundError(f"missing regenerated native position: {path}")
        score = score_position(graph_map[graph_name], str(path), "dagua", signature, ruler="v3")
        fresh_native = float(score["v3_tiered"])
        if not math.isclose(fresh_native, float(row["native"]), abs_tol=0.011):
            raise RuntimeError(
                f"{graph_name}: tied table says {row['native']}, fresh score is {fresh_native}"
            )
        contributions, aggregate = facet_headroom(score["v3_facets"])
        measured.append(
            {
                **row,
                "native": fresh_native,
                "needed_margin": max(0.0, STRICT_TIE_BAND - float(row["delta"])),
                "headroom": contributions,
                "aggregate_headroom": aggregate,
                "verdict": classify_headroom(fresh_native, aggregate),
                "wave_target": choose_wave_target(row, contributions),
            }
        )
    return measured


def render_headroom(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render measured tied rows as Markdown.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Enriched measurement rows.

    Returns
    -------
    str
        Complete ``TIED_HEADROOM.md`` contents.
    """
    lines = [
        "# Sprint-2 tied-row movable headroom",
        "",
        (
            "Headroom is exact uplift in the V3 linear headline if each applicable "
            "finisher-movable facet alone reached 1.0, with all weights fixed. Verdict: "
            "WINNABLE at >=0.500 aggregate points; otherwise CAPPED."
        ),
        "",
        "| row | native | field best | delta | needed | C4 | C5 | C6 | C9 | C10 | "
        "total | verdict | target |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        headroom = row["headroom"]
        field = f"{float(row['field_best']):.3f} ({row['field_engine']})"
        lines.append(
            f"| {row['graph']} | {float(row['native']):.3f} | {field} | "
            f"{float(row['delta']):+.3f} | {float(row['needed_margin']):.3f} | "
            + " | ".join(f"{float(headroom[code]):.3f}" for code in MOVABLE_FACETS)
            + f" | {float(row['aggregate_headroom']):.3f} | {row['verdict']} | "
            f"{row['wave_target']} |"
        )
    winnable = sum(row["verdict"] == "WINNABLE" for row in rows)
    lines.extend(
        [
            "",
            f"Summary: **{winnable}/{len(rows)} WINNABLE**, {len(rows) - winnable} CAPPED.",
            "",
        ]
    )
    return "\n".join(lines)


def canonical_name(relpath: str) -> str:
    """Return a path-independent canonical corpus/stem identity.

    Parameters
    ----------
    relpath : str
        Manifest relative path.

    Returns
    -------
    str
        Lowercase ``corpus/stem`` identity.
    """
    path = Path(relpath)
    return f"{path.parts[0].lower()}/{path.stem.lower()}"


def verify_cousins(
    train_manifest: Path, corpus_dir: Path
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Verify training entries are disjoint from both allocated manifests.

    Parameters
    ----------
    train_manifest : Path
        Sprint-2 training ``MANIFEST.json``.
    corpus_dir : Path
        Directory holding allocation manifests. Listed graph files are never opened.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[str]]
        Training file records and detected canonical-name or published-hash collisions.
    """
    training = json.loads(train_manifest.read_text())
    subset = json.loads((corpus_dir / "SUBSET.json").read_text())
    sealed = json.loads((corpus_dir / "SEALED_REMAINDER.json").read_text())
    allocated = list(subset["selected"]) + list(sealed["sealed"])
    if len(allocated) != 140:
        raise RuntimeError(f"expected 140 allocated rows, found {len(allocated)}")
    allocated_names = {canonical_name(str(entry["relpath"])) for entry in allocated}
    allocated_hashes = {
        str(entry[key]).lower()
        for entry in allocated
        for key in ("sha256", "graph_file_sha256")
        if entry.get(key)
    }
    collisions: List[str] = []
    files: List[Dict[str, Any]] = list(training["files"])
    for entry in files:
        name = canonical_name(str(entry["relpath"]))
        if name in allocated_names:
            collisions.append(f"name:{name}")
        digest = str(entry.get("sha256", "")).lower()
        if digest and digest in allocated_hashes:
            collisions.append(f"sha256:{digest}")
    return files, sorted(set(collisions))


def cousin_family(entry: Mapping[str, Any]) -> str:
    """Group a training entry into the packet's requested cousin families.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Training manifest file record.

    Returns
    -------
    str
        ``power-grid cousins``, ``planar batch``, or ``other``.
    """
    relpath = str(entry["relpath"])
    stem = Path(relpath).stem.lower()
    if relpath.startswith("suitesparse/") and stem.startswith(("can_", "dwt_", "bcspwr")):
        return "power-grid cousins"
    if relpath.startswith(("rome/", "north/")):
        return "planar batch"
    return "other"


def render_cousins(files: Sequence[Mapping[str, Any]], collisions: Sequence[str]) -> str:
    """Render the verified training cousin inventory as Markdown.

    Parameters
    ----------
    files : Sequence[Mapping[str, Any]]
        Training manifest records.
    collisions : Sequence[str]
        Detected allocated-set collisions.

    Returns
    -------
    str
        Complete ``COUSINS.md`` contents.
    """
    lines = [
        "# Sprint-2 training cousins",
        "",
        f"Allocated-set check: **{'CLEAN' if not collisions else 'COLLISION'}** over "
        f"{len(files)} cousins and 140 allocations.",
        "",
        "The allocation manifests publish canonical names but no content SHA-256 values; "
        "name disjointness was checked across all 140 rows, and hash disjointness was checked "
        "for every allocation entry that published a content hash. No allocated graph file "
        "was opened.",
        "",
        "Binding rule: thresholds are **FITTED on cousins**, **CONFIRMED on dev63**, and "
        "never fitted on dev63.",
    ]
    if collisions:
        lines.extend(["", "Collisions: " + ", ".join(collisions)])
    for family in ("power-grid cousins", "planar batch", "other"):
        members = sorted(
            (entry for entry in files if cousin_family(entry) == family),
            key=lambda entry: str(entry["relpath"]),
        )
        lines.extend(["", f"## {family} ({len(members)})", ""])
        lines.extend(f"- `{entry['relpath']}` — sha256 `{entry['sha256']}`" for entry in members)
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run both W1-D headroom and cousin measurements.

    Parameters
    ----------
    argv : Sequence[str] or None
        Explicit CLI arguments, or ``None`` for process arguments.

    Returns
    -------
    int
        Zero when both artifacts were generated and cousin disjointness is clean.
    """
    args = parse_args(argv)
    rows = parse_tied_rows(args.tied_rows)
    if len(rows) != 14:
        raise RuntimeError(f"expected 14 tied rows, found {len(rows)}")
    measured = measure_headroom(rows, args.dev_dir)
    files, collisions = verify_cousins(args.train_dir / "MANIFEST.json", args.corpus_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "TIED_HEADROOM.md").write_text(render_headroom(measured))
    (args.output_dir / "COUSINS.md").write_text(render_cousins(files, collisions))
    print(f"wrote {args.output_dir / 'TIED_HEADROOM.md'} ({len(measured)} rows)")
    print(f"wrote {args.output_dir / 'COUSINS.md'} ({len(files)} cousins)")
    return 1 if collisions else 0


if __name__ == "__main__":
    raise SystemExit(main())
