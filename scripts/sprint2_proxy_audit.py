"""Audit native marketplace proxy rank fidelity from persisted telemetry only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

RESEARCH_ROOT = Path.home() / ".claude/research/dagua/sprint2_improve"
DEFAULT_TELEMETRY_DIR = RESEARCH_ROOT / "measure/telemetry"
DEFAULT_OUTPUT = RESEARCH_ROOT / "measure/PROXY_RANK_AUDIT.md"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] or None
        Explicit arguments, or ``None`` to read process arguments.

    Returns
    -------
    argparse.Namespace
        Parsed telemetry and report paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, default=DEFAULT_TELEMETRY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def average_ranks(values: Sequence[float]) -> List[float]:
    """Return ascending average ranks with deterministic tie handling.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values to rank.

    Returns
    -------
    List[float]
        One-based average ranks in original order.
    """
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index, _value in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute a Pearson correlation, returning NaN for constant inputs.

    Parameters
    ----------
    first : Sequence[float]
        First variable.
    second : Sequence[float]
        Second variable of equal length.

    Returns
    -------
    float
        Pearson correlation or ``nan`` when undefined.
    """
    if len(first) != len(second) or len(first) < 2:
        return math.nan
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    centered = [(x - mean_first, y - mean_second) for x, y in zip(first, second)]
    numerator = sum(x * y for x, y in centered)
    denominator = math.sqrt(sum(x * x for x, _y in centered) * sum(y * y for _x, y in centered))
    return numerator / denominator if denominator else math.nan


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute Spearman rank correlation with average ranks.

    Parameters
    ----------
    first : Sequence[float]
        First score sequence.
    second : Sequence[float]
        Second score sequence.

    Returns
    -------
    float
        Spearman rho or ``nan`` when undefined.
    """
    return pearson(average_ranks(first), average_ranks(second))


def kendall_tau_b(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute Kendall tau-b, including ties in either score sequence.

    Parameters
    ----------
    first : Sequence[float]
        First score sequence.
    second : Sequence[float]
        Second score sequence.

    Returns
    -------
    float
        Kendall tau-b or ``nan`` when undefined.
    """
    concordant = discordant = ties_first = ties_second = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            dx = first[left] - first[right]
            dy = second[left] - second[right]
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                ties_first += 1
            elif dy == 0.0:
                ties_second += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_first) * (concordant + discordant + ties_second)
    )
    return (concordant - discordant) / denominator if denominator else math.nan


def graph_name_from_path(path: Path) -> str:
    """Recover a graph label from a per-graph telemetry filename.

    Parameters
    ----------
    path : Path
        JSONL path, conventionally named ``corpus__stem.jsonl``.

    Returns
    -------
    str
        Canonical graph label, or the filename stem for legacy logs.
    """
    return path.stem.replace("__", "/", 1)


def load_events(directory: Path) -> List[Dict[str, Any]]:
    """Load native marketplace events from JSONL files.

    Parameters
    ----------
    directory : Path
        Directory recursively containing telemetry JSONL.

    Returns
    -------
    List[Dict[str, Any]]
        Marketplace events annotated with graph and source path.
    """
    events: List[Dict[str, Any]] = []
    if not directory.is_dir():
        return events
    for path in sorted(directory.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("event") != "native_candidate_marketplace":
                continue
            payload["_graph"] = payload.get("graph") or graph_name_from_path(path)
            payload["_source"] = f"{path}:{line_number}"
            events.append(payload)
    return events


def paired_arms(event: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return finite proxy/honest arm pairs from one contest.

    Parameters
    ----------
    event : Mapping[str, Any]
        Native marketplace event.

    Returns
    -------
    List[Dict[str, Any]]
        Arms containing both finite ``raw_score`` and ``full_score`` values.
    """
    pairs: List[Dict[str, Any]] = []
    for arm in event.get("arms", []):
        proxy = arm.get("raw_score")
        honest = arm.get("full_score")
        if proxy is None or honest is None:
            continue
        if math.isfinite(float(proxy)) and math.isfinite(float(honest)):
            pairs.append(dict(arm))
    return pairs


def top_set(arms: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    """Return the top-decile arm-name set for a score key.

    Parameters
    ----------
    arms : Sequence[Mapping[str, Any]]
        Paired arms.
    key : str
        Score key to rank.

    Returns
    -------
    set[str]
        At least one arm, using ceiling decile size.
    """
    count = max(1, math.ceil(len(arms) * 0.10))
    ordered = sorted(arms, key=lambda arm: (-float(arm[key]), str(arm["name"])))
    return {str(arm["name"]) for arm in ordered[:count]}


def summarize(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize overall and per-route proxy fidelity.

    Route/structural-gate is the only persisted graph-class signal in legacy telemetry;
    per-graph files retain graph identity for drill-down.

    Parameters
    ----------
    events : Sequence[Mapping[str, Any]]
        Marketplace events.

    Returns
    -------
    List[Dict[str, Any]]
        Overall and class summaries.
    """
    groups: Dict[str, List[Mapping[str, Any]]] = {"overall": list(events)}
    for event in events:
        arms = event.get("arms", [])
        gate = str(arms[0].get("structural_gate", "unknown")) if arms else "unknown"
        label = f"{event.get('route', 'unknown')}:{gate}"
        groups.setdefault(label, []).append(event)
    summaries: List[Dict[str, Any]] = []
    for label, group_events in sorted(groups.items()):
        proxy: List[float] = []
        honest: List[float] = []
        agreements: List[float] = []
        drops: List[str] = []
        unobservable_rejected = 0
        contests_with_pairs = 0
        for event in group_events:
            arms = paired_arms(event)
            unobservable_rejected += sum(
                arm.get("full_score") is None for arm in event.get("arms", [])
            )
            if len(arms) < 2:
                continue
            contests_with_pairs += 1
            proxy.extend(float(arm["raw_score"]) for arm in arms)
            honest.extend(float(arm["full_score"]) for arm in arms)
            proxy_top = top_set(arms, "raw_score")
            honest_top = top_set(arms, "full_score")
            agreements.append(len(proxy_top & honest_top) / len(proxy_top | honest_top))
            top_k = int(event.get("top_k", 8))
            proxy_finalists = {
                str(arm["name"])
                for arm in sorted(
                    arms, key=lambda arm: (-float(arm["raw_score"]), str(arm["name"]))
                )[:top_k]
            }
            honest_winner = max(arms, key=lambda arm: (float(arm["full_score"]), str(arm["name"])))
            if str(honest_winner["name"]) not in proxy_finalists:
                drops.append(f"{event.get('_graph', 'unknown')}:{honest_winner['name']}")
        summaries.append(
            {
                "class": label,
                "events": len(group_events),
                "paired_contests": contests_with_pairs,
                "pairs": len(proxy),
                "spearman": spearman(proxy, honest),
                "kendall": kendall_tau_b(proxy, honest),
                "top_decile": sum(agreements) / len(agreements) if agreements else math.nan,
                "observed_drops": drops,
                "unobservable_rejected": unobservable_rejected,
            }
        )
    return summaries


def format_number(value: float) -> str:
    """Format a correlation value for Markdown.

    Parameters
    ----------
    value : float
        Numeric statistic.

    Returns
    -------
    str
        Three decimals, or ``n/a`` for undefined values.
    """
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def render_report(
    events: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]
) -> str:
    """Render the proxy audit as Markdown.

    Parameters
    ----------
    events : Sequence[Mapping[str, Any]]
        Loaded telemetry events.
    summaries : Sequence[Mapping[str, Any]]
        Computed class summaries.

    Returns
    -------
    str
        Complete report contents.
    """
    lines = [
        "# Sprint-2 proxy rank-fidelity audit",
        "",
        f"Persisted marketplace contests found: **{len(events)}**.",
        "",
        "Legacy events contain graph route/structural gate but not a classifier-family field, so "
        "that persisted structural signal is the class grouping below. Correlations use only arms "
        "with both proxy and honest scores.",
        "",
        "| class | events | paired contests | paired arms | Spearman | Kendall tau-b | "
        "top-decile Jaccard | observed winner drops | rejected without honest score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['class']} | {summary['events']} | {summary['paired_contests']} | "
            f"{summary['pairs']} | {format_number(float(summary['spearman']))} | "
            f"{format_number(float(summary['kendall']))} | "
            f"{format_number(float(summary['top_decile']))} | "
            f"{len(summary['observed_drops'])} | {summary['unobservable_rejected']} |"
        )
    overall = next((row for row in summaries if row["class"] == "overall"), None)
    lines.extend(["", "## Finalist-count recommendation", ""])
    if overall is None or int(overall["pairs"]) == 0:
        lines.append(
            "No persisted dev63 proxy/full pairs were available. FULL_REFEREE_TOP_K=8 cannot be "
            "declared safe or raised from this evidence; W1-C should preserve 8 and keep mandatory "
            "incumbent/family representatives until graph-identified telemetry is collected."
        )
    elif overall["observed_drops"]:
        lines.append(
            "The reconstructed proxy cut dropped an honest winner in observed fully scored data. "
            "W1-C should flex m above 8 for the affected classes listed below."
        )
    else:
        lines.append(
            "No honest winner was dropped within the fully scored paired arms. Preserve m=8. "
            "This is not proof for rejected arms because their honest scores were never persisted."
        )
    drops = [drop for row in summaries for drop in row["observed_drops"]]
    if drops:
        lines.extend(["", "Observed drops: " + ", ".join(sorted(set(drops)))])
    lines.extend(
        [
            "",
            "## Audit limitation",
            "",
            "Current telemetry stores `full_score=null` for proxy-rejected arms. Consequently, an "
            "honest winner among those arms is unobservable without re-scoring layouts. This audit "
            "mines persisted telemetry only and does not convert 'no observed drop' into a "
            "safety claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Load telemetry, calculate rank fidelity, and write the report.

    Parameters
    ----------
    argv : Sequence[str] or None
        Explicit arguments, or ``None`` for process arguments.

    Returns
    -------
    int
        Always zero after a valid report, including a transparent no-data report.
    """
    args = parse_args(argv)
    events = load_events(args.telemetry_dir)
    summaries = summarize(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(events, summaries))
    print(f"wrote {args.output} ({len(events)} contests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
