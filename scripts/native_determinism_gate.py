#!/usr/bin/env python3
"""Run idle and synthetic-load determinism checks for native dagua layout."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch

from dagua.eval.benchmark import _declares_hierarchy
from dagua.eval.competitors.dagua_competitor import DaguaCompetitor
from dagua.eval.graphs import TestGraph, get_test_graphs, is_semantically_directed
from dagua.metrics import composite_auto, full

DEFAULT_GRAPHS = ("r8_nested_scale_1k_budget", "rgg_500", "residual_block")
DEFAULT_SCORE_SPREAD = 0.15
DEFAULT_TIMEOUT_MARGIN_S = 180.0


class GateRunTimeoutError(RuntimeError):
    """Raised when one determinism-gate layout run exceeds its watchdog."""


class _WorkerLayoutTimeoutError(TimeoutError):
    """Gate-local benchmark worker alarm sentinel."""


@dataclass(frozen=True)
class GateRun:
    """One determinism-gate run result.

    Parameters
    ----------
    graph : str
        Test graph name.
    mode : str
        Load mode, either ``"idle"`` or ``"load"``.
    index : int
        Zero-based run index within the mode.
    verdict : str
        Stable admission verdict derived from telemetry and result status.
    score : float
        Native full-ruler composite score.
    runtime_s : float
        Wall-clock runtime reported by the competitor.
    process_runtime_s : float
        Process CPU seconds spent in the child around layout and scoring.
    cpu_wall_ratio : float
        Ratio of process CPU seconds to wall-clock seconds.
    torch_num_threads : int
        Torch intra-op thread count in the child process.
    device : str
        Dagua competitor device used by the child process.
    plan_shape : str
        Compact W5 plan/winner summary.
    marketplace_arms : list[dict[str, Any]]
        Flattened marketplace arm status records from native telemetry.
    telemetry : list[dict[str, Any]]
        Parsed native W5 and arm-skip telemetry records.
    """

    graph: str
    mode: str
    index: int
    verdict: str
    score: float
    runtime_s: float
    process_runtime_s: float
    cpu_wall_ratio: float
    torch_num_threads: int
    device: str
    plan_shape: str
    marketplace_arms: list[dict[str, Any]]
    telemetry: list[dict[str, Any]]


@dataclass(frozen=True)
class ParityReplayRow:
    """One parity-replay comparison row.

    Parameters
    ----------
    graph : str
        Benchmark graph name.
    baseline_signature : str
        Harvested idle-main admission and W5 plan signature.
    replay_signature : str
        Current ledger replay admission and W5 plan signature.
    matches : bool
        Whether the two signatures match exactly.
    expected_diff : bool
        Whether this row is in the documented expected-diff list.
    """

    graph: str
    baseline_signature: str
    replay_signature: str
    matches: bool
    expected_diff: bool


def _timeout_signal_handler(signum: int, frame: object) -> None:
    """Raise the same-named timeout exception as the benchmark worker.

    Parameters
    ----------
    signum : int
        Delivered signal number.
    frame : object
        Current interpreter frame.

    Returns
    -------
    None
        The function always raises.
    """
    del signum, frame
    raise _WorkerLayoutTimeoutError("worker layout timeout exceeded")


def _enable_worker_timeout(timeout_seconds: float) -> Optional[Any]:
    """Install the benchmark-faithful wall-clock alarm in this child.

    Parameters
    ----------
    timeout_seconds : float
        Maximum wall seconds before ``SIGALRM`` raises.

    Returns
    -------
    object or None
        Previous signal handler, or ``None`` when alarms are unavailable.
    """
    if timeout_seconds <= 0.0 or not hasattr(signal, "SIGALRM"):
        return None
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_signal_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    return previous_handler


def _disable_worker_timeout(previous_handler: Optional[Any]) -> None:
    """Clear the gate child alarm and restore the previous handler.

    Parameters
    ----------
    previous_handler : object or None
        Handler returned by ``_enable_worker_timeout``.

    Returns
    -------
    None
        Process-local signal state is restored.
    """
    if not hasattr(signal, "SIGALRM"):
        return
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    if previous_handler is not None:
        signal.signal(signal.SIGALRM, previous_handler)


def _burn_cpu(stop: Any) -> None:
    """Consume CPU until the parent requests shutdown.

    Parameters
    ----------
    stop : multiprocessing.Event
        Event set by the parent when the load phase should end.

    Returns
    -------
    None
        The function exits when ``stop`` is set.
    """
    value = 1.000001
    while not stop.is_set():
        value = math.sin(value) * math.cos(value) + 1.000001


def _load_workers(count: Optional[int]) -> int:
    """Return the number of synthetic-load worker processes to start.

    Parameters
    ----------
    count : int, optional
        User-supplied worker count. ``None`` chooses a conservative default.

    Returns
    -------
    int
        Positive worker count.
    """
    if count is not None:
        return max(1, int(count))
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count - 1)


def _start_load(count: Optional[int]) -> tuple[Any, list[mp.Process]]:
    """Start synthetic CPU load workers.

    Parameters
    ----------
    count : int, optional
        User-supplied worker count.

    Returns
    -------
    tuple[multiprocessing.Event, list[multiprocessing.Process]]
        Stop event and started worker processes.
    """
    stop = mp.Event()
    processes = [
        mp.Process(target=_burn_cpu, args=(stop,), name=f"dagua-det-load-{index}")
        for index in range(_load_workers(count))
    ]
    for process in processes:
        process.start()
    return stop, processes


def _stop_load(stop: Any, processes: Sequence[mp.Process]) -> None:
    """Stop synthetic-load worker processes.

    Parameters
    ----------
    stop : multiprocessing.Event
        Stop event returned by ``_start_load``.
    processes : Sequence[multiprocessing.Process]
        Started worker processes.

    Returns
    -------
    None
        All workers are joined or terminated.
    """
    stop.set()
    for process in processes:
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _graph_map() -> dict[str, TestGraph]:
    """Return corpus graphs keyed by name.

    Returns
    -------
    dict[str, TestGraph]
        Complete benchmark graph registry.
    """
    return {test_graph.name: test_graph for test_graph in get_test_graphs()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL telemetry records from ``path``.

    Parameters
    ----------
    path : Path
        Telemetry file path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed JSON object records. Missing files return an empty list.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _score(test_graph: TestGraph, pos: torch.Tensor) -> float:
    """Score one native layout with the full benchmark ruler.

    Parameters
    ----------
    test_graph : TestGraph
        Benchmark graph metadata.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Full-ruler composite score.
    """
    graph = test_graph.graph
    declared = _declares_hierarchy(test_graph)
    metrics = full(
        pos.to(dtype=torch.float32),
        graph.edge_index,
        node_sizes=graph.node_sizes,
        cluster_ids=graph.cluster_ids,
        direction=graph.direction,
        declared_hierarchical=declared,
        clusters=graph.clusters or None,
        cluster_parents=graph.cluster_parents or None,
        cluster_labels=graph.cluster_labels or None,
    )
    metrics["declared_hierarchical"] = declared
    return float(composite_auto(metrics, is_semantically_directed(test_graph)))


def _plan_shape(records: Sequence[dict[str, Any]]) -> str:
    """Return a compact plan-shape signature from native telemetry.

    Parameters
    ----------
    records : Sequence[dict[str, Any]]
        Parsed native telemetry records.

    Returns
    -------
    str
        Stable summary of W5 plan, winner, and budget-skip events.
    """
    w5 = [record for record in records if record.get("event") == "native_w5_finisher"]
    skips = [record for record in records if record.get("event") == "native_undirected_arm_skip"]
    if not w5 and not skips:
        return "no_native_telemetry"
    w5_parts = [
        "w5:{winner}:{skip}:{seeds}:{steps}:{checkpoints}".format(
            winner=record.get("winner_name"),
            skip=record.get("skipped_reason"),
            seeds=record.get("plan_seeds"),
            steps=record.get("plan_steps"),
            checkpoints=record.get("plan_checkpoints"),
        )
        for record in w5
    ]
    skip_parts = [
        "skip:{arm}:{reason}".format(arm=record.get("arm"), reason=record.get("reason"))
        for record in skips
    ]
    return "|".join(w5_parts + skip_parts)


def _marketplace_arms(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return flattened marketplace arm status records.

    Parameters
    ----------
    records : Sequence[dict[str, Any]]
        Parsed native telemetry records.

    Returns
    -------
    list[dict[str, Any]]
        One JSON-ready status record per marketplace arm.
    """
    arms: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "native_candidate_marketplace":
            continue
        for arm in record.get("arms", []):
            if not isinstance(arm, dict):
                continue
            arms.append(
                {
                    "route": record.get("route"),
                    "winner": record.get("winner"),
                    "name": arm.get("name"),
                    "family": arm.get("family"),
                    "status": arm.get("status"),
                    "reason": arm.get("reason"),
                    "full_score": arm.get("full_score"),
                }
            )
    return arms


def _record_matches_substring(record: dict[str, Any], arm_substr: str) -> bool:
    """Return whether an arm telemetry record matches a substring.

    Parameters
    ----------
    record : dict[str, Any]
        Parsed telemetry record.
    arm_substr : str
        Case-sensitive arm substring requested by the CLI.

    Returns
    -------
    bool
        ``True`` when the substring appears in the arm name or family.
    """
    candidates = (record.get("arm"), record.get("name"), record.get("family"))
    return any(arm_substr in str(candidate) for candidate in candidates if candidate is not None)


def _verdict(records: Sequence[dict[str, Any]], score: float) -> str:
    """Return a stable admission verdict for one run.

    Parameters
    ----------
    records : Sequence[dict[str, Any]]
        Parsed native telemetry records.
    score : float
        Full-ruler composite score.

    Returns
    -------
    str
        Verdict string used for stability comparison.
    """
    if not math.isfinite(score):
        return "non_finite_score"
    reasons = sorted(
        {
            str(record.get("reason"))
            for record in records
            if record.get("event") == "native_undirected_arm_skip"
        }
    )
    winners = [
        str(record.get("winner_name"))
        for record in records
        if record.get("event") == "native_w5_finisher"
    ]
    if reasons:
        return "skips:" + ",".join(reasons)
    if winners:
        return "w5:" + ",".join(winners)
    return "admitted_no_skip"


def _run_once_in_child_process(
    test_graph: TestGraph,
    mode: str,
    index: int,
    timeout_s: float,
) -> GateRun:
    """Run native dagua once inside a watchdog child process.

    Parameters
    ----------
    test_graph : TestGraph
        Graph to lay out.
    mode : str
        Load mode label.
    index : int
        Zero-based run index.
    timeout_s : float
        Per-run benchmark timeout.

    Returns
    -------
    GateRun
        Captured run result.
    """
    competitor = DaguaCompetitor()
    device = competitor.device
    child_started_wall = time.perf_counter()
    child_started_process = time.process_time()
    with tempfile.TemporaryDirectory(prefix="dagua-native-det-") as tmp:
        telemetry_path = Path(tmp) / "telemetry.jsonl"
        old_w5 = os.environ.get("DAGUA_W5_TELEMETRY_PATH")
        old_arm = os.environ.get("DAGUA_ARM_TELEMETRY_PATH")
        os.environ["DAGUA_W5_TELEMETRY_PATH"] = str(telemetry_path)
        os.environ["DAGUA_ARM_TELEMETRY_PATH"] = str(telemetry_path)
        previous_handler = _enable_worker_timeout(timeout_s)
        try:
            result = competitor.layout(test_graph.graph, timeout=timeout_s, seed=42)
        finally:
            _disable_worker_timeout(previous_handler)
            if old_w5 is None:
                os.environ.pop("DAGUA_W5_TELEMETRY_PATH", None)
            else:
                os.environ["DAGUA_W5_TELEMETRY_PATH"] = old_w5
            if old_arm is None:
                os.environ.pop("DAGUA_ARM_TELEMETRY_PATH", None)
            else:
                os.environ["DAGUA_ARM_TELEMETRY_PATH"] = old_arm
        if result.pos is None:
            raise RuntimeError(f"{test_graph.name} returned no positions: {result.error}")
        records = _read_jsonl(telemetry_path)
    score = _score(test_graph, result.pos)
    process_runtime_s = max(0.0, time.process_time() - child_started_process)
    wall_runtime_s = max(1.0e-9, time.perf_counter() - child_started_wall)
    return GateRun(
        graph=test_graph.name,
        mode=mode,
        index=index,
        verdict=_verdict(records, score),
        score=score,
        runtime_s=float(result.runtime_seconds),
        process_runtime_s=process_runtime_s,
        cpu_wall_ratio=process_runtime_s / wall_runtime_s,
        torch_num_threads=int(torch.get_num_threads()),
        device=device,
        plan_shape=_plan_shape(records),
        marketplace_arms=_marketplace_arms(records),
        telemetry=records,
    )


def _run_once_child(
    graph_name: str,
    mode: str,
    index: int,
    timeout_s: float,
    queue: Any,
) -> None:
    """Run one layout and publish the result or error to the parent process.

    Parameters
    ----------
    graph_name : str
        Benchmark graph name.
    mode : str
        Load mode label.
    index : int
        Zero-based run index.
    timeout_s : float
        Per-run benchmark timeout.
    queue : Any
        Multiprocessing queue used to return the child payload.

    Returns
    -------
    None
        The result is written to ``queue``.
    """
    try:
        test_graph = _graph_map()[graph_name]
        queue.put(("ok", _run_once_in_child_process(test_graph, mode, index, timeout_s)))
    except Exception as exc:  # noqa: BLE001 -- parent reports child failure as gate failure
        queue.put(("error", repr(exc)))


def _run_once(
    test_graph: TestGraph,
    mode: str,
    index: int,
    timeout_s: float,
    run_timeout_s: float,
) -> GateRun:
    """Run native dagua once and capture score plus telemetry.

    Parameters
    ----------
    test_graph : TestGraph
        Graph to lay out.
    mode : str
        Load mode label.
    index : int
        Zero-based run index.
    timeout_s : float
        Per-run benchmark timeout.
    run_timeout_s : float
        External wall-clock watchdog for the gate process.

    Returns
    -------
    GateRun
        Captured run result.
    """
    context = mp.get_context("spawn")
    queue: Any = context.Queue(maxsize=1)
    process = context.Process(
        target=_run_once_child,
        args=(test_graph.name, mode, index, timeout_s, queue),
        name=f"dagua-det-{test_graph.name}-{mode}-{index}",
    )
    process.start()
    process.join(timeout=max(1.0, float(run_timeout_s)))
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        raise GateRunTimeoutError(
            f"run timeout exceeded for {test_graph.name} {mode}[{index}] after {run_timeout_s:.1f}s"
        )
    if process.exitcode not in {0, None} and queue.empty():
        raise RuntimeError(
            f"{test_graph.name} {mode}[{index}] child exited with code {process.exitcode}"
        )
    status, payload = queue.get(timeout=1.0)
    if status == "error":
        raise RuntimeError(f"{test_graph.name} {mode}[{index}] failed: {payload}")
    return payload


def _run_mode(
    test_graph: TestGraph,
    mode: str,
    runs: int,
    timeout_s: float,
    run_timeout_s: float,
    load_workers: Optional[int],
) -> list[GateRun]:
    """Run one graph repeatedly in one load mode.

    Parameters
    ----------
    test_graph : TestGraph
        Graph to lay out.
    mode : str
        Load mode, either ``"idle"`` or ``"load"``.
    runs : int
        Number of repeated runs.
    timeout_s : float
        Per-run benchmark timeout.
    run_timeout_s : float
        External wall-clock watchdog for each run.
    load_workers : int, optional
        Synthetic-load worker count.

    Returns
    -------
    list[GateRun]
        Captured run results.
    """
    stop: Optional[Any] = None
    processes: list[mp.Process] = []
    if mode == "load":
        stop, processes = _start_load(load_workers)
        time.sleep(0.25)
    try:
        return [
            _run_once(test_graph, mode, index, timeout_s, run_timeout_s) for index in range(runs)
        ]
    finally:
        if stop is not None:
            _stop_load(stop, processes)


def _assert_stable(runs: Sequence[GateRun], threshold: float) -> list[str]:
    """Return determinism failures for a graph's collected runs.

    Parameters
    ----------
    runs : Sequence[GateRun]
        Runs for one graph across idle and load modes.
    threshold : float
        Maximum allowed score spread.

    Returns
    -------
    list[str]
        Human-readable failure messages.
    """
    failures: list[str] = []
    verdicts = {run.verdict for run in runs}
    if len(verdicts) != 1:
        failures.append(f"unstable verdicts: {sorted(verdicts)}")
    scores = [run.score for run in runs]
    spread = max(scores) - min(scores)
    if spread > threshold:
        failures.append(f"score spread {spread:.4f} exceeds {threshold:.4f}")
    return failures


def _assert_expect_no_skip(runs: Sequence[GateRun], arm_substrs: Sequence[str]) -> list[str]:
    """Return failures for expected arms that show skip telemetry.

    Parameters
    ----------
    runs : Sequence[GateRun]
        Runs for one graph across idle and load modes.
    arm_substrs : Sequence[str]
        Arm substrings that must not show budget/skip statuses.

    Returns
    -------
    list[str]
        Human-readable failure messages.
    """
    failures: list[str] = []
    for arm_substr in arm_substrs:
        for run in runs:
            for record in run.telemetry:
                if not _record_matches_substring(record, arm_substr):
                    continue
                if (
                    record.get("event") == "native_undirected_arm_skip"
                    and record.get("reason") == "insufficient_predicted_budget"
                ):
                    failures.append(
                        f"{run.mode}[{run.index}] {arm_substr} skipped: "
                        "insufficient_predicted_budget"
                    )
            for arm in run.marketplace_arms:
                if not _record_matches_substring(arm, arm_substr):
                    continue
                status = str(arm.get("status", ""))
                reason = str(arm.get("reason", ""))
                if "skip" in status or "skip" in reason:
                    failures.append(
                        f"{run.mode}[{run.index}] {arm_substr} marketplace skip: {status}/{reason}"
                    )
    return failures


def _assert_min_score(runs: Sequence[GateRun], min_score: Optional[float]) -> list[str]:
    """Return failures for runs below the requested score floor.

    Parameters
    ----------
    runs : Sequence[GateRun]
        Runs for one graph across idle and load modes.
    min_score : float, optional
        Minimum composite score required for every run.

    Returns
    -------
    list[str]
        Human-readable failure messages.
    """
    if min_score is None:
        return []
    return [
        f"{run.mode}[{run.index}] score {run.score:.4f} below {float(min_score):.4f}"
        for run in runs
        if run.score < float(min_score)
    ]


def _assert_stable_fallback(runs: Sequence[GateRun], min_score: Optional[float]) -> list[str]:
    """Return failures for the certified stable-fallback branch.

    Parameters
    ----------
    runs : Sequence[GateRun]
        Runs for one graph across idle and load modes.
    min_score : float, optional
        Optional high-score branch floor. When present, fallback scores must
        remain below it so the gate cannot silently pass the 84-class branch.

    Returns
    -------
    list[str]
        Human-readable failure messages.
    """
    failures: list[str] = []
    if any(not math.isfinite(run.score) for run in runs):
        failures.append("stable fallback produced a non-finite score")
    if min_score is not None and any(run.score >= float(min_score) for run in runs):
        failures.append(
            "stable fallback requested but at least one run reached "
            f"the high-score floor {float(min_score):.4f}"
        )
    return failures


def _print_runs(runs: Iterable[GateRun]) -> None:
    """Print gate run records as JSON lines.

    Parameters
    ----------
    runs : Iterable[GateRun]
        Run records to print.

    Returns
    -------
    None
        JSON payloads are written to stdout.
    """
    for run in runs:
        print(
            json.dumps(
                {
                    "graph": run.graph,
                    "mode": run.mode,
                    "index": run.index,
                    "verdict": run.verdict,
                    "score": run.score,
                    "runtime_s": run.runtime_s,
                    "process_runtime_s": run.process_runtime_s,
                    "cpu_wall_ratio": run.cpu_wall_ratio,
                    "torch_num_threads": run.torch_num_threads,
                    "device": run.device,
                    "plan_shape": run.plan_shape,
                    "marketplace_arms": run.marketplace_arms,
                    "telemetry": run.telemetry,
                },
                sort_keys=True,
            )
        )


def _replay_signature(run: GateRun) -> str:
    """Return the admission and W5 plan signature used by parity replay.

    Parameters
    ----------
    run : GateRun
        Captured gate run.

    Returns
    -------
    str
        Stable signature combining verdict and plan shape.
    """
    return f"{run.verdict}::{run.plan_shape}"


def _default_expected_diffs(graphs: Sequence[str]) -> list[str]:
    """Return the documented expected-diff rows present in ``graphs``.

    Parameters
    ----------
    graphs : Sequence[str]
        Requested replay graph names.

    Returns
    -------
    list[str]
        Rows in the scale_1k class, the only intended replay diffs.
    """
    return [name for name in graphs if "scale_1k" in name]


def _print_parity_rows(rows: Iterable[ParityReplayRow], expected_diffs: Sequence[str]) -> None:
    """Print parity-replay rows as JSON lines.

    Parameters
    ----------
    rows : Iterable[ParityReplayRow]
        Replay comparison rows.
    expected_diffs : Sequence[str]
        Documented expected-diff row names.

    Returns
    -------
    None
        JSON payloads are written to stdout.
    """
    expected = list(expected_diffs)
    for row in rows:
        print(
            json.dumps(
                {
                    "event": "native_parity_replay",
                    "graph": row.graph,
                    "baseline_signature": row.baseline_signature,
                    "replay_signature": row.replay_signature,
                    "matches": row.matches,
                    "expected_diff": row.expected_diff,
                    "expected_diffs": expected,
                },
                sort_keys=True,
            )
        )


def _run_parity_replay(args: argparse.Namespace) -> int:
    """Run the cheap admission and W5-plan parity replay command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``parity-replay`` arguments.

    Returns
    -------
    int
        ``0`` when all unexpected diffs are absent, else ``1``.
    """
    graphs = _graph_map()
    requested = list(args.graphs)
    expected_diffs = list(args.expected_diff or _default_expected_diffs(requested))
    expected_diff_set = set(expected_diffs)
    rows: list[ParityReplayRow] = []
    failures: list[str] = []
    run_timeout_s = (
        float(args.timeout) + max(0.0, float(args.run_timeout_margin))
        if args.run_timeout is None
        else float(args.run_timeout)
    )
    for graph_name in requested:
        if graph_name not in graphs:
            failures.append(f"{graph_name}: unknown graph")
            continue
        test_graph = graphs[graph_name]
        try:
            baseline = _run_once(test_graph, "idle-main", 0, args.timeout, run_timeout_s)
            replay = _run_once(test_graph, "ledger-replay", 0, args.timeout, run_timeout_s)
        except GateRunTimeoutError as exc:
            failures.append(f"{graph_name}: {exc}")
            continue
        baseline_signature = _replay_signature(baseline)
        replay_signature = _replay_signature(replay)
        matches = baseline_signature == replay_signature
        expected_diff = graph_name in expected_diff_set
        rows.append(
            ParityReplayRow(
                graph=graph_name,
                baseline_signature=baseline_signature,
                replay_signature=replay_signature,
                matches=matches,
                expected_diff=expected_diff,
            )
        )
        if matches and expected_diff:
            continue
        if not matches and not expected_diff:
            failures.append(f"{graph_name}: unexpected replay diff")
    _print_parity_rows(rows, expected_diffs)
    if failures:
        print("FAIL native parity replay", flush=True)
        for failure in failures:
            print(f" - {failure}", flush=True)
        return 1
    print("PASS native parity replay", flush=True)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str], optional
        Argument vector, excluding program name.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "parity-replay":
        parser = argparse.ArgumentParser(description="Replay native budget admission parity.")
        parser.add_argument("command", choices=("parity-replay",))
        parser.add_argument("graphs", nargs="*", default=list(DEFAULT_GRAPHS))
        parser.add_argument("--timeout", type=float, default=300.0)
        parser.add_argument("--run-timeout", type=float, default=None)
        parser.add_argument("--run-timeout-margin", type=float, default=DEFAULT_TIMEOUT_MARGIN_S)
        parser.add_argument("--expected-diff", action="append", default=[])
        return parser.parse_args(args_list)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", default=list(DEFAULT_GRAPHS))
    parser.add_argument("--engine", default="dagua", choices=("dagua",))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--run-timeout", type=float, default=None)
    parser.add_argument("--run-timeout-margin", type=float, default=DEFAULT_TIMEOUT_MARGIN_S)
    parser.add_argument("--score-spread", type=float, default=DEFAULT_SCORE_SPREAD)
    parser.add_argument("--load-workers", type=int, default=None)
    parser.add_argument("--expect-no-skip", action="append", default=[])
    parser.add_argument("--expect-stable-fallback", action="store_true")
    parser.add_argument("--min-score", type=float, default=None)
    parsed = parser.parse_args(args_list)
    setattr(parsed, "command", "gate")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the native determinism gate.

    Parameters
    ----------
    argv : Sequence[str], optional
        Argument vector, excluding program name.

    Returns
    -------
    int
        ``0`` when all graph verdicts and score spreads are stable, else ``1``.
    """
    args = parse_args(argv)
    if args.command == "parity-replay":
        return _run_parity_replay(args)
    graphs = _graph_map()
    failures: list[str] = []
    run_timeout_s = (
        float(args.timeout) + max(0.0, float(args.run_timeout_margin))
        if args.run_timeout is None
        else float(args.run_timeout)
    )
    for graph_name in args.graphs:
        if graph_name not in graphs:
            failures.append(f"{graph_name}: unknown graph")
            continue
        test_graph = graphs[graph_name]
        runs = []
        try:
            runs.extend(
                _run_mode(
                    test_graph,
                    "idle",
                    args.runs,
                    args.timeout,
                    run_timeout_s,
                    args.load_workers,
                )
            )
            runs.extend(
                _run_mode(
                    test_graph,
                    "load",
                    args.runs,
                    args.timeout,
                    run_timeout_s,
                    args.load_workers,
                )
            )
        except GateRunTimeoutError as exc:
            failures.append(f"{graph_name}: {exc}")
            continue
        _print_runs(runs)
        for failure in _assert_stable(runs, args.score_spread):
            failures.append(f"{graph_name}: {failure}")
        for failure in _assert_expect_no_skip(runs, args.expect_no_skip):
            failures.append(f"{graph_name}: {failure}")
        for failure in _assert_min_score(runs, args.min_score):
            failures.append(f"{graph_name}: {failure}")
        if args.expect_stable_fallback:
            for failure in _assert_stable_fallback(runs, args.min_score):
                failures.append(f"{graph_name}: {failure}")
    if failures:
        print("FAIL native determinism gate", flush=True)
        for failure in failures:
            print(f" - {failure}", flush=True)
        return 1
    print("PASS native determinism gate", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
