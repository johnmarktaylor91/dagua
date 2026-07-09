"""r80-S4 Stage 1 probe: undirected-class portfolio headroom check.

For every corpus graph the eval oracle classifies as semantically undirected,
compute candidate layouts using dagua's own bit-faithful sfdp/neato/kk
reimplementations (algorithm="sfdp"/"neato"/"kk"), apply the existing
size-aware overlap projector, and score every candidate with the identical
honest composite the baseline harness uses. Compare against the frozen
current-dagua row and frozen best-external row for each graph.

This script writes NO product code changes -- it is a standalone,
reusable probe. See .project-context/research/r79_native/briefs/
r80_s4_undirected_portfolio.md for the decision gate this feeds.

Usage
-----
    .venv/bin/python scripts/r80_probe_undirected_portfolio.py
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_WORKTREE = Path("/home/jtaylor/.claude/worktrees/dagua-native")
FROZEN_RESULTS = MAIN_WORKTREE / "eval_output" / "r79_baseline" / "results.json"
REPORT_PATH = REPO_ROOT / ".project-context" / "research" / "r79_native" / "P8_PORTFOLIO_PROBE.md"

TIE_BAND = 0.5
SEED = 42
CANDIDATE_ALGORITHMS = ["sfdp", "neato", "kk"]
# Graphs above this size lack frozen dagua rows in the baseline store (the
# harness itself does not complete tier="full" candidate evaluation at this
# scale within its timeout); skip full candidate runs above this cap and
# record why, rather than risk a multi-hour probe. All frozen-LOSS graphs
# (the gate-relevant subset) are well under this cap (max 500 nodes).
MAX_CANDIDATE_NODES = 600
# Per-candidate wall-clock cap. Candidates are run in a child process (this
# same script invoked with --worker) so a pathological convergence loop on
# one graph/algorithm pair (observed: neato on weighted small-world graphs)
# cannot stall the whole probe. Mirrors TIMEOUT_SECONDS in r79_baseline.py.
CANDIDATE_TIMEOUT_S = 150.0


@contextmanager
def _forbid_subprocess_spawns():
    """Raise if the wrapped block spawns a subprocess (no external-binary delegation)."""
    original_popen = subprocess.Popen

    def _blocked_popen(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"subprocess.Popen invoked during native-only candidate layout: "
            f"args={args!r} kwargs={kwargs!r}"
        )

    subprocess.Popen = _blocked_popen  # type: ignore[assignment]
    try:
        yield
    finally:
        subprocess.Popen = original_popen  # type: ignore[assignment]


def _load_frozen_rows() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return {graph_name: {engine_name: row}} for OK rows in the frozen store."""
    with open(FROZEN_RESULTS, "r") as f:
        payload = json.load(f)
    by_graph: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in payload["rows"]:
        if row.get("status") != "OK":
            continue
        by_graph.setdefault(row["graph"], {})[row["engine"]] = row
    return by_graph


def _best_external(engines: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for engine_name, row in engines.items():
        if engine_name == "dagua":
            continue
        if best is None or float(row["composite"]) > float(best["composite"]):
            best = row
    return best


def _run_candidate_in_process(
    graph: Any,
    algorithm: str,
) -> tuple[Optional[float], float, Optional[str]]:
    """Run one candidate algorithm + size-aware overlap projection, return (score, wall_s, error).

    Runs in the CURRENT process. Used by ``--worker`` mode (see ``main``),
    which is itself invoked as a child process with a wall-clock timeout so a
    pathological candidate cannot stall the whole probe.
    """
    from dagua.config import LayoutConfig
    from dagua.layout import layout
    from dagua.layout.projection import project_overlaps
    from dagua.metrics import composite_auto, evaluate

    t0 = time.perf_counter()
    try:
        config = LayoutConfig(
            algorithm=algorithm,
            seed=SEED,
            device="cpu",
            verbose=False,
        )
        with _forbid_subprocess_spawns():
            pos = layout(graph, config)
        pos = pos.detach().cpu().to(dtype=torch.float32)
        node_sizes = graph.node_sizes.to(dtype=pos.dtype)
        proj_pos = project_overlaps(pos.clone(), node_sizes)
        metrics = evaluate(graph, proj_pos, tier="full")
        score = float(composite_auto(metrics, is_semantically_directed=False))
        wall_s = time.perf_counter() - t0
        return score, wall_s, None
    except Exception as exc:  # noqa: BLE001
        wall_s = time.perf_counter() - t0
        return None, wall_s, f"{type(exc).__name__}: {exc}"


def _worker_loop_main() -> None:
    """Persistent worker: build the corpus once, then serve candidates over stdio.

    Protocol: parent writes ``"<graph_name> <algorithm>\\n"`` to this
    process's stdin; this process replies with exactly one JSON line
    (``{"score": ..., "wall_s": ..., "error": ...}``) per request on stdout.
    Emits a bare ``READY`` line once corpus construction (the expensive part
    -- ``get_test_graphs()`` takes ~2.5 minutes, dominated by node-size/font
    measurement across 116 graphs) completes, so the parent knows startup is
    done. Built once and reused across all candidates in a probe run so that
    per-candidate wall-clock enforcement (see ``_WorkerPool``) does not pay
    the corpus-build cost on every candidate.
    """
    from dagua.eval.graphs import get_test_graphs

    graphs_by_name = {tg.name: tg for tg in get_test_graphs()}
    for tg in graphs_by_name.values():
        tg.graph.compute_node_sizes()
    print("READY", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        graph_name, _, algorithm = line.partition(" ")
        match = graphs_by_name.get(graph_name)
        if match is None:
            print(
                json.dumps(
                    {"score": None, "wall_s": 0.0, "error": f"graph not found: {graph_name}"}
                ),
                flush=True,
            )
            continue
        score, wall_s, err = _run_candidate_in_process(match.graph, algorithm)
        print(json.dumps({"score": score, "wall_s": wall_s, "error": err}), flush=True)


class _WorkerPool:
    """Persistent-worker candidate runner with per-candidate timeout enforcement.

    A single long-lived child process (``--worker-loop``) builds the corpus
    once and serves candidates sequentially over stdin/stdout. The parent
    enforces ``CANDIDATE_TIMEOUT_S`` per candidate via a background reader
    thread + queue (``Popen.stdout.readline()`` has no native timeout). On
    timeout or worker death, the stalled worker is killed and a fresh one is
    started for subsequent candidates -- the ~2.5 minute corpus-build cost is
    only paid again when a candidate actually stalls, not on every call.
    """

    def __init__(self, startup_timeout_s: float = 600.0) -> None:
        self._startup_timeout_s = startup_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._start_worker()

    def _start_worker(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker-loop"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._queue = queue.Queue()

        def _reader() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for out_line in self._proc.stdout:
                self._queue.put(out_line)
            self._queue.put(None)  # EOF sentinel

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        try:
            ready_line = self._queue.get(timeout=self._startup_timeout_s)
        except Exception:
            ready_line = None
        if ready_line is None or ready_line.strip() != "READY":
            raise RuntimeError(f"probe worker failed to start (expected READY, got {ready_line!r})")

    def _kill_and_restart(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        self._start_worker()

    def run_candidate(
        self,
        graph_name: str,
        algorithm: str,
    ) -> tuple[Optional[float], float, Optional[str]]:
        """Run one candidate via the persistent worker, enforcing a timeout.

        Returns
        -------
        tuple[float | None, float, str | None]
            ``(score, wall_s, error)``.
        """
        assert self._proc is not None and self._proc.stdin is not None
        t0 = time.perf_counter()
        try:
            self._proc.stdin.write(f"{graph_name} {algorithm}\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._kill_and_restart()
            return None, time.perf_counter() - t0, "worker pipe broken; restarted worker"

        line = None
        deadline = t0 + CANDIDATE_TIMEOUT_S
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                candidate_line = self._queue.get(timeout=remaining)
            except Exception:  # queue.Empty
                break
            if candidate_line is None:
                break  # worker EOF (died)
            stripped = candidate_line.strip()
            if stripped.startswith("{"):
                line = stripped
                break
            # Stray non-protocol output (should not happen with verbose=False,
            # but tolerate defensively rather than misattribute it as a result).

        wall_s = time.perf_counter() - t0
        if line is None:
            self._kill_and_restart()
            return (
                None,
                wall_s,
                f"TIMEOUT or worker died: exceeded {CANDIDATE_TIMEOUT_S:.0f}s wall-clock cap",
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, wall_s, f"worker result JSON decode error: {exc}"
        inner_wall_s = payload.get("wall_s")
        return (
            payload.get("score"),
            float(inner_wall_s) if inner_wall_s is not None else wall_s,
            payload.get("error"),
        )

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    from dagua.eval.graphs import get_test_graphs, is_semantically_directed

    frozen = _load_frozen_rows()
    all_graphs = get_test_graphs()
    undirected = [tg for tg in all_graphs if not is_semantically_directed(tg)]
    # Ascending node count: fast/small graphs (including all frozen-LOSS
    # graphs, gate-relevant) complete and report first; large graphs that
    # lack frozen dagua rows land last and are size-capped (see
    # MAX_CANDIDATE_NODES).
    undirected.sort(key=lambda tg: (tg.graph.num_nodes, tg.name))

    print(f"Probing {len(undirected)} undirected corpus graphs...", flush=True)
    print("Starting persistent candidate worker (corpus build, ~2-3 min)...", flush=True)
    pool = _WorkerPool()
    print("Worker ready.", flush=True)

    rows: List[Dict[str, Any]] = []
    for tg in undirected:
        graph = tg.graph
        graph.compute_node_sizes()
        engines = frozen.get(tg.name, {})
        dagua_row = engines.get("dagua")
        best_ext = _best_external(engines)

        candidate_scores: Dict[str, Optional[float]] = {}
        candidate_times: Dict[str, float] = {}
        candidate_errors: Dict[str, Optional[str]] = {}
        if int(graph.num_nodes) > MAX_CANDIDATE_NODES:
            for algo in CANDIDATE_ALGORITHMS:
                candidate_scores[algo] = None
                candidate_times[algo] = 0.0
                candidate_errors[algo] = (
                    f"SKIPPED: {graph.num_nodes} nodes > MAX_CANDIDATE_NODES={MAX_CANDIDATE_NODES}"
                )
            print(
                f"  {tg.name:40s} SKIPPED (N={graph.num_nodes} > {MAX_CANDIDATE_NODES})",
                flush=True,
            )
        else:
            for algo in CANDIDATE_ALGORITHMS:
                score, wall_s, err = pool.run_candidate(tg.name, algo)
                candidate_scores[algo] = score
                candidate_times[algo] = wall_s
                candidate_errors[algo] = err
                status = "OK" if err is None else f"ERROR: {err}"
                print(
                    f"  {tg.name:40s} {algo:6s} score={score} wall={wall_s:.2f}s {status}",
                    flush=True,
                )

        valid_scores = [s for s in candidate_scores.values() if s is not None]
        best_candidate_score = max(valid_scores) if valid_scores else None

        row: Dict[str, Any] = {
            "graph": tg.name,
            "num_nodes": int(graph.num_nodes),
            "current_dagua": float(dagua_row["composite"]) if dagua_row else None,
            "best_external_name": best_ext["engine"] if best_ext else None,
            "best_external_score": float(best_ext["composite"]) if best_ext else None,
            "candidate_scores": candidate_scores,
            "candidate_times": candidate_times,
            "candidate_errors": candidate_errors,
            "best_candidate_score": best_candidate_score,
        }
        if dagua_row is not None and best_ext is not None:
            delta_frozen = float(dagua_row["composite"]) - float(best_ext["composite"])
            row["frozen_verdict"] = (
                "WIN"
                if delta_frozen > TIE_BAND
                else ("TIE" if delta_frozen >= -TIE_BAND else "LOSS")
            )
        else:
            row["frozen_verdict"] = "NO_FROZEN_DATA"
        if best_candidate_score is not None and best_ext is not None:
            row["best_candidate_vs_best_external_delta"] = best_candidate_score - float(
                best_ext["composite"]
            )
        else:
            row["best_candidate_vs_best_external_delta"] = None
        rows.append(row)

    pool.shutdown()

    # Decision gate: among frozen-LOSS graphs, count how many have
    # max(candidates) >= best_external - 0.5.
    loss_rows = [r for r in rows if r["frozen_verdict"] == "LOSS"]
    gate_pass_rows = [
        r
        for r in loss_rows
        if r["best_candidate_score"] is not None
        and r["best_candidate_score"] >= (r["best_external_score"] - 0.5)
    ]
    gate_count = len(gate_pass_rows)
    gate_total = len(loss_rows)
    gate_passes = gate_count >= 10

    print()
    print(
        f"DECISION GATE: {gate_count} / {gate_total} loss-graphs improved to >= best_external - 0.5"
    )
    print(f"GATE {'PASSES' if gate_passes else 'FAILS'} (threshold: >= 10)")

    _write_report(rows, loss_rows, gate_count, gate_total, gate_passes)


def _fmt(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def _write_report(
    rows: List[Dict[str, Any]],
    loss_rows: List[Dict[str, Any]],
    gate_count: int,
    gate_total: int,
    gate_passes: bool,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# r80-S4 Stage 1 Probe: Undirected-Class Portfolio Headroom")
    lines.append("")
    lines.append(
        "Standalone probe (no product code changed). Every undirected corpus graph laid "
        "out with dagua's own sfdp/neato/kk reimplementations + size-aware overlap "
        "projection, scored with the identical honest composite the baseline harness uses "
        "(composite_auto with is_semantically_directed=False). Compared against frozen "
        "current-dagua and frozen best-external rows from "
        "eval_output/r79_baseline/results.json in the main worktree."
    )
    lines.append("")
    lines.append(
        f"**DECISION GATE**: {gate_count} / {gate_total} frozen-LOSS graphs improved to "
        f"max(candidates) >= best_external - 0.5. Threshold: >= 10. "
        f"**GATE {'PASSES' if gate_passes else 'FAILS'}**."
    )
    lines.append("")
    lines.append("## Per-graph table (all undirected corpus graphs)")
    lines.append("")
    header = (
        "| graph | N | current-dagua | best-external | sfdp+proj | neato+proj | kk+proj "
        "| best-cand vs best-ext | frozen verdict | wall sfdp/neato/kk (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        best_ext_cell = (
            f"{r['best_external_name']} {_fmt(r['best_external_score'])}"
            if r["best_external_name"]
            else "N/A"
        )
        cand = r["candidate_scores"]
        times = r["candidate_times"]
        wall_cell = (
            f"{times.get('sfdp', 0.0):.1f}/{times.get('neato', 0.0):.1f}/{times.get('kk', 0.0):.1f}"
        )
        delta = r["best_candidate_vs_best_external_delta"]
        delta_cell = f"{delta:+.2f}" if delta is not None else "N/A"
        lines.append(
            "| "
            + " | ".join(
                [
                    r["graph"],
                    str(r["num_nodes"]),
                    _fmt(r["current_dagua"]),
                    best_ext_cell,
                    _fmt(cand.get("sfdp")),
                    _fmt(cand.get("neato")),
                    _fmt(cand.get("kk")),
                    delta_cell,
                    r["frozen_verdict"],
                    wall_cell,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Frozen-LOSS graphs only (gate-relevant subset)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in loss_rows:
        best_ext_cell = f"{r['best_external_name']} {_fmt(r['best_external_score'])}"
        cand = r["candidate_scores"]
        times = r["candidate_times"]
        wall_cell = (
            f"{times.get('sfdp', 0.0):.1f}/{times.get('neato', 0.0):.1f}/{times.get('kk', 0.0):.1f}"
        )
        delta = r["best_candidate_vs_best_external_delta"]
        delta_cell = f"{delta:+.2f}" if delta is not None else "N/A"
        gate_hit = (
            "YES"
            if r["best_candidate_score"] is not None
            and r["best_candidate_score"] >= (r["best_external_score"] - 0.5)
            else "no"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    r["graph"],
                    str(r["num_nodes"]),
                    _fmt(r["current_dagua"]),
                    best_ext_cell,
                    _fmt(cand.get("sfdp")),
                    _fmt(cand.get("neato")),
                    _fmt(cand.get("kk")),
                    delta_cell,
                    f"LOSS (gate-hit: {gate_hit})",
                    wall_cell,
                ]
            )
            + " |"
        )
    lines.append("")
    errors = [
        (r["graph"], algo, err)
        for r in rows
        for algo, err in r["candidate_errors"].items()
        if err is not None
    ]
    if errors:
        lines.append("## Candidate errors")
        lines.append("")
        for graph_name, algo, err in errors:
            lines.append(f"- {graph_name} / {algo}: {err}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker-loop":
        _worker_loop_main()
    else:
        main()
