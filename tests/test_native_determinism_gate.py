"""Unit tests for the native determinism gate helpers."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import pytest

from scripts import native_determinism_gate as gate


def _put_large_payload(queue: Any) -> None:
    """Child target: publish a payload far larger than the pipe buffer.

    Parameters
    ----------
    queue : Any
        Multiprocessing queue shared with the parent.
    """
    telemetry = [{"event": "unit", "blob": "x" * 1024} for _ in range(1024)]
    queue.put(("ok", telemetry))


def _gate_run(
    *,
    score: float,
    telemetry: list[dict[str, object]],
    marketplace_arms: list[dict[str, object]],
) -> gate.GateRun:
    """Build a minimal gate run for assertion helper tests.

    Parameters
    ----------
    score : float
        Composite score assigned to the run.
    telemetry : list[dict[str, object]]
        Parsed native telemetry records.
    marketplace_arms : list[dict[str, object]]
        Flattened marketplace arm records.

    Returns
    -------
    gate.GateRun
        Synthetic run result.
    """
    return gate.GateRun(
        graph="unit",
        mode="idle",
        index=0,
        verdict="admitted_no_skip",
        score=score,
        runtime_s=1.0,
        process_runtime_s=0.5,
        cpu_wall_ratio=0.5,
        torch_num_threads=1,
        device="cpu",
        plan_shape="unit",
        marketplace_arms=marketplace_arms,
        telemetry=telemetry,
    )


def test_expect_no_skip_fails_on_matching_budget_skip() -> None:
    """Expected arm assertions reject insufficient predicted-budget skips."""
    run = _gate_run(
        score=81.0,
        telemetry=[
            {
                "event": "native_undirected_arm_skip",
                "arm": "arm_s_stress",
                "reason": "insufficient_predicted_budget",
            }
        ],
        marketplace_arms=[],
    )

    failures = gate._assert_expect_no_skip([run], ["arm_s"])

    assert failures == ["idle[0] arm_s skipped: insufficient_predicted_budget"]


def test_expect_no_skip_allows_matching_accepted_marketplace_arm() -> None:
    """Accepted or winning marketplace arms satisfy the no-skip assertion."""
    run = _gate_run(
        score=84.0,
        telemetry=[],
        marketplace_arms=[
            {
                "name": "arm_s_stress_cluster",
                "family": "arm_s_stress",
                "status": "winner",
                "reason": "highest_full_score",
            }
        ],
    )

    assert gate._assert_expect_no_skip([run], ["arm_s"]) == []


def test_min_score_fails_below_floor() -> None:
    """Minimum score assertions reject every run below the floor."""
    run = _gate_run(score=79.5, telemetry=[], marketplace_arms=[])

    failures = gate._assert_min_score([run], 80.0)

    assert failures == ["idle[0] score 79.5000 below 80.0000"]


def test_marketplace_arms_flattens_candidate_records() -> None:
    """Marketplace telemetry is exposed as per-arm gate records."""
    records = [
        {
            "event": "native_candidate_marketplace",
            "route": "undirected",
            "winner": "arm_s_stress",
            "arms": [
                {
                    "name": "arm_s_stress",
                    "family": "arm_s_stress",
                    "status": "winner",
                    "reason": "highest_full_score",
                    "full_score": 84.2,
                }
            ],
        }
    ]

    assert gate._marketplace_arms(records) == [
        {
            "route": "undirected",
            "winner": "arm_s_stress",
            "name": "arm_s_stress",
            "family": "arm_s_stress",
            "status": "winner",
            "reason": "highest_full_score",
            "full_score": 84.2,
        }
    ]


def test_parse_args_accepts_new_acceptance_flags() -> None:
    """CLI parsing exposes score floors, skip assertions, and timeout margin."""
    args = gate.parse_args(
        [
            "r8_nested_scale_1k_budget",
            "--expect-no-skip",
            "arm_s",
            "--expect-stable-fallback",
            "--min-score",
            "80",
            "--run-timeout-margin",
            "120",
        ]
    )

    assert args.expect_no_skip == ["arm_s"]
    assert args.expect_stable_fallback is True
    assert args.min_score == 80.0
    assert args.run_timeout_margin == 120.0


def test_parse_args_uses_large_default_timeout_margin() -> None:
    """The default watchdog margin is large enough for slow scored rows."""
    args = gate.parse_args(["r8_nested_scale_1k_budget"])

    assert args.run_timeout_margin == 180.0


def test_parse_args_accepts_parity_replay_subcommand() -> None:
    """Parity replay has a cheap subcommand with explicit expected diffs."""
    args = gate.parse_args(
        [
            "parity-replay",
            "rgg_500",
            "r8_nested_scale_1k_budget",
            "--expected-diff",
            "r8_nested_scale_1k_budget",
        ]
    )

    assert args.command == "parity-replay"
    assert args.graphs == ["rgg_500", "r8_nested_scale_1k_budget"]
    assert args.expected_diff == ["r8_nested_scale_1k_budget"]


def test_stable_fallback_rejects_high_score_branch() -> None:
    """Fallback assertion fails if a run reaches the high-score floor."""
    run = _gate_run(score=84.2, telemetry=[], marketplace_arms=[])

    failures = gate._assert_stable_fallback([run], 80.0)

    assert failures == [
        "stable fallback requested but at least one run reached the high-score floor 80.0000"
    ]


def test_collect_child_payload_drains_large_payload_before_join() -> None:
    """A >64KB child payload must not convert success into a false timeout.

    The child's queue feeder thread blocks at exit while the payload exceeds
    the pipe buffer; joining before draining deadlocks until the watchdog
    fires. The drain-first helper must return the payload well within the
    timeout budget.
    """
    context = mp.get_context("fork")
    queue: Any = context.Queue(maxsize=1)
    process = context.Process(target=_put_large_payload, args=(queue,))
    process.start()
    try:
        status, payload = gate._collect_child_payload(
            process, queue, run_timeout_s=60.0, label="unit idle[0]"
        )
    finally:
        process.join(timeout=10.0)
        if process.is_alive():  # pragma: no cover -- cleanup path
            process.terminate()

    assert status == "ok"
    assert len(payload) == 1024


def test_collect_child_payload_reports_silent_child_exit() -> None:
    """A child that exits without publishing raises a labeled RuntimeError."""
    context = mp.get_context("fork")
    queue: Any = context.Queue(maxsize=1)
    process = context.Process(target=lambda: None)
    process.start()
    process.join(timeout=10.0)

    with pytest.raises(RuntimeError, match="unit idle\\[0\\] child exited"):
        gate._collect_child_payload(process, queue, run_timeout_s=5.0, label="unit idle[0]")


def test_gate_main_reports_child_error_as_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A child crash lands in the FAIL summary instead of a traceback."""
    monkeypatch.setattr(gate, "_graph_map", lambda: {"unit": object()})

    def _boom(*args: object, **kwargs: object) -> list[gate.GateRun]:
        raise RuntimeError("unit idle[0] child exited with code 1")

    monkeypatch.setattr(gate, "_run_mode", _boom)

    exit_code = gate.main(["unit"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "FAIL native determinism gate" in captured.out
    assert "unit: unit idle[0] child exited with code 1" in captured.out


def test_parity_replay_row_reports_expected_diff_structure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Parity replay output includes the documented expected-diff list."""
    row = gate.ParityReplayRow(
        graph="r8_nested_scale_1k_budget",
        baseline_signature="skip:fCoSE",
        replay_signature="admit:arm_s",
        matches=False,
        expected_diff=True,
    )

    gate._print_parity_rows([row], ["r8_nested_scale_1k_budget"])
    captured = capsys.readouterr()

    assert '"event": "native_parity_replay"' in captured.out
    assert '"expected_diffs": ["r8_nested_scale_1k_budget"]' in captured.out
