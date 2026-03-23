"""Tests for taxi (Manhattan) edge routing."""

from __future__ import annotations

import numpy as np
import pytest

from dagua.edges import BezierCurve, _compute_curve, _compute_taxi


class TestTaxiRouting:
    """Taxi routing should preserve endpoints and axis-aligned bends."""

    def test_tb_direction(self) -> None:
        """TB direction should route vertical then horizontal.

        Returns
        -------
        None
        """
        curve = _compute_taxi(0.0, 0.0, 100.0, -100.0, "TB")

        assert isinstance(curve, BezierCurve)
        assert curve.p0 == pytest.approx((0.0, 0.0))
        assert curve.p1 == pytest.approx((100.0, -100.0))
        assert curve.cp1 == pytest.approx((0.0, -35.0))
        assert curve.cp2 == pytest.approx((100.0, -65.0))
        assert curve.waypoints is not None
        assert np.allclose(
            np.asarray(curve.waypoints, dtype=float),
            np.asarray(
                (
                    (0.0, 0.0),
                    (0.0, -35.0),
                    (50.0, -35.0),
                    (50.0, -65.0),
                    (100.0, -65.0),
                    (100.0, -100.0),
                ),
                dtype=float,
            ),
        )

    def test_lr_direction(self) -> None:
        """LR direction should route horizontal then vertical.

        Returns
        -------
        None
        """
        curve = _compute_taxi(0.0, 0.0, 100.0, 50.0, "LR")

        assert isinstance(curve, BezierCurve)
        assert curve.cp1 == pytest.approx((35.0, 0.0))
        assert curve.cp2 == pytest.approx((65.0, 50.0))
        assert curve.waypoints is not None
        assert np.allclose(
            np.asarray(curve.waypoints, dtype=float),
            np.asarray(
                (
                    (0.0, 0.0),
                    (35.0, 0.0),
                    (35.0, 25.0),
                    (65.0, 25.0),
                    (65.0, 50.0),
                    (100.0, 50.0),
                ),
                dtype=float,
            ),
        )

    def test_same_point(self) -> None:
        """Coincident endpoints should stay degenerate.

        Returns
        -------
        None
        """
        curve = _compute_taxi(50.0, 50.0, 50.0, 50.0, "TB")

        assert curve.p0 == curve.p1
        assert curve.cp1 == curve.cp2 == curve.p0
        assert curve.waypoints is not None
        assert np.allclose(np.asarray(curve.waypoints, dtype=float), np.asarray(((50.0, 50.0),)))

    def test_dispatcher_uses_taxi_mode(self) -> None:
        """The routing dispatcher should delegate ``routing='taxi'`` correctly.

        Returns
        -------
        None
        """
        direct = _compute_taxi(10.0, 20.0, 90.0, 60.0, "LR")
        dispatched = _compute_curve(10.0, 20.0, 90.0, 60.0, "LR", "taxi", 0.4)

        assert dispatched == direct
