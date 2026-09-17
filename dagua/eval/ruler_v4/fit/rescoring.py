"""Scene-to-TYPE-M re-scoring bridge for P5 fitting inputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, Iterable, Mapping, Tuple

from dagua.eval.ruler_v4.fit.bank import JudgmentRow
from dagua.eval.ruler_v4.scene import ResultState, Scene
from dagua.eval.ruler_v4.score import OutputType, ScoringProfiles, score
from dagua.eval.ruler_v4.weight_table import WeightTable

SceneResolver = Callable[[str, JudgmentRow], Scene]


@dataclass(frozen=True)
class DrawingMeasurement:
    """Carry score-derived TYPE-M values for one banked drawing.

    Parameters
    ----------
    blind_drawing_id : str
        Schedule-owned opaque drawing id.
    observation_profile : str
        Profile under which the scene was judged and scored.
    subterms : mapping[str, float]
        Available frozen sub-term defects from ``score()``.
    measurement_version, policy_version : str
        Frozen artifact versions returned by ``score()``.
    """

    blind_drawing_id: str
    observation_profile: str
    subterms: Mapping[str, float]
    measurement_version: str
    policy_version: str

    def __post_init__(self) -> None:
        """Freeze the sub-term mapping.

        Raises
        ------
        ValueError
            If an identity or version is empty.
        """

        if not all(
            (
                self.blind_drawing_id,
                self.observation_profile,
                self.measurement_version,
                self.policy_version,
            )
        ):
            raise ValueError("measurement identities and versions must be nonempty")
        object.__setattr__(self, "subterms", MappingProxyType(dict(self.subterms)))


@dataclass(frozen=True)
class RescoredPair:
    """Join one judgment to independently re-scored A/B scenes.

    Parameters
    ----------
    judgment : JudgmentRow
        Accepted scheduled judgment.
    side_a, side_b : DrawingMeasurement
        TYPE-M scene measurements in displayed order.
    """

    judgment: JudgmentRow
    side_a: DrawingMeasurement
    side_b: DrawingMeasurement

    def subterm_delta(self, subterm_id: str) -> float:
        """Return the A-minus-B defect for one mutually available sub-term.

        Parameters
        ----------
        subterm_id : str
            Frozen score-visible sub-term id.

        Returns
        -------
        float
            ``side_a - side_b`` defect difference.

        Raises
        ------
        ValueError
            If the sub-term is not available on both sides.
        """

        if subterm_id not in self.side_a.subterms or subterm_id not in self.side_b.subterms:
            raise ValueError(f"pair lacks mutually available sub-term {subterm_id}")
        return self.side_a.subterms[subterm_id] - self.side_b.subterms[subterm_id]


class SceneRescorer:
    """Cache scene scoring while keeping observation profiles isolated.

    Parameters
    ----------
    scene_resolver : SceneResolver
        Callback resolving a schedule-owned blind id to a validated scene.
    weight_tables : mapping[str, WeightTable]
        Complete scoring table per opaque observation profile.
    profiles : mapping[str, ScoringProfiles]
        Frozen scoring profiles under the same keys.
    """

    def __init__(
        self,
        scene_resolver: SceneResolver,
        weight_tables: Mapping[str, WeightTable],
        profiles: Mapping[str, ScoringProfiles],
    ) -> None:
        """Initialize the profile-bound scorer.

        Parameters
        ----------
        scene_resolver : SceneResolver
            Blind-id scene resolver.
        weight_tables : mapping[str, WeightTable]
            Complete table per observation profile.
        profiles : mapping[str, ScoringProfiles]
            Scoring profiles under identical keys.

        Raises
        ------
        ValueError
            If profile keys are empty or disagree.
        """

        if not weight_tables or set(weight_tables) != set(profiles):
            raise ValueError("weight-table and scoring-profile keys must match and be nonempty")
        self._scene_resolver = scene_resolver
        self._weight_tables = MappingProxyType(dict(weight_tables))
        self._profiles = MappingProxyType(dict(profiles))
        self._cache: Dict[Tuple[str, str], DrawingMeasurement] = {}

    def score_drawing(self, blind_drawing_id: str, judgment: JudgmentRow) -> DrawingMeasurement:
        """Score one scheduled drawing through the public ``score()`` path.

        Parameters
        ----------
        blind_drawing_id : str
            Schedule-owned drawing identity.
        judgment : JudgmentRow
            Row supplying graph and observation-profile context.

        Returns
        -------
        DrawingMeasurement
            Fresh TYPE-M sub-term values, cached by drawing and profile.

        Raises
        ------
        ValueError
            If profile binding, graph identity, or TYPE-M output is invalid.
        """

        key = blind_drawing_id, judgment.observation_profile
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if judgment.observation_profile not in self._profiles:
            raise ValueError(f"unconfigured observation profile: {judgment.observation_profile}")
        scene = self._scene_resolver(blind_drawing_id, judgment)
        if scene.graph_hash != judgment.graph_hash:
            raise ValueError(
                f"scene graph {scene.graph_hash} does not match judgment graph "
                f"{judgment.graph_hash}"
            )
        result = score(
            scene,
            self._weight_tables[judgment.observation_profile],
            self._profiles[judgment.observation_profile],
        )
        if result.type_m.output_type is not OutputType.TYPE_M:
            raise ValueError("score bridge did not return TYPE-M output")
        subterms = {
            subterm_id: float(value)
            for breakdown in result.type_m.facets.values()
            if breakdown.result.state is ResultState.VALUE
            for subterm_id, value in breakdown.result.subterms.items()
        }
        measurement = DrawingMeasurement(
            blind_drawing_id=blind_drawing_id,
            observation_profile=judgment.observation_profile,
            subterms=subterms,
            measurement_version=result.measurement_version,
            policy_version=result.policy_version,
        )
        self._cache[key] = measurement
        return measurement

    def score_pair(self, judgment: JudgmentRow) -> RescoredPair:
        """Re-score both displayed sides of one judgment.

        Parameters
        ----------
        judgment : JudgmentRow
            Accepted scheduled judgment with joined drawing ids.

        Returns
        -------
        RescoredPair
            Judgment joined to fresh TYPE-M scene values.

        Raises
        ------
        ValueError
            If the two sides resolve under different artifact versions.
        """

        side_a = self.score_drawing(judgment.blind_id_a, judgment)
        side_b = self.score_drawing(judgment.blind_id_b, judgment)
        if (
            side_a.measurement_version != side_b.measurement_version
            or side_a.policy_version != side_b.policy_version
        ):
            raise ValueError("pair sides were scored under different artifact versions")
        return RescoredPair(judgment=judgment, side_a=side_a, side_b=side_b)

    def score_rows(self, judgments: Iterable[JudgmentRow]) -> Tuple[RescoredPair, ...]:
        """Re-score a stable sequence of bank judgments.

        Parameters
        ----------
        judgments : iterable[JudgmentRow]
            Accepted scheduled judgments.

        Returns
        -------
        tuple[RescoredPair, ...]
            Pairs in input order.
        """

        return tuple(self.score_pair(judgment) for judgment in judgments)
