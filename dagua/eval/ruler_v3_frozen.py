"""Frozen V3 ruler constants and provenance records.

This module is the Part-A freeze manifest for the scoring-only V3 ruler. It is
intended as the source for FINAL_RULER.md publication tables and as a stable
regression reference for pure-function gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FrozenConstant:
    """Publication record for one frozen V3 ruler constant.

    Parameters
    ----------
    name : str
        Stable manifest key.
    value : Any
        Frozen value or structured frozen value.
    source : str
        Code location where the value currently lives.
    note : str
        Brief provenance or usage note.
    """

    name: str
    value: Any
    source: str
    note: str


FROZEN_CONSTANTS: Dict[str, FrozenConstant] = {
    "tier_multipliers": FrozenConstant(
        name="tier_multipliers",
        value={"T1": 4.0, "T2": 2.0, "T3": 1.0},
        source="dagua/eval/ruler_v3.py:TIER_*_WEIGHT; dagua/eval/ruler_v3_groups.py:TIER_*_WEIGHT",
        note="Evidence-tier headline weighting, renormalized over applicable facets.",
    ),
    "c2_size_decay": FrozenConstant(
        name="c2_size_decay",
        value={"N_mid": 700.0, "s": 0.2, "floor_multiplier": 0.5},
        source="dagua/eval/ruler_v3.py:crossing_weight_multiplier",
        note="Applied to C2 effective aggregate weight only, not to raw crossing score.",
    ),
    "c2_angle_weighted_crossing_cost": FrozenConstant(
        name="c2_angle_weighted_crossing_cost",
        value={
            "min": 0.5,
            "max": 1.5,
            "best_angle_degrees": 90.0,
            "total_refinement_step_fraction": 0.49,
        },
        source="dagua/eval/ruler_v3.py:angle_weighted_crossing_score",
        note=(
            "V3 C2' count-dominant crossing score; angle readability is a bounded-total "
            "within-count refinement and C6 is diagnostic."
        ),
    ),
    "c3_graph_distance_radii": FrozenConstant(
        name="c3_graph_distance_radii",
        value=(1, 2, 3),
        source="dagua/eval/ruler_v3.py:score_core_v3 -> multi_radius_neighborhood_preservation",
        note="Frozen multi-k neighborhood-preservation radii.",
    ),
    "c5_primary_form": FrozenConstant(
        name="c5_primary_form",
        value="structure_floor_area_band",
        source="dagua/eval/ruler_v3.py:whitespace_sprawl_score",
        note="Scored C5 form selected by F5; fallback stays diagnostic metadata.",
    ),
    "c5_band_and_decay": FrozenConstant(
        name="c5_band_and_decay",
        value={
            "content_multiplier": 2.0,
            "structure_separation": 3.0,
            "ratio_lo": 1.0,
            "ratio_hi": 16.0,
            "crowding_decay": 0.5,
            "sprawl_decay": 1.25,
            "sprawl_ratio_factor": 4.0,
        },
        source="dagua/eval/ruler_v3.py:WHITESPACE_* and SPRAWL_RATIO_FACTOR",
        note="Asymmetric C5 band over visual area divided by frozen structure/content floors.",
    ),
    "c5_fallback_edge_length_band": FrozenConstant(
        name="c5_fallback_edge_length_band",
        value={"ratio_lo": 0.75, "ratio_hi": 16.0},
        source="dagua/eval/ruler_v3.py:EDGE_LENGTH_RATIO_*",
        note="Published diagnostic fallback form: mean edge length divided by mean node diagonal.",
    ),
    "c6_crossing_angle_ideal_degrees": FrozenConstant(
        name="c6_crossing_angle_ideal_degrees",
        value=90.0,
        source="dagua/eval/ruler_v3.py:crossing_angle_90_score",
        note="Diagnostic-only crossing-angle ideal; zero crossings publish vacuous 1.0.",
    ),
    "diagnostic_core_weights": FrozenConstant(
        name="diagnostic_core_weights",
        value={"C2": 6.0, "C6": 0.0, "C8": 0.0},
        source="dagua/eval/ruler_v3.py:CORE_DIAGNOSTIC_WEIGHTS/CORE_WEIGHT_OVERRIDES",
        note="C6 folded into C2' weight and formula; C8 retained only as diagnostic.",
    ),
    "c4_smooth_clearance_band": FrozenConstant(
        name="c4_smooth_clearance_band",
        value={"clearance_band_node_diagonals": 0.5, "label_inclusive_when_supplied": True},
        source="dagua/eval/ruler_v3.py:_smooth_clearance_occlusion_score",
        note=(
            "Smooth C4 clearance debt below half a mean node diagonal, "
            "label-inclusive when rendered."
        ),
    ),
    "overlap_severity_saturation": FrozenConstant(
        name="overlap_severity_saturation",
        value=0.25,
        source="dagua/eval/ruler_v3.py:OVERLAP_SEVERITY_SATURATION",
        note="Co-signed overlap-severity saturation point for the V3 headline fold.",
    ),
    "overlap_packing_fill_gate": FrozenConstant(
        name="overlap_packing_fill_gate",
        value={"fill_lo": 0.50, "fill_hi": 0.75},
        source="dagua/eval/ruler_v3.py:OVERLAP_PACKING_FILL_*",
        note="Co-signed packed-seam fill gate for the V3 headline overlap fold.",
    ),
    "canonical_node_height_ref": FrozenConstant(
        name="canonical_node_height_ref",
        value=34.0,
        source="dagua/eval/ruler_v3_groups.py:CANONICAL_NODE_HEIGHT_REF",
        note="Intrinsic-ruler canonicalization reference from default short-label node height.",
    ),
    "default_node_box": FrozenConstant(
        name="default_node_box",
        value={"width": 1.0, "height": 1.0},
        source="dagua/eval/ruler_v3.py:DEFAULT_NODE_WIDTH/DEFAULT_NODE_HEIGHT",
        note="Fallback CORE node box when explicit node sizes are omitted.",
    ),
    "degenerate_scale_and_occlusion_flags": FrozenConstant(
        name="degenerate_scale_and_occlusion_flags",
        value={"degenerate_scale_ratio": 0.25, "occlusion_floor_threshold": 0.75},
        source="dagua/eval/ruler_v3.py:DEGENERATE_SCALE_RATIO/OCCLUSION_FLOOR_THRESHOLD",
        note="Frozen row-flag thresholds for scale degeneracy and C4 floor.",
    ),
    "sprawl_collapse_c4_max": FrozenConstant(
        name="sprawl_collapse_c4_max",
        value=1.0,
        source="dagua/eval/ruler_v3.py:SPRAWL_COLLAPSE_C4_MAX",
        note="SPRAWL_COLLAPSE fires when the SPRAWL flag is set and C4 is below this boundary.",
    ),
    "coincident_collapse_radius": FrozenConstant(
        name="coincident_collapse_radius",
        value=0.25,
        source="dagua/eval/ruler_v3.py:COINCIDENT_COLLAPSE_RADIUS",
        note="Nearest-neighbor radius as a fraction of mean node diagonal for cf.",
    ),
    "coincident_collapse_fraction": FrozenConstant(
        name="coincident_collapse_fraction",
        value=0.30,
        source="dagua/eval/ruler_v3.py:COINCIDENT_COLLAPSE_FRACTION",
        note="Champion-eligibility threshold for COINCIDENT_COLLAPSE row flags.",
    ),
    "deterministic_seed_default": FrozenConstant(
        name="deterministic_seed_default",
        value=0,
        source="dagua/eval/ruler_v3.py:score_core_v3(seed=0)",
        note="Default seed for sampled V2-compatible primitives.",
    ),
    "core_sampling_budgets": FrozenConstant(
        name="core_sampling_budgets",
        value={
            "stress_sources": 200,
            "stress_targets": 1000,
            "crossing_samples": 1_000_000,
            "neighborhood_samples": 5000,
        },
        source="dagua/eval/ruler_v3.py:score_core_v3 defaults",
        note="Frozen pure-function budgets for core sampled facets.",
    ),
    "g2_slot_weights": FrozenConstant(
        name="g2_slot_weights",
        value={
            "slot_a_total": 7.0,
            "slot_b_total": 3.0,
            "slot_a_split": (2, 2, 1, 1, 1),
            "slot_b_split": (2, 1),
        },
        source="dagua/eval/ruler_v3_groups.py:G2_SLOT_* and GROUP_REGISTRY['G2']",
        note="Declared-cluster group internal slot split.",
    ),
    "hac_rule": FrozenConstant(
        name="hac_rule",
        value={
            "linkage": "average",
            "metric": "euclidean",
            "cluster_count": "number_of_unique_declared_labels",
        },
        source="dagua/eval/ruler_v3_groups.py:_score_g2/_score_g3/_hac_labels",
        note="Frozen geometric HAC rule for declared-cluster and planted-community ARI.",
    ),
    "g2_compactness_band": FrozenConstant(
        name="g2_compactness_band",
        value={
            "plateau_hi": 8.0,
            "log_decay": 1.0,
            "legible_spacing_guard": 2.0,
            "legible_spacing_eps": 1e-12,
        },
        source="dagua/eval/ruler_v3_groups.py:COMPACTNESS_*",
        note="Cluster compactness log-ratio band with no credit below node-diagonal spacing.",
    ),
    "g4_slot_weights_and_extent_band": FrozenConstant(
        name="g4_slot_weights_and_extent_band",
        value={
            "slot_a_total": 5.0,
            "slot_b_total": 5.0,
            "extent_ratio_lo": 0.5,
            "extent_ratio_hi": 2.0,
            "extent_log_decay": 1.0,
        },
        source="dagua/eval/ruler_v3_groups.py:G4_SLOT_* and TREE_RATIO_*",
        note="Rooted-tree layered/radial slot split and extent-ratio band.",
    ),
    "g5_quality_band": FrozenConstant(
        name="g5_quality_band",
        value={"delta": 0.05, "change_epsilon": 1e-9, "zero_change_scale": 0.05},
        source="dagua/eval/ruler_v3_groups.py:G5_*",
        note="Temporal quality-band and zero-change constants.",
    ),
    "g6_weighted_budgets_and_cv": FrozenConstant(
        name="g6_weighted_budgets_and_cv",
        value={
            "weight_cv_saturation": 1.0,
            "nondegenerate_weight_cv": 1e-9,
            "local_node_budget": 512,
            "weighted_stress_sources": 200,
            "weighted_stress_targets": 1000,
            "severe_floor": 0.55,
            "severe_floor_drop": 0.05,
            "soft_floor": 0.60,
            "severe_weight_multiplier": 2.0,
            "severe_breach_facets": ("G6_weighted_ksm", "G6_local_weight_monotonicity"),
        },
        source=(
            "dagua/eval/ruler_v3_groups.py:WEIGHT_CV_*, LOCAL_WEIGHT_*, G6_WEIGHTED_*; "
            "dagua/eval/ruler_v3.py:SEVERE_G6_FACETS/G6_FLOOR_DROP"
        ),
        note=(
            "Weighted-group graded gate, deterministic budgets, severe-breach weight ramp, "
            "and shared severe-G6 contract surface."
        ),
    ),
    "severe_g6_breach_flag": FrozenConstant(
        name="severe_g6_breach_flag",
        value="severe_g6_breach",
        source="dagua/eval/ruler_v3.py:SEVERE_G6_BREACH_FLAG",
        note="Score-neutral row flag for the absolute severe declared-weight semantics contract.",
    ),
    "gg3_buyback_gate": FrozenConstant(
        name="gg3_buyback_gate",
        value={"tier1_materiality": 5.0, "buyback_bar": 1.0, "hold_band_fraction": 0.02},
        source=(
            "scripts/ceremony_sa_attack.py:TIER1_ONLY_MATERIALITY_DROP/TWO_LAYOUT_BUYBACK_BAR; "
            "dagua/eval/ruler_v3.py:severe_g6_floor_breach"
        ),
        note=(
            "BLOCK requires material T1 drop plus two-layout buyback, "
            "or an independent severe G6 breach."
        ),
    ),
    "margin_audit_tripwire": FrozenConstant(
        name="margin_audit_tripwire",
        value={
            "headline": "tiered_linear",
            "family_envelope": "adjudication_instrument_only",
            "fallback": 12.2209,
            "softmin_tau": 1.0,
            "de_ramped_tier1_instrument": True,
            "audit_column": "tiered_capped",
            "hold_instrument_column": "tiered_hold_instrument",
            "A_f": {
                "weighted": 8.5915,
                "clustered": 9.3004,
                "dag": 7.7785,
                "generic_force": 10.6493,
                "tree": 12.2476,
                "ported": 2.8026,
            },
        },
        source=(
            "dagua/eval/ruler_v3.py:FAMILY_MARGIN_ALLOWANCES/FAMILY_SOFTMIN_TAU; "
            "scripts/ceremony_sa_attack.py:_load_margin_envelopes"
        ),
        note=(
            "Corpus headline tiered is the uncapped tiered_linear score. The "
            "clustered A=9.3004, tau=1.0 softmin is retained only for the "
            "GG-3/E1 hold adjudication instrument and audit publication; "
            "condition-5 aggregate deltas are computed on that capped view."
        ),
    ),
    "g7_port_and_route_thresholds": FrozenConstant(
        name="g7_port_and_route_thresholds",
        value={
            "severe_port_cap": 0.5,
            "side_cosine_threshold": 2**-0.5,
            "route_sample_canonical_distance": 1.0,
            "terminal_separation_fraction": 0.5,
        },
        source="dagua/eval/ruler_v3_groups.py:G7_*",
        note="Port severe-cap and route/terminal thresholds.",
    ),
    "frac_acyclic_grading": FrozenConstant(
        name="frac_acyclic_grading",
        value="effective_weight *= fraction_of_hierarchy_edges_after_declared_feedback_removal",
        source="dagua/eval/ruler_v3_groups.py:_score_g1",
        note="G1 directed-flow/depth-order graded applicability rule.",
    ),
    "freeze_probe_thresholds": FrozenConstant(
        name="freeze_probe_thresholds",
        value={
            "unit_alphas": (0.02, 0.1, 1.0, 10.0, 50.0),
            "position_scale_alphas": (0.02, 0.1, 1.0, 10.0, 50.0),
            "continuous_relative_tolerance": 1e-12,
            "crater_10x_max_score": 0.5,
            "crater_50x_max_score": 0.05,
            "legit_sprawl_min_score": 0.99,
        },
        source="tests/test_ruler_v3_freeze_invariance.py",
        note="Permanent GG-5/F5 pure-function acceptance thresholds.",
    ),
}
