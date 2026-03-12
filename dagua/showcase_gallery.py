"""User-facing showcase gallery generator for Dagua.

Builds a curated folder of stills and motion assets spanning:
- real-world industries and workflows
- representative graph structures
- aesthetically strong default renders
- reproducible motion exports (optimization + tour)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

import dagua
from dagua import DaguaGraph, LayoutConfig
from dagua.animation import AnimationConfig, PosterConfig, TourConfig
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle, PALETTE, make_fill, border_from_fill


@dataclass
class GalleryEntry:
    slug: str
    title: str
    industry: str
    use_case: str
    structure_tags: Sequence[str]
    visual_story: str
    build_graph: Callable[[], DaguaGraph]
    direction: str = "TB"
    scene: str = "auto"
    node_sep: float = 28.0
    rank_sep: float = 52.0
    steps: int = 110
    edge_opt_steps: int = 18


@dataclass
class GalleryAnimation:
    slug: str
    title: str
    kind: str
    caption: str
    build_graph: Callable[[], DaguaGraph]
    direction: str = "TB"
    steps: int = 70
    edge_opt_steps: int = 14
    scene: str = "auto"


@dataclass
class GalleryBuildResult:
    output_dir: str
    manifest_path: str
    readme_path: str
    still_paths: List[str] = field(default_factory=list)
    animation_paths: List[str] = field(default_factory=list)


def _node(base: str, shape: str = "roundrect") -> NodeStyle:
    fill = make_fill(base, blend=0.24)
    return NodeStyle(base_color=base, fill=fill, stroke=border_from_fill(base, darken=0.35), shape=shape)


def _edge(base: str = "#64748B", routing: str = "bezier") -> EdgeStyle:
    return EdgeStyle(color=base, routing=routing, curvature=0.36, opacity=0.68)


def _cluster(base: str) -> ClusterStyle:
    fill = make_fill(base, blend=0.14)
    return ClusterStyle(fill=fill, stroke=border_from_fill(base, darken=0.28))


def _add_nodes(graph: DaguaGraph, items: Sequence[tuple[str, str, str, Optional[NodeStyle]]]) -> None:
    for node_id, label, node_type, style in items:
        graph.add_node(node_id, label=label, type=node_type, style=style)


def _add_edges(graph: DaguaGraph, items: Sequence[tuple[str, str, Optional[str], Optional[EdgeStyle]]]) -> None:
    for src, dst, label, style in items:
        graph.add_edge(src, dst, label=label, style=style)


def _hospital_care_pathway() -> DaguaGraph:
    g = DaguaGraph(direction="LR")
    clinical = _node(PALETTE["blue"])
    diagnostics = _node(PALETTE["amber"], shape="ellipse")
    decision = _node(PALETTE["vermillion"])
    _add_nodes(
        g,
        [
            ("arrival", "Arrival", "clinical", clinical),
            ("triage", "Triage", "clinical", clinical),
            ("labs", "Labs", "diagnostic", diagnostics),
            ("imaging", "Imaging", "diagnostic", diagnostics),
            ("consult", "Specialist Review", "clinical", clinical),
            ("diagnosis", "Working Diagnosis", "decision", decision),
            ("careplan", "Care Plan", "clinical", clinical),
            ("followup", "Follow-up", "clinical", clinical),
        ],
    )
    _add_edges(
        g,
        [
            ("arrival", "triage", None, None),
            ("triage", "labs", "order", None),
            ("triage", "imaging", "order", None),
            ("triage", "consult", None, None),
            ("labs", "diagnosis", "results", _edge(PALETTE["amber"])),
            ("imaging", "diagnosis", "results", _edge(PALETTE["amber"])),
            ("consult", "diagnosis", "assessment", None),
            ("diagnosis", "careplan", None, _edge(PALETTE["vermillion"])),
            ("careplan", "followup", None, None),
        ],
    )
    idx = g._id_to_index
    g.add_cluster("diagnostics", [idx["labs"], idx["imaging"]], label="Diagnostics", style=_cluster(PALETTE["amber"]))
    g.add_cluster("treatment", [idx["diagnosis"], idx["careplan"], idx["followup"]], label="Treatment", style=_cluster(PALETTE["blue"]))
    return g


def _fraud_decision_engine() -> DaguaGraph:
    g = DaguaGraph(direction="TB")
    ingest = _node(PALETTE["sky"])
    risk = _node(PALETTE["reddish_purple"], shape="ellipse")
    action = _node(PALETTE["vermillion"])
    _add_nodes(
        g,
        [
            ("txn", "Incoming Transaction", "ingest", ingest),
            ("profile", "Customer Profile", "ingest", ingest),
            ("geo", "Geo Signal", "risk", risk),
            ("device", "Device Signal", "risk", risk),
            ("merchant", "Merchant Signal", "risk", risk),
            ("score", "Risk Score", "decision", action),
            ("approve", "Approve", "outcome", _node(PALETTE["bluish_green"])),
            ("review", "Manual Review", "outcome", _node(PALETTE["amber"])),
            ("block", "Block", "outcome", _node(PALETTE["vermillion"])),
        ],
    )
    _add_edges(
        g,
        [
            ("txn", "geo", None, None),
            ("txn", "device", None, None),
            ("txn", "merchant", None, None),
            ("profile", "geo", "history", None),
            ("profile", "device", "history", None),
            ("profile", "merchant", "history", None),
            ("geo", "score", None, None),
            ("device", "score", None, None),
            ("merchant", "score", None, None),
            ("score", "approve", "< low", _edge(PALETTE["bluish_green"])),
            ("score", "review", "middle", _edge(PALETTE["amber"])),
            ("score", "block", "> high", _edge(PALETTE["vermillion"])),
        ],
    )
    idx = g._id_to_index
    g.add_cluster("signals", [idx["geo"], idx["device"], idx["merchant"]], label="Real-time signals", style=_cluster(PALETTE["reddish_purple"]))
    return g


def _supply_chain_control_tower() -> DaguaGraph:
    g = DaguaGraph(direction="LR")
    plan = _node(PALETTE["blue"])
    ops = _node(PALETTE["sky"])
    shipment = _node(PALETTE["amber"], shape="ellipse")
    alert = _node(PALETTE["vermillion"])
    _add_nodes(
        g,
        [
            ("forecast", "Demand Forecast", "plan", plan),
            ("buy", "Procurement", "plan", plan),
            ("plant_w", "West Plant", "ops", ops),
            ("plant_e", "East Plant", "ops", ops),
            ("port_la", "LA Port", "shipment", shipment),
            ("port_sav", "Savannah Port", "shipment", shipment),
            ("dc_w", "West DC", "ops", ops),
            ("dc_c", "Central DC", "ops", ops),
            ("dc_e", "East DC", "ops", ops),
            ("stores", "Retail Stores", "ops", _node(PALETTE["bluish_green"])),
            ("alert", "Expedite / Rebalance", "alert", alert),
        ],
    )
    _add_edges(
        g,
        [
            ("forecast", "buy", None, None),
            ("buy", "plant_w", None, None),
            ("buy", "plant_e", None, None),
            ("plant_w", "port_la", None, None),
            ("plant_e", "port_sav", None, None),
            ("port_la", "dc_w", None, None),
            ("port_la", "dc_c", "overflow", None),
            ("port_sav", "dc_c", None, None),
            ("port_sav", "dc_e", None, None),
            ("dc_w", "stores", None, None),
            ("dc_c", "stores", None, None),
            ("dc_e", "stores", None, None),
            ("forecast", "alert", "surge", _edge(PALETTE["vermillion"])),
            ("alert", "dc_c", "re-route", _edge(PALETTE["vermillion"])),
            ("alert", "dc_e", "re-route", _edge(PALETTE["vermillion"])),
        ],
    )
    idx = g._id_to_index
    g.add_cluster("manufacturing", [idx["plant_w"], idx["plant_e"]], label="Manufacturing", style=_cluster(PALETTE["blue"]))
    g.add_cluster("distribution", [idx["dc_w"], idx["dc_c"], idx["dc_e"]], label="Distribution", style=_cluster(PALETTE["amber"]))
    return g


def _ml_platform_release() -> DaguaGraph:
    g = DaguaGraph(direction="TB")
    data = _node(PALETTE["sky"])
    model = _node(PALETTE["blue"])
    evaluate = _node(PALETTE["amber"], shape="ellipse")
    prod = _node(PALETTE["bluish_green"])
    _add_nodes(
        g,
        [
            ("raw", "Raw Data", "data", data),
            ("features", "Feature Store", "data", data),
            ("train", "Training Job", "model", model),
            ("validate", "Validation", "eval", evaluate),
            ("bias", "Bias Audit", "eval", evaluate),
            ("latency", "Latency Test", "eval", evaluate),
            ("registry", "Model Registry", "model", model),
            ("deploy", "Canary Deploy", "prod", prod),
            ("monitor", "Monitoring", "prod", prod),
            ("rollback", "Rollback Path", "prod", _node(PALETTE["vermillion"])),
        ],
    )
    _add_edges(
        g,
        [
            ("raw", "features", None, None),
            ("features", "train", None, None),
            ("train", "validate", None, None),
            ("train", "bias", None, None),
            ("train", "latency", None, None),
            ("validate", "registry", None, None),
            ("bias", "registry", None, None),
            ("latency", "registry", None, None),
            ("registry", "deploy", None, None),
            ("deploy", "monitor", None, None),
            ("monitor", "rollback", "degrade", _edge(PALETTE["vermillion"])),
            ("monitor", "features", "drift", _edge(PALETTE["amber"])),
        ],
    )
    idx = g._id_to_index
    g.add_cluster("evaluation", [idx["validate"], idx["bias"], idx["latency"]], label="Release gates", style=_cluster(PALETTE["amber"]))
    g.add_cluster("production", [idx["deploy"], idx["monitor"], idx["rollback"]], label="Production loop", style=_cluster(PALETTE["bluish_green"]))
    return g


def _robotics_autonomy_stack() -> DaguaGraph:
    g = DaguaGraph(direction="LR")
    sense = _node(PALETTE["sky"], shape="ellipse")
    fuse = _node(PALETTE["blue"])
    plan = _node(PALETTE["amber"])
    control = _node(PALETTE["bluish_green"])
    _add_nodes(
        g,
        [
            ("camera", "Camera", "sense", sense),
            ("lidar", "LiDAR", "sense", sense),
            ("imu", "IMU", "sense", sense),
            ("fusion", "Sensor Fusion", "core", fuse),
            ("perception", "Perception", "core", fuse),
            ("planner", "Trajectory Planner", "plan", plan),
            ("controller", "Controller", "control", control),
            ("actuators", "Actuators", "control", control),
            ("telemetry", "Telemetry", "control", _node(PALETTE["reddish_purple"])),
        ],
    )
    _add_edges(
        g,
        [
            ("camera", "fusion", None, None),
            ("lidar", "fusion", None, None),
            ("imu", "fusion", None, None),
            ("fusion", "perception", None, None),
            ("perception", "planner", None, None),
            ("planner", "controller", None, None),
            ("controller", "actuators", None, None),
            ("actuators", "telemetry", "state", None),
            ("telemetry", "planner", "feedback", _edge(PALETTE["reddish_purple"])),
        ],
    )
    idx = g._id_to_index
    g.add_cluster("sensors", [idx["camera"], idx["lidar"], idx["imu"]], label="Sensors", style=_cluster(PALETTE["sky"]))
    return g


def _drug_discovery_program() -> DaguaGraph:
    g = DaguaGraph(direction="TB")
    science = _node(PALETTE["blue"])
    assay = _node(PALETTE["amber"], shape="ellipse")
    gate = _node(PALETTE["vermillion"])
    _add_nodes(
        g,
        [
            ("target", "Target Selection", "science", science),
            ("screen", "High-throughput Screen", "science", science),
            ("potency", "Potency Assay", "assay", assay),
            ("selectivity", "Selectivity Assay", "assay", assay),
            ("adme", "ADME Panel", "assay", assay),
            ("hits", "Lead Series", "decision", gate),
            ("tox", "Toxicology", "assay", assay),
            ("pk", "PK Study", "assay", assay),
            ("nominate", "Candidate Nomination", "decision", _node(PALETTE["bluish_green"])),
        ],
    )
    _add_edges(
        g,
        [
            ("target", "screen", None, None),
            ("screen", "potency", None, None),
            ("screen", "selectivity", None, None),
            ("screen", "adme", None, None),
            ("potency", "hits", None, None),
            ("selectivity", "hits", None, None),
            ("adme", "hits", None, None),
            ("hits", "tox", None, None),
            ("hits", "pk", None, None),
            ("tox", "nominate", None, None),
            ("pk", "nominate", None, None),
        ],
    )
    idx = g._id_to_index
    g.add_cluster("screening", [idx["potency"], idx["selectivity"], idx["adme"]], label="Screening filters", style=_cluster(PALETTE["amber"]))
    g.add_cluster("preclinical", [idx["tox"], idx["pk"]], label="Preclinical package", style=_cluster(PALETTE["blue"]))
    return g


def _customer_support_escalation() -> DaguaGraph:
    g = DaguaGraph(direction="LR")
    inbound = _node(PALETTE["sky"])
    triage = _node(PALETTE["amber"])
    specialist = _node(PALETTE["blue"])
    outcome = _node(PALETTE["bluish_green"])
    _add_nodes(
        g,
        [
            ("ticket", "New Ticket", "inbound", inbound),
            ("classify", "Auto-classify", "triage", triage),
            ("billing", "Billing Queue", "specialist", specialist),
            ("product", "Product Queue", "specialist", specialist),
            ("infra", "Infra Queue", "specialist", specialist),
            ("manager", "Escalation Manager", "specialist", _node(PALETTE["vermillion"])),
            ("resolve", "Resolution", "outcome", outcome),
            ("kb", "Knowledge Base Update", "outcome", _node(PALETTE["reddish_purple"])),
        ],
    )
    _add_edges(
        g,
        [
            ("ticket", "classify", None, None),
            ("classify", "billing", None, None),
            ("classify", "product", None, None),
            ("classify", "infra", None, None),
            ("billing", "resolve", None, None),
            ("product", "resolve", None, None),
            ("infra", "resolve", None, None),
            ("product", "manager", "stuck", _edge(PALETTE["vermillion"])),
            ("manager", "resolve", None, _edge(PALETTE["vermillion"])),
            ("resolve", "kb", "learn", _edge(PALETTE["reddish_purple"])),
            ("kb", "classify", "better routing", _edge(PALETTE["reddish_purple"])),
        ],
    )
    return g


def _multimodal_assistant_system() -> DaguaGraph:
    g = DaguaGraph(direction="TB")
    ingress = _node(PALETTE["sky"], shape="ellipse")
    modality = _node(PALETTE["blue"])
    core = _node(PALETTE["amber"])
    output = _node(PALETTE["bluish_green"])
    _add_nodes(
        g,
        [
            ("user", "User Request", "in", ingress),
            ("text", "Text Encoder", "modality", modality),
            ("image", "Vision Encoder", "modality", modality),
            ("tools", "Tool Router", "modality", modality),
            ("planner", "Planner", "core", core),
            ("memory", "Working Memory", "core", core),
            ("reason", "Reasoning Core", "core", core),
            ("response", "Response Draft", "out", output),
            ("safety", "Safety Check", "out", output),
            ("final", "Final Answer", "out", output),
        ],
    )
    _add_edges(
        g,
        [
            ("user", "text", None, None),
            ("user", "image", None, None),
            ("user", "tools", None, None),
            ("text", "planner", None, None),
            ("image", "planner", None, None),
            ("tools", "planner", None, None),
            ("planner", "memory", None, None),
            ("planner", "reason", None, None),
            ("memory", "reason", "context", _edge(PALETTE["reddish_purple"])),
            ("reason", "response", None, None),
            ("planner", "response", "plan skip", _edge(PALETTE["amber"])),
            ("response", "safety", None, None),
            ("safety", "final", None, None),
        ],
    )
    idx = g._id_to_index
    g.add_cluster("modalities", [idx["text"], idx["image"], idx["tools"]], label="Input modalities", style=_cluster(PALETTE["sky"]))
    g.add_cluster("core", [idx["planner"], idx["memory"], idx["reason"]], label="Core reasoning", style=_cluster(PALETTE["amber"]))
    return g


def _gallery_entries() -> List[GalleryEntry]:
    return [
        GalleryEntry(
            slug="hospital_care_pathway",
            title="Hospital Care Pathway",
            industry="Healthcare",
            use_case="Show a care team how intake, diagnostics, diagnosis, and follow-up fit together.",
            structure_tags=["parallel diagnostics", "clusters", "decision funnel"],
            visual_story="Parallel diagnostic branches stay legible while treatment remains the visual destination.",
            build_graph=_hospital_care_pathway,
            direction="LR",
            scene="zoom_pan",
        ),
        GalleryEntry(
            slug="fraud_decision_engine",
            title="Fraud Decision Engine",
            industry="Finance",
            use_case="Explain real-time risk scoring and manual review paths to operations or compliance teams.",
            structure_tags=["fan-in", "branching outcomes", "signal cluster"],
            visual_story="A dense signal fan-in resolves into three crisp operational outcomes.",
            build_graph=_fraud_decision_engine,
            direction="TB",
            scene="auto",
        ),
        GalleryEntry(
            slug="supply_chain_control_tower",
            title="Supply Chain Control Tower",
            industry="Operations",
            use_case="Visualize how demand, plants, ports, warehouses, and reroute decisions interact.",
            structure_tags=["wide logistics network", "clusters", "long-span reroute edges"],
            visual_story="The wide physical network stays readable while exception paths remain obvious.",
            build_graph=_supply_chain_control_tower,
            direction="LR",
            scene="panorama",
            node_sep=34,
            rank_sep=58,
        ),
        GalleryEntry(
            slug="ml_platform_release",
            title="ML Platform Release Loop",
            industry="Software / MLOps",
            use_case="Show the path from data to gated release, canary deployment, monitoring, and rollback.",
            structure_tags=["release gates", "feedback loop", "nested subsystems"],
            visual_story="Evaluation gates read as a real release barrier rather than a tangle of side tasks.",
            build_graph=_ml_platform_release,
            direction="TB",
            scene="layer_sweep",
        ),
        GalleryEntry(
            slug="robotics_autonomy_stack",
            title="Robotics Autonomy Stack",
            industry="Robotics",
            use_case="Communicate the handoff from sensing through planning to control, with telemetry feedback.",
            structure_tags=["sensor fan-in", "control loop", "clustered sources"],
            visual_story="Parallel sensing feels coherent and the feedback edge reads as a control loop, not noise.",
            build_graph=_robotics_autonomy_stack,
            direction="LR",
            scene="zoom_pan",
        ),
        GalleryEntry(
            slug="drug_discovery_program",
            title="Drug Discovery Program",
            industry="Biotech / Pharma",
            use_case="Present screening, assay filters, and preclinical gates in one clean view.",
            structure_tags=["funnel", "parallel assays", "gated progression"],
            visual_story="The figure reads like a program funnel, with side assays supporting a single decision spine.",
            build_graph=_drug_discovery_program,
            direction="TB",
            scene="auto",
        ),
        GalleryEntry(
            slug="customer_support_escalation",
            title="Customer Support Escalation",
            industry="Customer Operations",
            use_case="Show routing, specialist queues, escalations, and knowledge capture for service teams.",
            structure_tags=["branching queues", "feedback", "operational loop"],
            visual_story="Escalation and learning loops are visible without overpowering the main service flow.",
            build_graph=_customer_support_escalation,
            direction="LR",
            scene="cathedral",
        ),
        GalleryEntry(
            slug="multimodal_assistant_system",
            title="Multimodal Assistant System",
            industry="AI Systems",
            use_case="Explain how modalities, planning, memory, reasoning, and safety compose into a product.",
            structure_tags=["parallel modalities", "skip path", "clustered core"],
            visual_story="Modern model architecture motifs feel polished rather than diagrammatically clichéd.",
            build_graph=_multimodal_assistant_system,
            direction="TB",
            scene="motif_orbit",
        ),
    ]


def _gallery_animations() -> List[GalleryAnimation]:
    return [
        GalleryAnimation(
            slug="optimize_multimodal_assistant",
            title="Optimization Story: Multimodal Assistant",
            kind="optimization",
            caption="A faithful post-hoc optimization film showing the graph settle into a readable hierarchy.",
            build_graph=_multimodal_assistant_system,
            direction="TB",
            steps=60,
            edge_opt_steps=10,
        ),
        GalleryAnimation(
            slug="tour_supply_chain_control_tower",
            title="Tour: Supply Chain Control Tower",
            kind="tour",
            caption="A cinematic sweep from network context into the most operationally interesting logistics regions.",
            build_graph=_supply_chain_control_tower,
            direction="LR",
            scene="zoom_pan",
            steps=80,
            edge_opt_steps=12,
        ),
    ]


def _layout_config(entry_steps: int, entry_edge_steps: int, direction: str, node_sep: float, rank_sep: float) -> LayoutConfig:
    return LayoutConfig(
        steps=entry_steps,
        edge_opt_steps=entry_edge_steps,
        direction=direction,
        node_sep=node_sep,
        rank_sep=rank_sep,
        seed=42,
    )


def _write_index(path: Path, manifest: Dict) -> None:
    lines = [
        "# Dagua Showcase Gallery",
        "",
        "Autogenerated examples spanning industries, graph structures, and cinematic exports.",
        "",
        "## Stills",
        "",
    ]
    for entry in manifest["stills"]:
        lines.extend(
            [
                f"### {entry['title']}",
                "",
                f"- Industry: {entry['industry']}",
                f"- Use case: {entry['use_case']}",
                f"- Structure tags: {', '.join(entry['structure_tags'])}",
                f"- Visual story: {entry['visual_story']}",
                "",
                f"![{entry['title']}](stills/{Path(entry['path']).name})",
                "",
            ]
        )
    if manifest["animations"]:
        lines.extend(["## Animations", ""])
        for entry in manifest["animations"]:
            lines.extend(
                [
                    f"### {entry['title']}",
                    "",
                    f"- Kind: {entry['kind']}",
                    f"- Caption: {entry['caption']}",
                    "",
                    f"![{entry['title']}](animations/{Path(entry['path']).name})",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_showcase_gallery(
    output_dir: str = "docs/gallery",
    include_animations: bool = True,
    limit: Optional[int] = None,
    sample_steps: Optional[int] = None,
    animation_steps: Optional[int] = None,
) -> GalleryBuildResult:
    out = Path(output_dir)
    stills_dir = out / "stills"
    anim_dir = out / "animations"
    stills_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)

    entries = _gallery_entries()
    animations = _gallery_animations()
    if limit is not None:
        entries = entries[:limit]
        animations = animations[: max(1, min(limit, len(animations)))]

    still_manifest: List[Dict[str, object]] = []
    animation_manifest: List[Dict[str, object]] = []

    for entry in entries:
        graph = entry.build_graph()
        config = _layout_config(
            sample_steps or entry.steps,
            entry.edge_opt_steps,
            entry.direction,
            entry.node_sep,
            entry.rank_sep,
        )
        positions = dagua.layout(graph, config)
        output_path = stills_dir / f"{entry.slug}.png"
        dagua.poster(
            graph,
            positions=positions,
            config=config,
            output=str(output_path),
            poster_config=PosterConfig(
                scene=entry.scene,
                dpi=220,
                show_titles=False,
            ),
        )
        still_manifest.append(
            {
                "slug": entry.slug,
                "title": entry.title,
                "industry": entry.industry,
                "use_case": entry.use_case,
                "structure_tags": list(entry.structure_tags),
                "visual_story": entry.visual_story,
                "path": str(output_path),
            }
        )

    if include_animations:
        for entry in animations:
            graph = entry.build_graph()
            config = _layout_config(
                animation_steps or entry.steps,
                entry.edge_opt_steps,
                entry.direction,
                30.0,
                54.0,
            )
            output_path = anim_dir / f"{entry.slug}.gif"
            if entry.kind == "optimization":
                dagua.animate(
                    graph,
                    config=config,
                    output=str(output_path),
                    animation_config=AnimationConfig(
                        fps=16,
                        dpi=110,
                        max_layout_frames=24,
                        max_edge_frames=12,
                    ),
                )
            else:
                positions = dagua.layout(graph, config)
                dagua.tour(
                    graph,
                    positions=positions,
                    config=config,
                    output=str(output_path),
                    tour_config=TourConfig(
                        scene=entry.scene,
                        fps=18,
                        dpi=120,
                        hold_start_frames=6,
                        hold_end_frames=8,
                    ),
                )
            animation_manifest.append(
                {
                    "slug": entry.slug,
                    "title": entry.title,
                    "kind": entry.kind,
                    "caption": entry.caption,
                    "path": str(output_path),
                }
            )

    manifest = {
        "stills": still_manifest,
        "animations": animation_manifest,
    }
    manifest_path = out / "gallery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme_path = out / "README.md"
    _write_index(readme_path, manifest)

    return GalleryBuildResult(
        output_dir=str(out),
        manifest_path=str(manifest_path),
        readme_path=str(readme_path),
        still_paths=[entry["path"] for entry in still_manifest],
        animation_paths=[entry["path"] for entry in animation_manifest],
    )
