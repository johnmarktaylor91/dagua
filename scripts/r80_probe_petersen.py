"""End-to-end referee-divergence probe for petersen_10 (r80-S2b task).

Instruments the undirected-portfolio contest during a REAL benchmark-style
dagua engine run and decomposes the difference between what the contest
referee scores and what the benchmark scores:

(a) contest-internal score for every candidate (incumbent + challengers,
    plus offline with/without-convergent cleanup variants);
(b) benchmark composite of the winner's positions AS SELECTED and of the
    engine's ACTUAL final returned positions;
(c) per-term breakdown of every step where the numbers diverge.

Decomposition logic:
- contest_score(winner) vs benchmark_score(winner-as-selected): scoring
  FRAME divergence (node sizes / direction flavor / cluster ids / tier).
- benchmark_score(winner-as-selected) vs benchmark_score(final returned):
  POST-SELECTION transformation of positions after the referee scored.

Usage: python scripts/r80_probe_petersen.py [graph_name]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r79_baseline import SEED, TIMEOUT_SECONDS, build_corpus, get_competitor

from dagua.eval.graphs import is_semantically_directed
from dagua.layout.ops.pipelines import native_undirected as nu
from dagua.metrics import composite_auto, evaluate

TERMS = [
    "overlap_count",
    "crossing_rate",
    "edge_length_cv",
    "sampled_stress",
    "angular_res_mean_deg",
    "cluster_mean_sep_ratio",
    "dag_consistency",
    "depth_spearman_rho",
    "edge_straightness_mean_deg",
    "aspect_ratio",
    "bbox_width",
    "bbox_height",
]

records: dict = {"scores": [], "project": [], "portfolio": []}

_orig_score = nu._score_undirected_candidate
_orig_project = nu._project_candidate
_orig_portfolio = nu.layout_native_undirected_portfolio


def spy_score(pos, problem, cluster_ids):
    """Record every contest scoring call (order: incumbent, then challengers)."""
    score = _orig_score(pos, problem, cluster_ids)
    records["scores"].append(
        {"score": float(score), "pos": pos.detach().clone(), "cluster_ids": cluster_ids}
    )
    return score


def spy_project(pos, problem, convergent=False):
    """Record challenger cleanup inputs/outputs and the problem tensors."""
    out = _orig_project(pos, problem, convergent=convergent)
    records["project"].append(
        {
            "raw": pos.detach().clone(),
            "projected": out.detach().clone(),
            "problem": problem,
            "convergent": bool(convergent),
        }
    )
    return out


def spy_portfolio(problem, state, ctx, config):
    """Record the portfolio route's returned (selected) positions."""
    out = _orig_portfolio(problem, state, ctx, config)
    records["portfolio"].append({"problem": problem, "selected": out.detach().clone()})
    return out


nu._score_undirected_candidate = spy_score
nu._project_candidate = spy_project
nu.layout_native_undirected_portfolio = spy_portfolio


def bench_metrics(graph, pos):
    """Benchmark-identical metrics for positions (r79_baseline.run_engine)."""
    return evaluate(graph, pos.detach().cpu().to(dtype=torch.float32), tier="full")


def bench_score(graph, pos, directed: bool) -> float:
    """Benchmark-identical composite for positions."""
    return float(composite_auto(bench_metrics(graph, pos), directed))


def term_table(label_a: str, ma: dict, label_b: str, mb: dict) -> None:
    """Print a per-term comparison of two metric dicts."""
    print(f"    {'term':28s} {label_a:>12s} {label_b:>12s} {'delta':>10s}")
    for key in TERMS:
        va, vb = ma.get(key), mb.get(key)
        if va is None and vb is None:
            continue
        va = float("nan") if va is None else float(va)
        vb = float("nan") if vb is None else float(vb)
        print(f"    {key:28s} {va:12.4f} {vb:12.4f} {vb - va:+10.4f}")


def main() -> int:
    """Run the instrumented probe.

    Returns
    -------
    int
        Process exit status.
    """
    name = sys.argv[1] if len(sys.argv) > 1 else "petersen_10"
    corpus = {g.name: g for g in build_corpus()}
    tg = corpus[name]
    graph = tg.graph
    directed = is_semantically_directed(tg)
    print(
        f"=== {name}: n={graph.num_nodes} e={int(graph.edge_index.shape[1])} "
        f"benchmark_directed_flavor={directed} ==="
    )

    competitor = get_competitor("dagua")
    result = competitor.layout(graph, timeout=TIMEOUT_SECONDS, seed=SEED)
    final_pos = result.pos.detach().cpu().to(dtype=torch.float32)

    print(
        f"\ncontest scoring calls: {len(records['scores'])} | "
        f"challenger cleanups: {len(records['project'])} | "
        f"portfolio invocations: {len(records['portfolio'])}"
    )

    if not records["portfolio"]:
        print("PORTFOLIO ROUTE NEVER RAN -- divergence is upstream of the contest.")
        print(
            f"benchmark composite of final returned positions: "
            f"{bench_score(graph, final_pos, directed):.3f}"
        )
        return 0

    problem = records["portfolio"][-1]["problem"]
    selected = records["portfolio"][-1]["selected"]

    # (frame checks)
    print("\nframe checks:")
    print(f"  problem.direction={problem.direction!r} graph.direction={graph.direction!r}")
    gns = graph.node_sizes
    pns = problem.node_sizes
    same_sizes = (
        gns is not None
        and pns is not None
        and gns.shape == pns.shape
        and torch.allclose(gns.to(torch.float32), pns.to(torch.float32))
    )
    print(f"  node_sizes identical problem-vs-graph: {same_sizes}")
    if not same_sizes and gns is not None and pns is not None:
        print(
            f"    graph sizes mean {gns.float().mean(0)} | problem sizes mean {pns.float().mean(0)}"
        )

    # (a) contest-internal candidate scores, labeled by call order
    labels = ["incumbent"]
    labels += [f"challenger_{i}" for i in range(1, len(records["scores"]))]
    print("\n(a) contest-internal scores (call order):")
    winner_idx = 0
    for i, rec in enumerate(records["scores"]):
        marker = ""
        if torch.allclose(rec["pos"].to(torch.float32), selected.to(torch.float32), atol=1e-4):
            marker = "  <-- SELECTED"
            winner_idx = i
        print(f"  {labels[i]:14s} contest_score={rec['score']:8.3f}{marker}")

    # offline with/without-convergent variants of each challenger cleanup
    from dagua.layout.projection import project_overlaps

    if records["project"] and problem.node_sizes is not None:
        print("\n(a2) challenger cleanup variants (contest frame scores):")
        cluster_ids = records["scores"][0]["cluster_ids"]
        node_sizes = problem.node_sizes.to(dtype=torch.float32)
        for j, prec in enumerate(records["project"]):
            raw = prec["raw"].detach().clone().to(torch.float32)
            legacy = raw.detach().clone()
            project_overlaps(legacy, node_sizes)  # default legacy path
            conv = raw.detach().clone()
            project_overlaps(conv, node_sizes, iterations=200, convergent=True)
            s_raw = _orig_score(raw, problem, cluster_ids)
            s_leg = _orig_score(legacy, problem, cluster_ids)
            s_conv = _orig_score(conv, problem, cluster_ids)
            b_leg = bench_score(graph, legacy, directed)
            b_conv = bench_score(graph, conv, directed)
            print(
                f"  challenger[{j}] raw={s_raw:7.3f} | legacy-clean contest={s_leg:7.3f} "
                f"bench={b_leg:7.3f} | convergent-clean contest={s_conv:7.3f} bench={b_conv:7.3f}"
            )

    # (b) benchmark composites
    winner_pos = records["scores"][winner_idx]["pos"]
    contest_winner_score = records["scores"][winner_idx]["score"]
    bench_winner = bench_score(graph, winner_pos, directed)
    bench_final = bench_score(graph, final_pos, directed)
    print("\n(b) composites:")
    print(f"  contest score of winner (referee's number):     {contest_winner_score:8.3f}")
    print(f"  benchmark composite of winner-as-selected:      {bench_winner:8.3f}")
    print(f"  benchmark composite of FINAL returned positions:{bench_final:8.3f}")

    same_pos = winner_pos.shape == final_pos.shape and torch.allclose(
        winner_pos.to(torch.float32), final_pos, atol=1e-4
    )
    print(f"  final positions identical to winner-as-selected: {same_pos}")
    if not same_pos and winner_pos.shape == final_pos.shape:
        # similarity-transform check: compare normalized pairwise distances
        def _pdist_unit(p):
            d = torch.cdist(p, p)
            m = d[d > 0]
            return d / m.mean() if m.numel() else d

        pd_sel = _pdist_unit(winner_pos.to(torch.float32))
        pd_fin = _pdist_unit(final_pos)
        shape_dev = float((pd_sel - pd_fin).abs().max().item())
        print(
            f"  max normalized pairwise-distance deviation (0 = pure similarity "
            f"transform): {shape_dev:.6f}"
        )
        scale = float(
            torch.cdist(final_pos, final_pos).max()
            / max(
                float(
                    torch.cdist(winner_pos.to(torch.float32), winner_pos.to(torch.float32)).max()
                ),
                1e-9,
            )
        )
        print(f"  bbox scale factor selected->final: {scale:.4f}")

    # (c) per-term breakdowns of each divergence step
    print(
        "\n(c1) frame divergence: contest full() vs benchmark evaluate() on the "
        "SAME winner positions:"
    )
    contest_metrics = {
        k: float(v)
        for k, v in __import__("dagua.metrics", fromlist=["full"])
        .full(
            winner_pos.to(torch.float32),
            problem.edge_index.cpu(),
            node_sizes=None if problem.node_sizes is None else problem.node_sizes.float().cpu(),
            cluster_ids=records["scores"][0]["cluster_ids"],
            direction=problem.direction,
        )
        .items()
        if isinstance(v, (int, float))
    }
    bench_m_winner = bench_metrics(graph, winner_pos)
    term_table("contest", contest_metrics, "benchmark", bench_m_winner)

    print(
        "\n(c2) post-selection divergence: benchmark evaluate() on winner-as-"
        "selected vs FINAL returned positions:"
    )
    bench_m_final = bench_metrics(graph, final_pos)
    term_table("winner", bench_m_winner, "final", bench_m_final)

    print("\nsummary:")
    print(
        f"  frame gap (contest score vs bench score, same positions): "
        f"{bench_winner - contest_winner_score:+.3f}"
    )
    print(
        f"  post-selection gap (bench winner vs bench final):          "
        f"{bench_final - bench_winner:+.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
