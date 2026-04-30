# Round 12 Blocked -- davidson_harel live baseline

Status: BLOCKED
Family: davidson_harel
Date: 2026-04-29

## Blocker

The required live multi-seed baseline did not complete inside the prompt's
10-minute baseline budget:

```bash
python scripts/algo_fidelity_live_compare.py classic_davidson_harel igraph_davidson_harel \
    --seeds 5 --output-dir eval_output/algo_fidelity/round_12/baseline
```

Observed runtime was just over 10 minutes with the Python process still
CPU-active and no files written under
`eval_output/algo_fidelity/round_12/baseline`. The process was stopped and no
algorithm code was changed.

This triggers the round's missing-context gate: "live_compare for
davidson_harel times out (graph too slow)".

## Source findings before abort

I verified the igraph reference exists at:
`/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c`.

Highest-confidence divergences found while waiting for the baseline:

1. Energy defaults and scaling differ.
   - igraph uses unnormalized terms with weights
     `{node_dist=1.0, border=0.0, edge_lengths=0.0001,
     edge_crossings=1.0, node_edge_dist=0.2}` at
     `davidson_harel.c:161-166`.
   - dagua uses normalized terms and weights
     `{border=0.1, edge_lengths=0.2, edge_crossings=2.0,
     node_edge_dist=0.5}` in
     `dagua/layout/ops/davidson_harel.py:16-20` and
     `dagua/layout/ops/davidson_harel.py:180-193`.

2. Move schedule differs.
   - igraph initializes `move_radius = width / 2` where
     `width = sqrt(no_nodes) * 10`, shuffles nodes each round, and tries
     30 circular directions per node at
     `davidson_harel.c:151-162`, `davidson_harel.c:233-255`,
     `davidson_harel.c:262-263`, and `davidson_harel.c:422-423`.
   - dagua currently attempts one random square move per node per round,
     with radius derived from energy-scaled temperature at
     `dagua/layout/ops/davidson_harel.py:353-380`.

Note: the prompt said igraph "picks the best move out of `no_tries`", but the
source shows it tests 30 shuffled directions and accepts each move if downhill
or by the simulated-annealing probability.

## Next recommended unblock

Reduce the live comparison scope before applying code changes. Options:

- Add a graph/count filter to `scripts/algo_fidelity_live_compare.py` and run a
  5-seed baseline on the smallest Davidson-Harel comparator graphs first.
- Run `--seeds 2` as a triage-only timing probe, then rerun `--seeds 5` on a
  smaller graph subset.
- If the round is allowed to touch comparison infrastructure, add progress
  flushing so long-running families produce partial CSV/JSON output before the
  final aggregate.

Once the baseline is available, the first focused lever should be the move
schedule alignment: 30 circular tries per node with `move_radius = extent` and
acceptance `exp(-delta_energy / move_radius)`, paired with igraph's unnormalized
energy weights.
