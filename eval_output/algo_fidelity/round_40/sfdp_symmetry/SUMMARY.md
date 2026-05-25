# Round 40 SFDP Symmetric Star Residual

## Root Cause

The remaining star-topology raw Procrustes residual is not caused by a Dagua
versus Graphviz node-iteration divergence in the smoke harness.

Trace findings:

- `eval_output/algo_fidelity/round_39/sfdp_rng/smoke_check.py` builds the star
  in insertion order: `n0`, then `n1..n7`, with edges `n0 -> n1..n7`.
- `dagua.graphviz_utils.to_dot()` serializes explicit node statements in
  `range(graph.num_nodes)` order before serializing edges in stored edge order.
- Graphviz 7.0.5 `makeMatrix()` assigns `ND_id` by `agfstnode()/agnxtnode()`.
  For this DOT input, Graphviz JSON reports `_gvid` and `name` as
  `(0, n0), (1, n1), ..., (7, n7)`.
- Dagua's SFDP Graphviz-fidelity path builds the base graph in `range(N)` row
  order and initializes coarsest positions by consuming `GraphvizRandom.drand()`
  into rows `0..N-1`.
- The star graph does not coarsen in the current fidelity path:
  Graphviz-style supervariable grouping would produce only three coarse nodes,
  below `SFDPHierarchyConfig.min_coarse_size=4`, so the random-position row
  order is the original node order.

The raw residual is therefore an automorphism-label residual on symmetric leaf
nodes. Graphviz and Dagua produce the same star geometry, but the force path can
select different labels for indistinguishable leaf slots. This is consistent
with the Round 39 diagnosis: best leaf-permuted RMSD is below `0.002`, while raw
RMSD is about `0.165` when one symmetric pair is swapped.

## Fix Applied

No code fix was applied.

I tested two plausible Graphviz-fidelity implementation mismatches and reverted
both because they did not close the residual:

- Two-pass force/move update order in `dagua/layout/ops/pipelines/sfdp.py`:
  star stayed near `0.165` and one path seed regressed to `0.245809560`.
- Always using the Graphviz quadtree path for tiny graphs:
  one star seed improved, but seed 1 stayed `0.171281814` and seed 2 rose to
  `0.015513220`.

Because the verified node order already matches, a targeted code change would
need to canonicalize star automorphisms or change the fidelity metric to use
symmetry-aware matching. That is outside this task's allowed SFDP ordering fix
unless a separate metric/canonicalization spec is approved.

## Smoke RMSD

Command:

```bash
python eval_output/algo_fidelity/round_39/sfdp_rng/smoke_check.py
```

Before and after honest investigation are unchanged:

| Topology | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: |
| path | 0.023626077 | 0.019741366 | 0.013578818 |
| star | 0.165420600 | 0.002054337 | 0.164348903 |
| clustered | 0.000179138 | 0.052798253 | 0.000205787 |

## Verdict

Unresolved as a raw-label metric issue.

The geometry is at the numerical floor under leaf automorphism matching, but raw
RMSD is not bit-exact because symmetric leaves are label-indistinguishable under
the SFDP objective. The next appropriate change is a separate fidelity-analysis
update that either computes a star-leaf automorphism RMSD or adds a general
Hungarian-matched diagnostic alongside raw Procrustes RMSD.
