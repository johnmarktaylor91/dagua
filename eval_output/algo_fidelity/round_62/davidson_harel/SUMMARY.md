# Round 62 Davidson-Harel Real Port Summary

## Source Lines Ported

- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:40-78`:
  segment intersection and point-to-segment distance helpers.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:149-166`:
  node/edge counts, square bounds, move radius, 30 proposal directions, and energy weights.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:198-237`:
  seeded coordinate handling and circular proposal direction initialization.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:239-253`:
  per-round vertex shuffle and per-vertex proposal shuffle.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:259-420`:
  local move delta for node distribution, edge length, crossings, and fine-tuning node-edge
  distance.
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c:422-442`:
  Boltzmann acceptance and geometric temperature decay.
- `/home/jtaylor/projects/_references/igraph/src/layout/align.c:107-123` and
  `/home/jtaylor/projects/_references/igraph/src/layout/align.c:133-301`:
  post-layout centering, nematic tensor alignment, eigenvector rotation, and axis ordering.

## Dagua Implementation

- `dagua/layout/ops/pipelines/davidson_harel.py:34-371` now contains the pure-Python support
  port: igraph-style bounded RNG/shuffle, geometry helpers, default weight resolution, and
  `igraph_layout_align` equivalent.
- `dagua/layout/ops/pipelines/davidson_harel.py:374-589` now runs the sequential local-delta
  Davidson-Harel annealing/fine-tuning loop without importing or invoking python-igraph.
- `dagua/layout/ops/pipelines/davidson_harel.py:746-755` routes fidelity mode to the pure port.

## Delegation Check

Confirmed empty:

```text
git diff dagua/layout/ops/pipelines/davidson_harel.py | grep -E "^\\+.*(import igraph|from igraph)"
```

Also checked both forbidden files with:

```text
rg -n "import igraph|from igraph|layout\\(\"davidson_harel|subprocess\\.run" \
  dagua/layout/ops/pipelines/davidson_harel.py dagua/layout/ops/davidson_harel.py
```

No matches.

## Smoke RMSD

Before, from the round-41 smoke summary, the delegated-after path was exact but the local
pre-delegation implementation had mean Procrustes RMSD `0.374099`.

Round-62 pure-port smoke:

```text
python eval_output/algo_fidelity/round_62/davidson_harel/smoke_harness.py
max_rmsd=2.26871059776e-16
max_abs=964.246219957
```

The RMSD is reflection-tolerant Procrustes RMSD, matching the existing round-41 smoke method.
Raw max-abs differences remain for some seeds because LAPACK eigenvector signs in
`igraph_layout_align` are sign-ambiguous; the annealing trajectory and aligned geometry match
up to axis reflection.

## Final Verdict

Pass for the R62 completeness target: no runtime delegation remains, and the pure-Python port
matches the igraph Davidson-Harel adapter under the smoke threshold (`max RMSD < 1e-6`).
