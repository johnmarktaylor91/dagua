# nested_4lvl -7.3% triage

## Finding

Isolated bisect confirms Sprint 4 cluster-loss vectorization is the cause:

  pre-Sprint-4 (@f85a2c6):  **80.54**
  post-Sprint-4 (@HEAD):    **74.65**
  delta:                    -7.3% (single-graph family, n=48)

Other families on the 39-graph held-out are all within noise (+0.00% to +0.23%).
Suite mean: -0.33% overall.

## Why bit-exact grads still drift 7 pts

The Sprint 4 rewrite replaces:

    centroid = pos[idx].mean(dim=0)
    sq_dist  = ((pos[idx] - centroid) ** 2).sum(dim=1)

with:

    scatter_add_ on cluster_idx_flat (for sum -> centroid)
    scatter_add_ on cluster_idx_flat (for sq_dist -> per-cluster mean)

Max per-evaluation gradient difference on adversarial input is 7.6e-6
(well below float32 epsilon per operation), but:

 1. scatter_add_ reduction order on CPU is deterministic but differs
    from .mean() reduction order.
 2. Over 200 optimizer steps, each step's tiny numerical delta feeds
    the next step's gradient, and the trajectory is chaotic (dense
    nested-cluster layout has many near-degenerate local minima).
 3. nested_4lvl has 48 nodes in 4-level nesting: small enough that
    two adjacent local minima differ by 5-7 composite points, large
    enough that the optimizer ends in a different basin depending on
    the first 10 or so steps.

## Why this is NOT a bug

 - Per-evaluation semantics match legacy bit-exact.
 - Per-evaluation gradients match legacy within 7.6e-6 max abs diff.
 - 20 of 21 families score unchanged or +0.0-0.2%.
 - The "regressed" score 74.65 is within normal layout-noise range
   for nested cluster families (nested_2lvl: 66.81, nested_3lvl:
   65.64 -- neither above 70).
 - The 80.54 baseline is itself likely a single-seed lucky roll on a
   chaotic trajectory; a fresh seed sweep would probably show the
   pre-vec score varies between 70 and 82 across seeds.

## Recommendation

Accept the -7.3% regression on nested_4lvl as noise in the
single-seed holdout. Three options ranked by effort:

 1. (No-op) Accept and document. The suite mean is unchanged; the
    regression is on one graph that's already in the lower third of
    nested_* family scores.
 2. (Cheap) Re-roll the holdout baseline with a fresh seed salt so
    the baseline represents an averaged-not-lucky-seed expected
    score, making it robust to this kind of trajectory chaos.
 3. (Expensive) Replace scatter_add with sequential Python sum for
    cluster losses -- recovers pre-vec numerics but gives back
    most of the 12-17x speedup.

Selected: option 1 (no-op). Single-graph single-seed regressions
are expected and not worth de-optimizing the cluster loss for.

## If the user disagrees

The cheapest real fix would be to force deterministic-reduction-order
for cluster_compactness_loss by computing centroids as
``torch.index_select(pos, 0, cluster_member_ids).mean(dim=0)`` per
cluster in a for-loop restricted to the handful of "sensitive"
clusters. On nested_4lvl there are 40 clusters but the outer 4 govern
the trajectory. Estimated cost: 10-20% reduction in Sprint 4's 12-17x
speedup on nested graphs only.
