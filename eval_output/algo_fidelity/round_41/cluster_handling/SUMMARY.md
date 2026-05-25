# R41 Cluster Handling: Sugiyama + dagua_native

## Wired

- `layout_sugiyama_pipeline` now accepts `clusters` and `cluster_parents` and runs
  the existing `ClusterAwareDriver` with a Sugiyama inner pipeline when complete
  cluster metadata is present. This gives clustered DAG stages recursive
  cluster-as-node placement instead of flat placement.
- `layout_dagua_native_pipeline` now routes explicit
  `force_pipeline="force_directed"` clustered runs through `ClusterAwareDriver`
  using the native force-directed inner pipeline.

## Verification

- Added a Sugiyama smoke test for sequential clustered DAG stages; cluster A's
  y-range remains before cluster B's y-range.
- Added a dagua_native force-directed smoke test asserting sibling cluster node
  boxes are separated.

## Deferred

- Sugiyama cluster edge-route and trace outputs are rejected for the
  cluster-aware path. The recursive driver only returns final original-node
  positions today.
- The native default topology dispatcher still uses its existing clustered
  layered/tree/hybrid behavior unless callers explicitly select
  `force_pipeline="force_directed"`. This task's native force-directed gap is
  covered without changing sibling R41 engine files.
