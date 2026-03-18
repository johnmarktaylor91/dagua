# 1B-Node Scaling Bug Catalog

Comprehensive record of every bug encountered while scaling dagua's layout engine
to 1 billion nodes. These bugs span memory management, algorithmic complexity,
checkpoint persistence, and code path coverage. Together they form a field guide
for anyone working on large-scale graph layout -- or any system that crosses the
threshold from "fits comfortably in memory" to "every allocation matters."

## Summary Table

| # | Category | Bug | Severity | Root Cause (one line) |
|---|----------|-----|----------|----------------------|
| 1 | Algorithmic | Redundant SampledAccessPattern per term per step | Medium | No sharing mechanism for sampled terms |
| 2 | Code Path | Overlap projection on step 0 | Medium | `step % N == 0` true at step 0 |
| 3 | Algorithmic | Spacing loss as global access at 200M+ | High | Global access pattern runs on full graph in subset_gpu mode |
| 4 | Algorithmic | classify_graph computing degree at 1B | High | No fast path for extreme scale |
| 5 | Algorithmic | classify_graph computing layering on CPU | High | Hardcoded device="cpu", redundant computation |
| 6 | Algorithmic | CSR build O(E log E) at 1.5B edges | Critical | Comparison sort where counting sort suffices |
| 7 | Checkpoint | Source fingerprint invalidating checkpoints | Medium | Fingerprint includes actively-edited source |
| 8 | Checkpoint | Incomplete hierarchy marked complete | Medium | No partial-result handling |
| 9 | Memory | Hierarchy checkpoint loading 62GB at once | Critical | Sequential torch.load with no lazy loading |
| 10 | Memory | Dead references after "offload" | Critical | del local alias != free object |
| 11 | Memory | glibc malloc not returning pages to OS | Critical | glibc free list vs OS page return |
| 12 | Code Path | Original graph not offloaded in restore path | High | Optimization in one branch, not both |
| 13 | Algorithmic | GPU layering O(waves x E) for deep graphs | High | Algorithm designed for wide graphs, not deep |
| 14 | Code Path | Streaming threshold ignoring edge count | High | Threshold checked N only, not E |
| 15 | Memory | Temp file offload filling /tmp | High | tempfile defaults to root partition |
| 16 | Checkpoint | fine_layer_assignments not reloaded from disk | High | Reload function not updated when checkpoint schema grew |
| 17 | Memory | Duplicate references wasting 12GB during build | Medium | Graph object holds references alongside local aliases |

---

## Category 1: Memory Management

Bugs where the system ran out of memory or wasted memory due to reference
management, allocation strategy, or OS-level behavior.

---

### Bug 9: Hierarchy checkpoint loading all 62GB into RAM at once

**What happened.** `_load_hierarchy_checkpoint` loaded all 3 hierarchy level
files (28 + 19 + 16 = 62GB) into RAM simultaneously via sequential `torch.load`
in a loop with no lazy loading.

**Root cause.** The checkpoint loader was written for small graphs where loading
everything is fine. At 1B scale, the refinement loop only needs one level at a
time, but the loader materialized all of them upfront.

**Fix applied.** Lazy loading: only keep `fine_to_coarse` mapping for
non-coarsest levels. Point `offload_path` at checkpoint files so levels can be
reloaded on demand during refinement.

**Architectural lesson.** Checkpoint loaders for large data should load lazily.
Design for the access pattern: if consumers process levels sequentially, don't
load them all at once. The "load everything into a list" pattern that works at
small scale becomes a memory bomb at large scale.

---

### Bug 10: Graph object holding dead references after offload

**What happened.** Code did `del cpu_ei, cpu_ns` to "free" edge index (24GB)
and node sizes after writing them to disk. Memory was not freed. The graph
object's `_edge_index_tensor` attribute still held the reference.

**Root cause.** Python reference counting. `del x` only decrements the refcount
of the object `x` points to. If another reference exists (e.g., on the graph
object), the memory stays alive. The "offload" wrote to disk without actually
freeing anything.

**Fix applied.** Null `graph._edge_index_tensor` and `graph.node_sizes` after
saving to disk, not just the local aliases.

**Architectural lesson.** When "offloading" tensors to disk, you must null ALL
references, not just local variables. In Python, `del x` is not `free(x)`. This
is the single most common memory bug in Python programs that manage large
objects. Audit every reference path, not just the one in front of you.

---

### Bug 11: glibc malloc not returning freed pages to OS

**What happened.** After loading a 28GB checkpoint file and then freeing it,
`gc.collect()` ran, Python released the objects, but RSS did not drop. glibc
kept the freed pages in its internal free list. Subsequent allocations pushed
RSS past 125GB and the system ran out of memory.

**Root cause.** glibc's `malloc` uses `mmap` for very large allocations (which
ARE returned to the OS on free) but uses `brk`/`sbrk` for medium-sized
allocations. The latter are managed in a free list and never returned to the OS
unless `malloc_trim` is called explicitly. PyTorch tensor allocations can fall
into this category depending on the allocator path.

**Fix applied.** After every large deallocation:
```python
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
```

**Architectural lesson.** On Linux with large allocations, Python's garbage
collector alone is not enough to reclaim memory. You must explicitly call
`malloc_trim(0)` to return freed pages to the OS. This is invisible at small
scale (a few hundred MB of fragmentation doesn't matter) but fatal at 100GB+
scale where every GB counts. Any Python program operating near the memory
ceiling on Linux needs this.

---

### Bug 15: Temp file offload filling /tmp

**What happened.** Hierarchy offload wrote 62GB of level data to `/tmp`, which
only had ~76GB of space on the root partition. Combined with other temp files
from the session, this filled the disk.

**Root cause.** `tempfile.mkdtemp()` defaults to `/tmp`, which on most Linux
systems lives on the root partition. The code didn't consider that 62GB of temp
data might not fit.

**Fix applied.** Use `/mnt/locker` (6TB spinning disk) for temp offload when
available, with fallback to `/tmp`.

**Architectural lesson.** Large temp files should go to the largest available
filesystem, not `/tmp`. Any offload/checkpoint system should accept a configurable
temp directory and default to a location with ample space. Check `df` before
writing tens of GB.

---

### Bug 17: Graph object duplicate references wasting 12GB during fresh hierarchy build

**What happened.** At 1.5B nodes, `graph._precomputed_layer_assignments` (6GB),
`graph.node_sizes` (6GB), and `graph._edge_index_tensor` (36GB) stayed alive
on the graph object while `build_hierarchy` used the same data via independent
local aliases. Both the graph object references and the local aliases kept the
tensors alive, doubling memory usage for the overlapping period.

**Root cause.** Same fundamental issue as Bug 10: Python objects hold references
until explicitly cleared. The graph object is a convenient carrier for data, but
when that data is also captured in local variables for processing, neither `del`
on the locals nor letting them go out of scope will free the data if the graph
object still holds its own reference.

**Fix applied.** Null graph object references (`graph._precomputed_layer_assignments
= None`, etc.) before calling `build_hierarchy`, after capturing the data in local
aliases.

**Architectural lesson.** At extreme scale, "convenient extra references" cost
gigabytes. The pattern of keeping data on an object "just in case" while also
passing it to functions creates invisible duplication. Either the object owns the
data or local code does -- not both simultaneously.

---

## Category 2: Algorithmic Complexity

Bugs where the algorithm's time or space complexity was inappropriate for the
input size. These are the bugs where small-scale testing gives no signal because
the cost only becomes apparent at 10^8+ elements.

---

### Bug 1: Redundant SampledAccessPattern creation per term per step

**What happened.** Each sampled loss term (repel, overlap) created its own
`SampledAccessPattern` with an independent `torch.unique` call on ~9M elements.
With multiple sampled terms, this redundant work added up.

**Root cause.** Edge-type terms had `SharedEdgeSubsetData` for sharing computed
subsets across terms, but sampled terms had no equivalent sharing mechanism. Each
term independently computed its own access pattern from scratch.

**Fix applied.** Added `SharedSampledSubsetData` with cross-step caching of the
pattern when the sampled context (`sampled_ctx`) hasn't changed.

**Architectural lesson.** When you add a sharing optimization for one term type,
apply the same pattern to ALL term types. Optimization asymmetry between
structurally-similar code paths is a code smell. The edge terms and sampled terms
had the same access pattern structure but different sharing behavior -- a sign
that the abstraction was incomplete.

---

### Bug 3: Spacing loss as global access pattern at 200M+ scale

**What happened.** `spacing_consistency_loss` ran on the full 200M-node tensor
every step via `GlobalAccessPattern`. It performed `argsort` per layer, and with
`layers=0` (all nodes in one layer), it argsorted 200M elements.

**Root cause.** `access_kind="global"` means the `subset_gpu` executor runs the
loss on the FULL graph, not a sampled subset. The `is_heavy=True` flag existed
on the term but wasn't used as a gate to skip it in subset_gpu mode.

**Fix applied.** Skip heavy global terms (`is_heavy=True`) in subset_gpu mode
for N > 50M.

**Architectural lesson.** In subset_gpu mode, ANY global access pattern is
potentially catastrophic. The "heavy" flag was metadata without enforcement --
a classic case of marking something as dangerous without actually preventing the
danger. Flags that indicate "this is expensive" should be enforced as gates, not
advisory annotations.

---

### Bug 4: classify_graph computing degree on 1B nodes

**What happened.** `_compute_degree` allocated 20GB+ of `ones` tensors for
1.5B edges just to determine the graph family, which is always `GENERAL` at
1B nodes (no billion-node graph is a "chain" or "tree").

**Root cause.** `classify_graph` ran the full analysis regardless of scale.
The classification is useful for choosing layout strategies at small scale but
produces an obvious answer at extreme scale.

**Fix applied.** Early return `GENERAL` for N > 10M.

**Architectural lesson.** Classification and analysis functions should have fast
paths for extreme scale where the answer is trivially known. Don't compute a
detailed answer when a simple heuristic gives the same result. This applies
broadly: any function that analyzes input characteristics to make a decision
should check if the decision is already obvious before doing expensive analysis.

---

### Bug 5: classify_graph computing longest_path_layering on CPU

**What happened.** `_resolve_layer_assignments` called `longest_path_layering`
with `device="cpu"`, taking 45+ minutes at 1B nodes. A CUDA path existed but
wasn't used. Worse, the computation was redundant -- the multilevel pipeline
computes its own layering downstream.

**Root cause.** Hardcoded `device="cpu"` in the classifier. The function was
written before GPU layering existed and never updated.

**Fix applied.** Skipped entirely for N > 10M (covered by the early-return fix
from Bug 4).

**Architectural lesson.** Don't recompute what downstream code computes anyway.
The classifier and the multilevel pipeline both independently computed layering
-- a DRY violation that cost 45 minutes at scale. When adding a fast path to
one consumer, check if other consumers are doing the same work.

---

### Bug 6: CSR build using O(E log E) argsort at 1.5B edges

**What happened.** `_build_csr_numpy` used `np.argsort(kind='stable')` on 1.5B
elements to sort edges by source node for CSR construction. Stable mergesort is
O(E log E) with a large constant factor. At 1.5B edges: 30-60 minutes.

**Root cause.** Comparison-based sort used where the keys are integers in a
known range `[0, N)` -- exactly the case where counting sort gives O(E).

**Fix applied.** Numba JIT counting sort for O(N + E) construction. Fallback to
unstable quicksort (lower constant factor than mergesort) with int32 keys.

**Architectural lesson.** At billion-element scale, the difference between O(E)
and O(E log E) is ~30x (log2(1.5B) ~ 30.5). Comparison sort is never the right
answer for integer keys in a known range. This is textbook but easy to overlook
when the code was written for 10K-element graphs where both complete in
milliseconds. Use counting sort or radix sort for integer keys.

---

### Bug 13: GPU layering O(waves x E) for deep graphs

**What happened.** `_gpu_longest_path_layering` scans ALL edges per wavefront
iteration without CSR adjacency. At 1.14M nodes with 31K layers and 60M edges:
31K waves x 60M edges = 1.9 trillion element accesses and 186K CUDA kernel
launches.

**Root cause.** The GPU layering algorithm was designed for wide graphs (few
layers, many nodes per layer) where the parallelism benefit outweighs the
redundant work. For deep graphs (many layers, few nodes per layer), the CPU CSR
path is O(N + E) -- linear, no redundancy.

**Fix applied.** Skip GPU layering when `E/N > 10` (heuristic indicating a deep
graph where the CPU CSR path is faster).

**Architectural lesson.** GPU is not always faster than CPU. When the algorithm
complexity differs -- O(waves * E) on GPU vs O(N + E) on CPU -- the "slow"
device with the better algorithm wins. GPU parallelism only helps when the
algorithm is the same and the per-element work parallelizes. A heuristic gate
that selects the right algorithm for the graph shape is essential.

---

## Category 3: Checkpoint and Persistence

Bugs in the checkpoint save/load system that caused wasted computation,
incorrect state restoration, or cascading failures.

---

### Bug 7: Source fingerprint invalidating checkpoints on every code edit

**What happened.** `bench_large.py` hashes source files (engine.py,
multilevel.py, etc.) to detect stale checkpoints. During active development,
every edit to these files invalidated the hierarchy checkpoint, forcing a
multi-hour rebuild.

**Root cause.** The fingerprint includes the full source of files being actively
edited. It can't distinguish between "changed the coarsening algorithm" (stale
checkpoint) and "added a log line" (checkpoint still valid).

**Fix applied.** Temporarily disabled fingerprint check during development.

**Architectural lesson.** Checkpoint validation should separate "data-affecting
changes" from "all code changes." A content hash of the full source is too
aggressive -- it creates a tension between iterative development and checkpoint
reuse. Better approaches: version the checkpoint format explicitly, or hash only
the specific functions/parameters that affect the output.

---

### Bug 8: Incomplete hierarchy marked as complete

**What happened.** Manually marked a 3-level hierarchy (coarsest level at 37M
nodes) as "complete" to bypass checkpoint validation, but 37M is far above
`min_nodes=2000`. The hierarchy was incomplete.

**Root cause.** User/developer error driven by frustration with checkpoint
invalidation (Bug 7). The checkpoint system was all-or-nothing: either the
hierarchy was "complete" or it was rebuilt from scratch.

**Fix applied.** Accept incomplete hierarchies and continue coarsening from the
last completed level.

**Architectural lesson.** Checkpoint loaders should handle partial results
gracefully. "Resume from where we left off" is always better than "start over."
When a process has natural stages (hierarchy levels), each stage should be
independently checkpointed and resumable.

---

### Bug 16: fine_layer_assignments not loaded on demand during refinement

**What happened.** `_reload_level_from_disk` only loaded `edge_index` and
`node_sizes`, not `fine_layer_assignments`. Each refinement level recomputed
layering from scratch, costing 10-15 minutes per level.

**Root cause.** The reload function was written when the offload system only
saved `edge_index` and `node_sizes`. When the checkpoint format was extended to
include `fine_layer_assignments`, the reload function was not updated.

**Fix applied.** Also load `fine_layer_assignments` in `_reload_level_from_disk`
when present in the checkpoint file.

**Architectural lesson.** When extending what a checkpoint saves, update ALL
consumers that read from it. The save path and load path must evolve together.
This is a specific instance of the general rule: when you add a field to a
serialization format, grep for every deserialization site and update it.

---

## Category 4: Code Path Coverage

Bugs where an optimization, guard, or fix was applied to one code path but not
to a parallel or alternative path that needed the same treatment.

---

### Bug 2: Overlap projection running on step 0

**What happened.** `project_overlaps` did `argsort` of 200M elements on CPU on
the very first optimization step, silently. No log output indicated this was
happening.

**Root cause.** `step % overlap_interval == 0` evaluates to `True` when
`step=0` regardless of the interval value. This is a classic modular arithmetic
edge case. Additionally, no log line preceded the projection, so the operation
was invisible in output.

**Fix applied.** Skip projection on step 0 (`step > 0 and step % interval == 0`).
Added log line before projection. Increased interval to 200 for N > 50M.

**Architectural lesson.** Modular operations in a loop should never silently
consume minutes. Two rules: (1) Always log before expensive operations. If you
can't see it in the log, you can't debug it. (2) Watch out for `step % N == 0`
at `step=0` -- this is a recurring bug pattern in optimization loops.

---

### Bug 12: Original graph not offloaded in hierarchy restore path

**What happened.** The original graph offload code (nulling `_edge_index_tensor`,
writing to disk) only ran in the fresh-build path (`else` branch), not in the
checkpoint-restore path (`if` branch). When restoring from checkpoint, the
original graph's 24GB edge index stayed in memory unnecessarily.

**Root cause.** The offload optimization was added inside the `else` branch
(build hierarchy from scratch) without checking whether the `if` branch
(restore from checkpoint) needed the same treatment.

**Fix applied.** Moved the offload block to run after both branches merge.
Added a `_original_graph_path is None` guard so it only runs once.

**Architectural lesson.** When adding an optimization to one code path, always
check if the alternative path needs the same optimization. `if/else` branches
that handle the same logical operation (whether fresh or cached) often need the
same cleanup, resource release, or state normalization at the end. A good
pattern: put shared cleanup AFTER the branch point, not inside either branch.

---

### Bug 14: Coarsen streaming threshold only checking node count, not edge count

**What happened.** `coarsen_once` used the non-streaming path for 37M nodes
(below the 100M streaming threshold) even though it had 942M edges. The
non-streaming path allocated `torch.ones_like` tensors of size E, consuming
15GB.

**Root cause.** `_STREAMING_THRESHOLD` only checked `N`, not `E`. The threshold
was designed around node count because early testing focused on node-heavy graphs
where N and E scaled together.

**Fix applied.** Also trigger streaming when `E > 100M`.

**Architectural lesson.** Memory usage depends on BOTH nodes and edges.
Thresholds that gate memory-saving optimizations should check whichever
dimension dominates memory for the operation in question. For adjacency
operations, edge count matters more. For per-node operations, node count
matters more. Check both.

---

## Cross-Cutting Patterns

Several patterns recur across multiple bugs. These are the systemic issues.

### Pattern: Reference Discipline (Bugs 10, 17)
Python's reference counting means "offloading" or "freeing" large tensors
requires nulling every reference path. `del local_var` is necessary but not
sufficient. Audit the object graph, not just the local scope.

**Rule:** When writing offload/free code, list every reference to the target
tensor, then null each one. Add a comment listing them.

### Pattern: Optimization Symmetry (Bugs 1, 12, 14, 16)
When an optimization is added to one code path, the parallel path often needs
the same treatment. Sharing for edges but not samples (Bug 1). Offload in
fresh-build but not restore (Bug 12). Streaming threshold for N but not E
(Bug 14). Reload for edge_index but not layer_assignments (Bug 16).

**Rule:** When adding an optimization, immediately grep for the parallel case.
If a function has both a "compute" and "restore" path, both need the same
post-processing.

### Pattern: Advisory Flags Without Enforcement (Bug 3)
`is_heavy=True` existed as metadata but nothing enforced it. Flags that mean
"this is dangerous at scale" must be enforced as gates, not treated as
documentation.

**Rule:** If a flag exists to mark something as expensive/dangerous, write
the enforcement code at the same time as the flag. Unenforced flags are
misleading comments.

### Pattern: Obvious Answers at Extreme Scale (Bugs 4, 5, 13)
Classification (always GENERAL), layering (redundant), GPU vs CPU selection
(wrong algorithm). At extreme scale, many analytical questions have trivially
known answers.

**Rule:** Before running expensive analysis, check if the answer is already
determined by the scale of the input. Add early returns for N > threshold.

### Pattern: O(N log N) vs O(N) Matters at 10^9 (Bug 6)
At a billion elements, log2(N) ~ 30. An O(N log N) algorithm is 30x slower
than O(N). Use counting sort for integer keys, radix sort for fixed-width keys.

**Rule:** For any sort on integer keys in a known range, use counting sort.
Profile sort operations above 100M elements.

### Pattern: Linux Memory Management (Bug 11)
Python gc.collect() is necessary but not sufficient on Linux. glibc may hold
freed pages in its free list indefinitely. malloc_trim(0) is required to
return pages to the OS.

**Rule:** After freeing > 1GB of data, call both `gc.collect()` and
`ctypes.CDLL("libc.so.6").malloc_trim(0)`. Log the RSS before and after.

### Pattern: Silent Expensive Operations (Bugs 2, 3, 4, 5)
Multiple bugs involved operations consuming minutes of CPU time with no log
output. The developer couldn't tell what the system was doing.

**Rule:** Log before every operation that might take > 1 second. Include the
data size. "Computing CSR for 1.5B edges..." is far more useful than silence
followed by an OOM crash.
