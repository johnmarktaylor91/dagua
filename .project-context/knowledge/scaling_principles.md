# Scaling Operational Principles

Derived from 17 bugs during the 1B-node scaling campaign (2026-03-18).
Hardened through adversarial critique by Claude and Codex agents.

These are not suggestions. They are operating rules for billion-scale work.

---

## A. Memory Discipline

### A1. Budget peak memory, not target allocation
Before any stage that allocates proportional to N or E, compute estimated PEAK
memory: target buffers + temporaries + dtype promotions + sort workspaces +
autograd overhead (typically 3-4x base tensor). Abort if estimated peak exceeds
70% of available RAM or VRAM. The estimate must be computed, not guessed.
- Prevents: OOM from hidden temporaries, sort buffers, dtype upcasts

### A2. Verify resource release by measurement, not by code inspection
After any offload/free operation, assert that the resource metric decreased by
the expected amount: RSS for CPU, `torch.cuda.memory_allocated()` for GPU,
open fd count for files. If it didn't decrease, there's a dangling reference.
`del x` means nothing if another scope holds the same object.
- Prevents: Graph object holding 24GB after "offload", glibc hoarding pages

### A3. Reclaim memory by resource class
After freeing >1GB: (1) `gc.collect()` for Python refcounting, (2)
`malloc_trim(0)` for glibc page return, (3) `torch.cuda.empty_cache()` for
CUDA pool, (4) verify RSS/VRAM decreased. All four steps, every time.
Skipping any one allows a different resource class to hold the memory.
- Prevents: glibc malloc arena inflation, CUDA cached-but-unused blocks

### A4. Gate on topology sketch, not raw counts
Any threshold or mode selection must consider N, E, max_degree, depth,
max_layer_width, and E/N ratio. A graph with 37M nodes and 942M edges is
fundamentally different from 37M nodes and 50M edges. Use the topology
sketch to drive algorithm selection, not just `if N > threshold`.
- Prevents: Wrong execution mode for graphs with unexpected shape

### A5. Temp storage must pass a capability check
Before writing large temp files: check free bytes, user quota, throughput
(local vs network), and cleanup guarantees. Never default to /tmp without
verifying it can hold the artifacts. Prefer local SSD over network mounts
for intermediate data.
- Prevents: /tmp overflow, network mount stalls during checkpoint writes

---

## B. Algorithm Selection

### B1. Document cost model, not just Big-O
Every hot-path function needs: asymptotic complexity, estimated pass count,
peak auxiliary memory, device residency, and trigger frequency. Big-O alone
is too weak -- an O(E) pass that runs 31K times per step is O(31K*E).
Add runtime scaling assertions in integration tests: if 10M takes >50x
what 1M took, the annotation is wrong.
- Prevents: O(waves*E) GPU layering hiding behind "O(E) per wave"

### B2. Select sort algorithm from key range AND memory budget
Counting sort for dense bounded keys. Radix sort for sparse bounded keys.
Comparison sort when key range is unbounded. Never reflexively pick one.
At 1.5B elements, the wrong choice is 30 minutes vs 30 seconds.
- Prevents: O(E log E) argsort where O(E) counting sort suffices

### B3. Use sketch-based classification at scale
For N > 10M, replace full classification with bounded-memory probes:
density (E/N), depth estimate (BFS from roots), max degree, component count.
These are O(N) or O(E) and sufficient to choose the right algorithm.
Never compute O(N^2) properties just to confirm the graph is "large."
- Prevents: 20GB degree computation for graphs that are obviously GENERAL

### B4. Choose traversal from measurement, not doctrine
"GPU is faster" and "CSR is faster" are both wrong as universal rules.
GPU wave-scan is O(waves*E), CPU CSR is O(N+E). Measure depth, frontier
width, and device budget. Use the algorithm whose complexity class matches
the actual graph shape. Log the decision and the numbers.
- Prevents: GPU layering being 1000x slower than CPU for deep graphs

---

## C. Code Path Completeness

### C1. Share immutable, versioned structures
When multiple consumers need the same derived data, compute once and share.
Shared structures must be immutable (or copied on write). Tag with a version
hash of inputs. Consumers must check the version matches their expectations.
- Prevents: Redundant computation, stale shared caches

### C2. Build and restore paths must be provably symmetric
Every resource allocated in "build" must have a corresponding operation in
"restore," and vice versa. After writing either path, diff them and verify
1:1 correspondence. Add a round-trip test: build -> save -> restore -> verify
semantic equivalence with `torch.allclose()` on non-trivial inputs.
- Prevents: Offload in build but not restore, missing fields in reload

### C3. Test every boundary transition, not just step 0
Enumerate ALL periodic triggers and boundary transitions: step 0->1, batch
boundaries, level transitions, device transfers, dtype changes. Test each
explicitly. Test the LCM of all periodic intervals to catch interaction bugs.
`step % N == 0` always fires at step 0 -- guard with `step > 0` if step-0
execution is expensive or meaningless.
- Prevents: Projection argsort at step 0, periodic trigger collisions

---

## D. Checkpoint and Persistence

### D1. Fingerprint schema version and data shape, not source code
Cache validity = schema version (increments on semantic changes) + data
shapes + dtype + coordinate space. Source code hashes cause false invalidation
during active development. Shape-only checks allow silent semantic drift.
Include both.
- Prevents: Checkpoint invalidation on every code edit, stale semantic data

### D2. Validate structural invariants, not just presence
After building or restoring a hierarchy: check depth, referential integrity
(every parent/child resolves), node-count conservation between levels,
and that the coarsest level meets the target size. Presence of files is
not completeness.
- Prevents: Incomplete hierarchy accepted as complete, dangling references

### D3. Checkpoint schema is code, not documentation
Maintain an explicit manifest (dataclass, constant, or typed schema) of
every field that constitutes a valid checkpoint. Save iterates the manifest.
Load asserts every field was restored. Adding a field to the saved object
without updating the manifest must cause a test failure.
- Prevents: Save/load asymmetry when new state is added

---

## E. Debugging at Scale

### E1. Enforce per-stage budgets with fail-fast guards
Every pipeline stage must have a declared time and memory budget based on
the input topology sketch. If estimated cost exceeds budget, abort with a
diagnostic BEFORE execution. Log actual cost after completion. If actual
exceeds estimate by >2x, emit a warning. This is a circuit breaker, not
a dashboard.
- Prevents: 30-minute stages that could have been aborted in 1 second

### E2. Diagnose comprehensively before fixing
When a failure is observed, DO NOT fix the first cause you find. Instead:
(a) enumerate all suspicious behaviors in the failing run, (b) trace the
full execution path and list every O(N+) operation with byte-level estimates,
(c) check each independently, (d) fix ALL confirmed issues in one pass,
(e) re-run. Declare "fixed" only when the diagnostic sweep is clean.
Serial hypothesis testing (find one bug, fix, hope) is the root cause of
whack-a-mole debugging.
- Prevents: 10+ rounds of fix-fail-fix-fail

### E3. Run threshold ladder, not single smoke test
Test at 3+ scale points (100K, 1M, 10M) and verify wall time and RSS
scale as expected. Test just below AND just above every mode-switch
threshold. Include adversarial topology variants: extreme depth, extreme
width, extreme degree skew, high E/N ratio. A single 1M-node smoke test
misses threshold-dependent bugs entirely.
- Prevents: Bugs that only manifest at specific scale thresholds

### E4. Every long-run bug must create a reusable guardrail
A fix is incomplete until it adds at least one of: a runtime assertion,
a threshold test, a round-trip test, an ownership assertion, a budget
check, or a topology-aware mode-selection test. Patches without guardrails
are anecdotes, not defenses. The next sibling bug will slip through.
- Prevents: Fixing the symptom without defending against the class

### E5. Every fix must pass three gates
After fixing a bug: (1) no crash at the failing scale, (2) output matches
reference within tolerance, (3) resource usage within 1.5x of pre-bug
baseline. If any gate fails, the fix is incomplete. "It runs" is not
"it works."
- Prevents: Fixes that prevent crashes but produce wrong output or regress perf

---

## Quick Reference

| Category | Principles | Key Rule |
|----------|-----------|----------|
| Memory | A1-A5 | Budget peak, verify release, reclaim by class |
| Algorithms | B1-B4 | Cost model > Big-O, measure > doctrine |
| Code paths | C1-C3 | Immutable sharing, symmetric build/restore, test boundaries |
| Checkpoints | D1-D3 | Schema version, structural invariants, manifest as code |
| Debugging | E1-E5 | Fail-fast budgets, diagnose comprehensively, guardrail every fix |
