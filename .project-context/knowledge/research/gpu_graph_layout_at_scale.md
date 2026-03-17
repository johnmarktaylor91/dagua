# GPU Graph Layout at Scale: Research Survey

*Date: 2026-03-17*
*Context: Dagua OOMs at 200M nodes on 11GB GPU. Researching how existing systems handle 100M-1B+.*

---

## 1. cuGraph ForceAtlas2 (NVIDIA RAPIDS)

**Scale:** Up to ~500M edges on 32GB GPU (device-only); larger with UVM (unified virtual memory).
**Memory model:** Peak allocation = 30*V (Python API) / 17*V (C++ API). For 200M nodes at 30*V with float32 = 24GB. Uses COO edge format to parallelize per-edge.
**Gradients:** Analytical (hand-coded). No autograd. Forces computed directly from position differences.
**Barnes-Hut:** Yes, default enabled. O(V log V) repulsion via quadtree. theta=0.5 default.
**Key insight:** COO > CSR for GPU layout because each thread processes one edge independently. No workload imbalance from degree skew. Memory is 30 floats per vertex -- that's it.
**Composable losses:** No. Hardcoded ForceAtlas2 physics model only.
**UVM:** cuGraph uses RAPIDS Memory Manager (RMM) with unified memory to exceed device VRAM, spilling to host memory transparently. Performance degrades but doesn't OOM.

Source: https://docs.rapids.ai/api/cugraph/nightly/api_docs/api/cugraph/cugraph.force_atlas2/
Source: https://medium.com/rapids-ai/tackling-large-graphs-with-rapids-cugraph-and-unified-virtual-memory-b5b69a065d4

## 2. GPUGraphLayout (Leiden University, Brinkmann et al.)

**Scale:** 4M nodes / 120M edges in 14 minutes. "Millions of nodes feasible."
**Memory model:** Not documented precisely. Uses Barnes-Hut from LonestarGPU.
**Gradients:** Analytical. Direct force computation, no autograd framework.
**Barnes-Hut:** Yes, CUDA implementation from Burtscher & Pingali's LonestarGPU.
**Key insight:** 40-123x speedup over CPU by adapting ForceAtlas2 data structures to CUDA memory coalescing, shared memory usage, and thread workload balance.
**Composable losses:** No. ForceAtlas2 only.

Source: https://github.com/govertb/GPUGraphLayout
Source: https://liacs.leidenuniv.nl/~takesfw/pdf/network-visualization-gpu-icpp2018.pdf

## 3. sfdp (Graphviz, Yifan Hu)

**Scale:** Tested to ~100K nodes. Designed for "very large graphs" but CPU-only.
**Memory model:** O(V + E) -- linear. Barnes-Hut octree is O(V). Edge collapsing coarsening.
**Gradients:** Analytical. Classical spring-electrical model with closed-form force derivatives.
**Multilevel:** Edge collapsing coarsening -> layout small -> interpolate -> refine large. Key to escaping local minima and reducing iterations at each level.
**Barnes-Hut:** Yes, octree. O(V log V) per iteration instead of O(V^2).
**Key insight:** Multilevel is essential. Without it, quality degrades badly on large graphs. The coarsening preserves essential connectivity while reducing problem size 2-4x per level.
**Composable losses:** No. Fixed spring-electrical model.

Source: http://yifanhu.net/SOFTWARE/SFDP/index.html
Source: https://graphviz.org/docs/layouts/sfdp/

## 4. Graph Drawing by SGD (Zheng et al. 2018) + (SGD)^2 (Ahmed et al. 2022)

**Scale:** 115K nodes (Zheng), millions of nodes ((SGD)^2).
**Memory model:** O(n*h) where h = pivot count. Full stress requires O(n^2) for all-pairs distances; pivots reduce to O(n*h). Per-iteration: only ONE pair updated at a time.
**Gradients:** Analytical. Explicit derivative: dQ_ij/dX_i = 4*w_ij*r where r = displacement.
**Edge/pair sampling:** Core innovation. Instead of computing full gradient over all pairs, randomly sample one pair per step. Stress = sum of pairwise terms -> stochastic decomposition.
**Learning rate:** Exponential decay eta_max * exp(-lambda*t), then 1/t for convergence guarantee.
**Key insight:** SGD on pairwise stress terms = O(1) memory per update, O(1) compute per update. Scale comes from doing MANY cheap updates rather than few expensive full-gradient steps. No quadratic memory ever allocated.
**Composable losses:** YES -- (SGD)^2 explicitly supports "any criterion that can be described by a differentiable function." Criteria include: stress, ideal edge lengths, neighborhood preservation, node resolution, angular resolution, aspect ratio. Each criterion samples different elements (pairs for stress, edges for edge length, etc.).
**GPU compatibility:** Not GPU-native in the papers, but the sampling approach is inherently parallelizable -- sample batches of pairs and update in parallel.

THIS IS THE MOST RELEVANT SYSTEM FOR DAGUA.

Source: https://ar5iv.labs.arxiv.org/html/1710.04626
Source: https://arxiv.org/abs/2112.01571

## 5. Rapid GPU-Based Pangenome Graph Layout (Li et al., SC 2024)

**Scale:** 11M nodes (chromosome 1). All 24 human chromosomes.
**Memory model:** ~32-64 bytes per node. Lean data structure, static allocation. No dynamic GPU malloc during kernels. Estimated 2-4 GB for 11M nodes on RTX A6000 (48GB).
**Gradients:** Analytical. Hand-coded stress gradient: stress_ij = ((||v_i - v_j|| - d_ref) / d_ref)^2. Manual gradient descent update, no autograd.
**Sampling:** Sampled path stress -- 100*|p| random pairs per path. Converts O(n^2) to O(n).
**GPU optimizations:**
  - Cache-friendly data layout (struct packing)
  - Coalesced random states (struct-of-arrays for PRNG)
  - Warp merging (eliminate branch divergence)
**Key insight:** Sampling makes it linear. 57x over multithreaded CPU. Quality maintained because randomness is essential for convergence anyway.
**Composable losses:** No. Path stress only.

Source: https://arxiv.org/html/2409.00876v1

## 6. N-body / Molecular Dynamics on GPU

### GPU Gems 3 (NVIDIA, all-pairs)
**Scale:** 16K bodies at 38 timesteps/sec (all-pairs). Larger with Barnes-Hut.
**Memory per particle:** 32 bytes (float4 position + float4 acceleration).
**Key insight:** Tiling. Load p bodies into shared memory, compute p^2 interactions from 2p reads. Memory traffic = N^2/p per timestep. 20 FLOPs per pair.
**Composable:** Any pairwise force expressible as analytical function of distance.

### 50M atoms on 11GB GPU (single GPU MD, 2019)
**Scale:** 50M atoms on GTX 1080 Ti (11GB) -- 2.5x faster than LAMMPS GPU.
**Key insight:** Dynamic cell lists instead of neighbor lists. Only cells containing atoms consume memory. Two-step atom location scheme. No neighbor list = massive memory savings because neighbor lists are O(N*max_neighbors).
**Memory model:** ~220 bytes/atom estimated (11GB / 50M). Eliminates redundant storage.

### HOOMD-blue
**Scale:** 108M particles across 3375 GPUs.
**Key insight:** All data device-resident. CPU is just a driver. BVH (bounding volume hierarchy) instead of cell lists for heterogeneous particle sizes.

Source: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
Source: https://www.sciencedirect.com/science/article/abs/pii/S0965997818306811

## 7. Rapid Multipole Method on GPU (Garland et al., 2008)

**Scale:** Hundreds of thousands of nodes in seconds.
**Memory model:** k-d tree instead of quadtree. Stackless traversal with success/failure pointers (2 pointers per node). O(V log V + E) time.
**Key insight:** k-d tree outperforms quadtree on GPU because of streaming SIMD-friendly traversal pattern with two-pointer scheme.
**Composable:** No. Spring-electrical model only.

Source: https://mgarland.org/files/papers/layoutgpu.pdf

## 8. NeuLay: GNN-Accelerated Layout (Nature Communications, 2023)

**Scale:** ~23K nodes (limited by GNN training cost).
**Key insight:** Reparametrize node positions as GNN outputs. 10-100x speedup in convergence (fewer iterations to reach lower energy). GNN captures structural patterns that FDL misses.
**Gradients:** Yes, full autograd through GNN. But the model is small (2 GCN layers).
**Composable:** Theoretically yes ("any energy minimization problem on graphs").
**Limitation:** Must retrain per graph. Not a general-purpose layout engine.

Source: https://www.nature.com/articles/s41467-023-37189-2

## 9. GraphWaGu (WebGPU, 2022)

**Scale:** 100K nodes, 2M edges.
**Key insight:** Barnes-Hut quadtree in WebGPU compute shaders. Pointerless quadtree with Hilbert ordering for bottom-up parallel construction.
**Composable:** No. Fruchterman-Reingold only.

Source: https://stevepetruzza.io/pubs/graphwagu-2022.pdf

## 10. cosmos.gl / Cosmograph

**Scale:** 1M+ nodes in browser via WebGL.
**Key insight:** All computation in fragment/vertex shaders. No CPU-GPU transfer during layout. Apache Arrow for efficient initial data transfer.
**Composable:** No. Standard force model only.

Source: https://github.com/cosmosgl/graph

---

## Synthesis: What All Successful Systems Have in Common

### Pattern 1: No Autograd
EVERY system that scales past 1M nodes uses analytical (hand-coded) gradients.
Not a single one uses PyTorch autograd or any AD framework. The overhead of
building and storing the computational graph is simply incompatible with the
memory budget at scale. At 200M nodes, even 1 extra float per node = 800MB.

### Pattern 2: Force Approximation (Barnes-Hut / FMM)
All-pairs repulsion is O(N^2) in memory and compute. Every scaled system uses
either Barnes-Hut (quadtree/octree, O(N log N)) or FMM (O(N)), or SGD sampling
(O(batch_size)). The specific choice varies but the principle is universal:
never materialize the full N x N interaction matrix.

### Pattern 3: Sampling / Stochastic Updates
The most scalable approaches (SGD graph drawing, pangenome layout) don't even
build spatial trees. They just SAMPLE random pairs/edges and update. This is
O(1) memory per update and empirically converges well. This is closest to what
dagua could adopt while preserving composability.

### Pattern 4: Linear Memory Per Node
Successful systems budget 30-64 bytes per node (cuGraph: 30 floats, MD: ~220
bytes for complex potentials, pangenome: 32-64 bytes). Compare to dagua with
autograd: position (8B) + grad (8B) + Adam state (32B) + autograd overhead
(unknown but substantial) = easily 100+ bytes per node before any loss
computation, and intermediate tensors during forward/backward can blow this up
by 10-100x depending on loss complexity.

### Pattern 5: Multilevel / Coarsening
Most systems targeting >100K nodes use multilevel approaches. Coarsen the graph,
layout the coarse version, interpolate, refine. This both reduces the problem
size AND provides better initialization (escaping local minima). sfdp, dagua's
own multilevel.py, and MD neighbor lists all exploit spatial locality.

### Pattern 6: Device-Resident Data
The 50M-atom MD paper and HOOMD-blue both emphasize: keep ALL data on GPU. Every
CPU-GPU transfer is a performance cliff. Hybrid device mode (compute on CPU,
transfer gradients) is a crutch, not a solution.

---

## Implications for Dagua

### The Core Problem
Dagua's differentiability through PyTorch autograd is both its greatest strength
(composable user-defined loss functions) and its fundamental scaling limitation.
At 200M nodes, autograd's intermediate tensor storage dwarfs the actual data.

### Option A: Analytical Gradients for Built-in Losses (Hybrid Approach)
Write custom torch.autograd.Function with hand-coded backward() for each
built-in loss. Forward computes loss scalar; backward computes gradient
analytically without materializing intermediate tensors. This preserves the
PyTorch optimizer interface while eliminating autograd overhead for known losses.

User-defined losses still go through autograd (and are thus limited to smaller
graphs), but 95% of usage uses built-in losses.

Memory savings: Eliminate the 2-3x autograd intermediate factor entirely for
built-in losses. At 200M nodes this could be the difference between OOM and fit.

### Option B: SGD Pairwise Sampling ((SGD)^2 style)
Instead of computing ALL repulsion/attraction/stress at once, sample batches of
pairs/edges per step. Each step touches O(batch_size) pairs, not O(N^2) or O(N).
Autograd overhead is then proportional to batch_size, not N.

This is the ONLY technique proven to scale to millions of nodes while supporting
arbitrary differentiable loss functions ((SGD)^2 paper).

Dagua already does edge sampling. The missing piece is node-pair sampling for
repulsion and overlap.

### Option C: Barnes-Hut for Repulsion
Build a quadtree on GPU, approximate far-field repulsion. Reduces repulsion from
O(N^2) to O(N log N). cuGraph and GPUGraphLayout both do this. The quadtree
itself is O(N) memory.

Challenge: quadtree construction and traversal require custom CUDA kernels, and
making them differentiable through autograd is non-trivial. Would likely need
Option A (analytical gradients) for the Barnes-Hut loss.

### Option D: Unified Virtual Memory (UVM)
Use CUDA UVM (or PyTorch's analogous managed memory) to transparently spill to
host RAM. cuGraph does this via RMM. Performance degrades but doesn't crash.

Easiest to implement but slowest. Should be a last-resort fallback, not the
primary strategy.

### Recommended Priority
1. **Option A + B combined**: Analytical gradients for built-in losses + SGD
   pair sampling for repulsion/overlap. This directly follows the (SGD)^2 pattern
   and is the most proven approach for composable, scalable layout.
2. **Option C**: Barnes-Hut for repulsion as a second phase optimization.
3. **Option D**: UVM as a safety net to prevent hard OOM crashes.

### Memory Budget Target
To fit 200M nodes on 11GB GPU:
- 200M * 2 coords * 4 bytes = 1.6 GB (positions)
- 200M * 2 coords * 4 bytes * 2 = 3.2 GB (Adam m + v states)
- Total base: ~5 GB
- Remaining for computation: ~5.5 GB (after CUDA context)
- This means loss computation must use AT MOST 5.5GB of intermediates.
- With analytical gradients + sampling, this is achievable.
- With full autograd on all nodes: impossible.
