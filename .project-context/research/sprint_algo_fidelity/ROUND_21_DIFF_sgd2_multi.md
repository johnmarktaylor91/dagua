# Round 21 adversarial diff: sgd2_multi

Pairing: dagua `classic_sgd2_multi` vs reference `sgd2_multi_ref`.

Scope: diagnosis only. No dagua source files were edited. The only write from this round is this report.

Key caveat: the configured reference adapter expects upstream `GD2` modules at `/tmp/graph-drawing/gd2.py` and `/tmp/graph-drawing/criteria.py` (`dagua/eval/competitors/sgd2_multi_competitor.py:23`, `dagua/eval/competitors/sgd2_multi_competitor.py:102`, `dagua/eval/competitors/sgd2_multi_competitor.py:307`). That clone path did not exist at start of diagnosis, and a fresh clone of `github.com/tiga1231/graph-drawing` did not contain those modules. Therefore, exact upstream `GD2.optimize` internals cannot be line-cited from the current environment. The line-cited reference behavior below is the benchmark adapter and its runtime monkeypatches, plus the files present in the fetched upstream repository.

## 1. Files read

Dagua implementation:

- `dagua/layout/ops/sgd2_multi.py`
- `dagua/layout/ops/pipelines/sgd2_multi.py`
- `dagua/layout/ops/graph_utils.py`
- `dagua/layout/_archive/classic/sgd2_multi.py`
- `dagua/layout/_archive/classic/_graph_distances.py`
- `dagua/eval/variants.py`
- `dagua/eval/competitors/classic_competitor.py`
- `dagua/eval/competitors/sgd2_multi_competitor.py`
- `dagua/eval/competitors/sgd2_competitor.py` was located as related `s_gd2` adapter context, but the `sgd2_multi_ref` path does not use it.
- `eval_output/fidelity_report/report.md`
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`

Reference-side search and fetched upstream files:

- Attempted import/path search for `sgd2`, `s_gd2`, `graph_sgd2`, and installed modules with `sgd` in their names. No `sgd2` or `sgd2_multi` package was importable in the active environment.
- `/tmp/graph-drawing` was absent at the start. The adapter hard-codes that path (`dagua/eval/competitors/sgd2_multi_competitor.py:23`) and reports availability only when `/tmp/graph-drawing/gd2.py` exists (`dagua/eval/competitors/sgd2_multi_competitor.py:26-28`).
- Fetched `https://github.com/tiga1231/graph-drawing` into `/tmp/graph-drawing` for source inspection. The fetched HEAD has no `gd2.py`, no `criteria.py`, and no `utils.py`; `git ls-tree` and `git log --name-only` found only generated/demo GD2 data paths, not source modules.
- `/tmp/graph-drawing/lovasz_losses.py`
- `/tmp/graph-drawing/socketio/models/stress.py`

Existing reports:

- `eval_output/fidelity_report/report.md` marks all `sgd2_multi_*` variants weak-equivalent except `sgd2_multi_with_crossing`, which is partial-match: `sgd2_multi_default` median RMSD 0.168, `stress_only` 0.184, `lr001` 0.219, `lr01` 0.113, `batch8` 0.171, `batch128` 0.171, `with_aspect` 0.203, `with_crossing` 0.150 with only 69 OK cases (`eval_output/fidelity_report/report.md:76-83`).
- The sprint summary does not include an sgd2_multi-specific narrative; it frames stochastic comparison methodology and accepted residuals generally (`.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:111-130`).

## 2. Overall pipeline structure

Dagua side:

1. The public competitor `classic_sgd2_multi` is registered through `_ClassicLayoutSpec` with import path `dagua.layout.ops.pipelines.sgd2_multi`, function `layout_sgd2_multi_pipeline`, default criteria `{"stress": 1.0, "ideal_edge_length": 1.0}`, and default `lr=0.01` (`dagua/eval/competitors/classic_competitor.py:270-277`).
2. The concrete competitor wrapper `ClassicSGD2Multi.layout` calls `_quick_classic` with the same pipeline and hard-coded criteria/lr (`dagua/eval/competitors/classic_competitor.py:1763-1796`).
3. The pipeline builder wires two ops: `_InitSGD2MultiState` followed by `_RunSGD2MultiOptimization` (`dagua/layout/ops/pipelines/sgd2_multi.py:162-226`).
4. The public pipeline validates `num_nodes`, `steps`, `lr`, `momentum`, `grad_clamp`, `batch_size`, and `edge_weights` shape (`dagua/layout/ops/pipelines/sgd2_multi.py:285-303`).
5. A canonical `s_gd2` fallback is attempted only when `criteria is None`, `criteria_schedules is None`, `steps > 0`, and native hyperparameters are exactly lr 1.0, momentum 0.7, grad clamp 4.0, batch size 16 (`dagua/layout/ops/pipelines/sgd2_multi.py:305-320`). The benchmarked `classic_sgd2_multi_*` variants pass explicit criteria and usually lr 0.01, so this fallback is not used for the dagua-vs-`sgd2_multi_ref` family.
6. `_InitSGD2MultiState` resolves criteria schedules, seeds torch, builds graph state, and stores prepared state into `state.extras` (`dagua/layout/ops/sgd2_multi.py:1983-2036`).
7. `_RunSGD2MultiOptimization` builds one `_CyclicSampler` per active criterion, initializes positions as `torch.randn([N,2]) * sqrt(N)`, creates optional crossing detector state, creates optional vertex-resolution state, then runs `steps` iterations of `SGD2MultiOptStep` plus `SGD2MultiConvergenceCheck` (`dagua/layout/ops/sgd2_multi.py:2129-2212`).

Reference adapter side:

1. The registered competitor is `SGD2MultiRef`, name `sgd2_multi_ref`, `max_nodes = 5000`, with variant params `criteria_weights`, `grad_clamp`, `max_iter`, `optimizer_kwargs`, and `sample_sizes` (`dagua/eval/competitors/sgd2_multi_competitor.py:237-245`).
2. Availability is `(_SGD2_REPO / "gd2.py").exists()` where `_SGD2_REPO = Path("/tmp/graph-drawing")` (`dagua/eval/competitors/sgd2_multi_competitor.py:23-28`). In the current environment this was false before fetching, and still false after fetching current upstream HEAD because `gd2.py` is absent.
3. The adapter builds a symmetric SciPy CSR adjacency from dagua edge tensors (`dagua/eval/competitors/sgd2_multi_competitor.py:309-317`), runs `scipy.sparse.csgraph.shortest_path(adj, directed=False)` only to reject disconnected graphs (`dagua/eval/competitors/sgd2_multi_competitor.py:318-326`), then constructs an undirected `networkx.Graph` with nodes and non-self-loop edges (`dagua/eval/competitors/sgd2_multi_competitor.py:328-339`).
4. The adapter seeds torch and NumPy only, not Python `random`, when a seed is supplied (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`).
5. Default adapter optimize kwargs are stress-only, `max_iter=2000`, and optimizer lr 0.01 (`dagua/eval/competitors/sgd2_multi_competitor.py:345-349`). Variant config overrides usually add ideal edge length, crossings, aspect, batch/sample sizes, and grad clamp (`dagua/eval/variants.py:1607-1768`).
6. The adapter supplies default `sample_sizes = {criterion: 128}` when the variant does not provide sample sizes (`dagua/eval/competitors/sgd2_multi_competitor.py:359-364`), disables visualization (`dagua/eval/competitors/sgd2_multi_competitor.py:366-369`), sets final-only evaluation defaults (`dagua/eval/competitors/sgd2_multi_competitor.py:370-374`), and defaults `grad_clamp` to 5.0 (`dagua/eval/competitors/sgd2_multi_competitor.py:375`).
7. The adapter instantiates `GD2(G_nx)`, strips crossing criteria if upstream has no non-incident edge pairs, and runs `gd2.optimize(**optimize_kwargs)` under two compatibility patch contexts (`dagua/eval/competitors/sgd2_multi_competitor.py:377-400`).

Major structural divergence:

- Dagua implements the multicriteria loop in local PyTorch ops. The reference competitor delegates the core loop to missing upstream `GD2.optimize`. Because the upstream source is absent in this environment, line-cited verification is possible only for the adapter, adapter patches, and fetched ancillary files.
- Dagua has a non-reference `s_gd2` fallback path in the public pipeline for default native hyperparams (`dagua/layout/ops/pipelines/sgd2_multi.py:25-100`, `dagua/layout/ops/pipelines/sgd2_multi.py:305-320`). It is gated off for this family by explicit criteria in the variants, but it is a latent behavior difference for direct callers.

## 3. Energy / loss / objective

### Criterion resolution

Dagua:

- Default when no criteria and no schedules: `{"stress": 1.0}` (`dagua/layout/ops/sgd2_multi.py:654-681`).
- Classic competitor default overrides that to `{"stress": 1.0, "ideal_edge_length": 1.0}` with `lr=0.01` (`dagua/eval/competitors/classic_competitor.py:270-277`, `dagua/eval/competitors/classic_competitor.py:1788-1796`).
- Variant configs align dagua criteria to reference `criteria_weights` for default, stress-only, crossing, aspect, lr, and batch variants (`dagua/eval/variants.py:1607-1768`).

Reference adapter:

- Adapter base default is stress-only (`dagua/eval/competitors/sgd2_multi_competitor.py:345-349`), but variants override it to match the dagua side (`dagua/eval/variants.py:1616-1768`).
- `criteria_weights` are copied into `optimize_kwargs` before running upstream (`dagua/eval/competitors/sgd2_multi_competitor.py:359-360`).

### Stress

Dagua formula:

- Shortest-path distances are computed by BFS or Dijkstra and converted to float32 tensors (`dagua/layout/ops/sgd2_multi.py:417-453`).
- Stress pairs are upper triangle, offset 1, finite only; weights are `1 / (distance^2 + 1e-6)` (`dagua/layout/ops/sgd2_multi.py:456-483`).
- Loss is `mean(pair_weights * (||x_i - x_j|| - d_ij)^2)` (`dagua/layout/ops/sgd2_multi.py:858-885`).
- The active criterion samples via `_CyclicSampler` if present, otherwise `torch.randint` (`dagua/layout/ops/sgd2_multi.py:1560-1580`).

Reference adapter patch formula:

- The adapter monkeypatches `criteria.stress` to sample with `np.random.choice(n, sampleSize)` when no sample is supplied, gather `D` and `W` by advanced indexing, compute `PairwiseDistance()(x0, x1)`, and return `mean(W * (pdist - D)^2)` unless `reduce="sum"` (`dagua/eval/competitors/sgd2_multi_competitor.py:147-186`).
- The adapter itself builds a SciPy shortest-path matrix for disconnected rejection, but that `dist` variable is not passed into `GD2` in the visible adapter code (`dagua/eval/competitors/sgd2_multi_competitor.py:318-326`, `dagua/eval/competitors/sgd2_multi_competitor.py:377-400`).

Residual differences:

- Dagua samples only upper-triangle unordered pairs (`dagua/layout/ops/sgd2_multi.py:472-483`). The patched reference stress samples arbitrary ordered `(i0, i1)` pairs with replacement if `sample` is not supplied (`dagua/eval/competitors/sgd2_multi_competitor.py:159-169`). If upstream `GD2.optimize` provides explicit DataLoader samples, this difference may be neutralized, but the missing source prevents verification.
- Dagua stress excludes diagonal pairs by upper-triangle offset 1 (`dagua/layout/ops/sgd2_multi.py:472-477`). Patched reference random sampling can include `i0 == i1` in the no-sample path (`dagua/eval/competitors/sgd2_multi_competitor.py:161-168`), yielding zero-distance diagonal terms if upstream uses that path.
- Dagua uses float32 stress distances and weights after conversion (`dagua/layout/ops/sgd2_multi.py:448-453`). Reference SciPy shortest paths are float64, and patched stress keeps D/W dtype as passed by upstream (`dagua/eval/competitors/sgd2_multi_competitor.py:167-182`).

### Ideal edge length

Dagua formula:

- Edges are cleaned to unique undirected non-self edges (`dagua/layout/ops/sgd2_multi.py:282-312`).
- Ideal loss is `mean(((||x_u - x_v|| - target) / target)^2)` with default target 1.0 and epsilon guard (`dagua/layout/ops/sgd2_multi.py:888-913`).
- Criterion samples from `state.edges` via cyclic sampler or random sampling (`dagua/layout/ops/sgd2_multi.py:1581-1591`).

Reference adapter patch formula:

- Patched `ideal_edge_length` defaults `targetLengths = {e: 1 for e in G.edges}` (`dagua/eval/competitors/sgd2_multi_competitor.py:121-123`).
- It samples `list(G.edges)` via Python `random.sample` if `sampleSize` is supplied and no explicit sample is supplied (`dagua/eval/competitors/sgd2_multi_competitor.py:124-129`).
- It computes `edgeLengths = (source - target).norm(dim=1)`, target tensor, and `((edgeLengths - tl) / tl).pow(2)`, reduced by mean unless `sum` (`dagua/eval/competitors/sgd2_multi_competitor.py:133-143`).

Residual differences:

- Dagua deduplicates undirected edges through `torch.unique` (`dagua/layout/ops/sgd2_multi.py:309-312`); reference uses `networkx.Graph`, which also deduplicates undirected edges and drops parallel multiplicity (`dagua/eval/competitors/sgd2_multi_competitor.py:331-339`).
- Dagua uses torch RNG through `_CyclicSampler` (`dagua/layout/ops/sgd2_multi.py:821-855`); the patched reference fallback path uses Python `random.sample` (`dagua/eval/competitors/sgd2_multi_competitor.py:126-127`), but the adapter never seeds Python `random` (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`). If upstream calls the fallback path, same seed does not imply same edge batch.

### Neighborhood preservation

Dagua formula:

- For each sampled root, bounded BFS to depth 2 collects positives (`dagua/layout/ops/sgd2_multi.py:1003-1039`, `dagua/layout/ops/sgd2_multi.py:1042-1073`).
- It adds `0.5 * positive_count` random negatives using `torch.randint`, unique-sorts nodes, computes logits `-torch.cdist(sampled_pos, sampled_pos) + 1.5`, builds induced adjacency plus identity, then applies Lovasz hinge (`dagua/layout/ops/sgd2_multi.py:962-1000`, `dagua/layout/ops/sgd2_multi.py:1077-1118`).
- Lovasz gradient and hinge formulas match standard `errors = 1 - logits * signs`, descending sort, dot with Jaccard gradient (`dagua/layout/ops/sgd2_multi.py:916-959`).

Reference side:

- The fetched upstream repository includes `lovasz_losses.py`; its `lovasz_grad` computes `intersection = gts - cumsum`, `union = gts + cumsum(1-gt)`, then first differences (`/tmp/graph-drawing/lovasz_losses.py:20-32`).
- Its `lovasz_hinge_flat` computes `signs = 2*labels - 1`, `errors = 1 - logits * signs`, descending sort, and dot with `lovasz_grad` (`/tmp/graph-drawing/lovasz_losses.py:96-113`).
- The unavailable `criteria.py` means exact upstream neighborhood sampling, `k_dist`, and BFS depth cannot be line-cited from the actual `GD2` source.

Residual differences:

- Dagua uses torch negative sampling (`dagua/layout/ops/sgd2_multi.py:1082-1083`). If upstream used NumPy or Python random for negative nodes, seeds and batches diverge.
- Dagua sorts the unique sampled nodes (`dagua/layout/ops/sgd2_multi.py:1077-1083`), which stabilizes target matrix order but can differ from insertion-order reference sampling.

### Crossings

Dagua formula:

- Non-incident edge pairs are precomputed from unique undirected edges with all upper-triangle edge-pair combinations and four endpoint inequality tests (`dagua/layout/ops/sgd2_multi.py:527-556`).
- Crossing labels use orientation signs and an epsilon-inclusive collinearity/on-segment test (`dagua/layout/ops/sgd2_multi.py:1199-1239`).
- Detector architecture is Linear(8,128), LayerNorm, LeakyReLU, Linear(128,512), LayerNorm, LeakyReLU, Linear(512,64), LayerNorm, LeakyReLU, Linear(64,1), Sigmoid (`dagua/layout/ops/sgd2_multi.py:172-205`).
- Training loss is BCE on detached edge-pair positions for two Adam steps at lr 0.01; position loss is BCE against zeros with sum reduction (`dagua/layout/ops/sgd2_multi.py:1242-1284`, `dagua/layout/ops/sgd2_multi.py:1354-1389`).
- The decomposed op samples one crossing batch, trains detector, caches `last_crossing_left/right`, then uses the same cached batch for the position loss (`dagua/layout/ops/sgd2_multi.py:1720-1741`, `dagua/layout/ops/sgd2_multi.py:1805-1816`).

Reference adapter:

- The adapter strips crossing criteria when `gd2.non_incident_edge_pairs` is empty because upstream DataLoader rejects `num_samples=0` (`dagua/eval/competitors/sgd2_multi_competitor.py:377-398`).
- Exact upstream crossing detector architecture is unavailable because `gd2.py`/`criteria.py` are missing. The dagua op header says numeric work was copied from the classic module and matches classic combinations (`dagua/layout/ops/sgd2_multi.py:1-8`), but that is not an upstream line ref.

Residual differences:

- Dagua gracefully returns zero loss when no non-incident pairs are available (`dagua/layout/ops/sgd2_multi.py:1341-1351`, `dagua/layout/ops/sgd2_multi.py:1606-1623`); the reference adapter removes the crossing criterion or falls back to stress-only (`dagua/eval/competitors/sgd2_multi_competitor.py:377-398`). That changes the active objective in edge-poor graphs.
- The `with_crossing` family is the only partial-match sgd2_multi variant in the report (`eval_output/fidelity_report/report.md:83`), consistent with neural detector/RNG/order residuals being the largest remaining divergence.

### Crossing-angle maximization

Dagua formula:

- Labels are exact crossings, vectors are edge direction vectors, cosine similarities are squared, and loss is `mean(label * sim^2 / (1 - sim^2 + eps))` (`dagua/layout/ops/sgd2_multi.py:1392-1422`).

Reference side:

- No current variant in `dagua/eval/variants.py:1607-1768` targets `crossing_angle_maximization`.
- The adapter has only generic `criteria_weights` forwarding (`dagua/eval/competitors/sgd2_multi_competitor.py:350-357`); upstream source for this criterion is unavailable.

### Aspect ratio

Dagua formula:

- If sampled positions have fewer than two points, zero loss (`dagua/layout/ops/sgd2_multi.py:1440-1445`).
- Center positions, compute `torch.linalg.svd`, ratio `s[1] / s[0]` clamped to `[eps, 1-eps]`, target clamped to `[0,1]`, and BCE sum (`dagua/layout/ops/sgd2_multi.py:1442-1452`).
- Criterion samples nodes through cyclic sampler for aspect variants (`dagua/layout/ops/sgd2_multi.py:1637-1643`).

Reference adapter patch formula:

- Patched `aspect_ratio` samples `np.random.choice(n, min(n, sampleSize), replace=False)` if sample size is supplied (`dagua/eval/competitors/sgd2_multi_competitor.py:204-211`).
- The patch returns zero when sample has fewer than two points to avoid upstream SVD crash (`dagua/eval/competitors/sgd2_multi_competitor.py:213-217`).
- It uses `torch.svd(sample - sample.mean(dim=0)).S` and BCE sum on `singular_values[1] / singular_values[0]` against `target[1] / target[0]` (`dagua/eval/competitors/sgd2_multi_competitor.py:219-224`).

Residual differences:

- Dagua uses `torch.linalg.svd`; patched reference uses legacy `torch.svd` (`dagua/layout/ops/sgd2_multi.py:1442-1443`, `dagua/eval/competitors/sgd2_multi_competitor.py:219-224`). Singular values should match numerically for 2D samples, but kernel details and precision can differ at tiny samples.
- Dagua samples aspect batches with torch randperm cyclic epochs (`dagua/layout/ops/sgd2_multi.py:821-855`, `dagua/layout/ops/sgd2_multi.py:1637-1643`); patched reference fallback uses NumPy without replacement (`dagua/eval/competitors/sgd2_multi_competitor.py:209-211`).

### Angular resolution

Dagua formula:

- Incident edge pairs are generated for each node with degree >= 2 (`dagua/layout/ops/sgd2_multi.py:486-524`).
- Loss computes cosine similarity between incident vectors, angle `arccos(clamp(sim, -0.99, 0.99))`, optimal angle `2*pi/degree`, normalized ReLU violation `(-angle + optimal)/optimal`, then BCE against zeros (`dagua/layout/ops/sgd2_multi.py:1455-1485`).

Reference side:

- No current variant in `dagua/eval/variants.py:1607-1768` targets `angular_resolution`.
- The fetched upstream repository contains generated `gd2_layouts_angular_resolution` JSON data paths but no source implementation.

### Vertex resolution

Dagua formula:

- Samples stress pairs, computes pair distances, target distance `dmax / sqrt(N)`, smooths target using previous target and `_VERTEX_RESOLUTION_SMOOTHNESS = 0.1`, then mean squared ReLU of `1 - distance/smoothed_target` (`dagua/layout/ops/sgd2_multi.py:1488-1523`).

Reference side:

- No current variant in `dagua/eval/variants.py:1607-1768` targets `vertex_resolution`.
- Upstream source unavailable.

## 4. Force / gradient computation

Neither side implements hand-coded forces for the multicriteria `sgd2_multi` family visible here. Both use PyTorch autograd over scalar losses:

- Dagua creates `positions = torch.nn.Parameter(...)` (`dagua/layout/ops/sgd2_multi.py:2152-2156`), accumulates scalar weighted losses (`dagua/layout/ops/sgd2_multi.py:1797-1827`), calls `loss.backward()` (`dagua/layout/ops/sgd2_multi.py:1828`), clamps position gradients componentwise (`dagua/layout/ops/sgd2_multi.py:1829-1830`), then `optimizer.step()` (`dagua/layout/ops/sgd2_multi.py:1831`).
- Classic archived implementation does the same in monolithic form (`dagua/layout/_archive/classic/sgd2_multi.py:1597-1617`).
- Reference adapter forwards `grad_clamp` into upstream `gd2.optimize` (`dagua/eval/competitors/sgd2_multi_competitor.py:350-357`, `dagua/eval/competitors/sgd2_multi_competitor.py:375`, `dagua/eval/competitors/sgd2_multi_competitor.py:399-400`), but upstream force/gradient loop is not available for line citation.

Potential force/gradient residuals:

- Dagua clamps position gradients by element with `positions.grad.clamp_(-grad_clamp, grad_clamp)` (`dagua/layout/ops/sgd2_multi.py:1829-1830`). If upstream clamps norm, clamps per-loss gradients before summation, or clamps all trainable parameters including crossing detector, trajectories diverge.
- Dagua trains crossing detector before the position backward, using a separate Adam optimizer for detector only (`dagua/layout/ops/sgd2_multi.py:1354-1389`, `dagua/layout/ops/sgd2_multi.py:2160-2169`). If upstream includes detector training loss in the same graph or different ordering, `with_crossing` diverges.

## 5. Initialization

Dagua:

- `_set_seed` seeds `torch.manual_seed(seed)` and CUDA manual seed only (`dagua/layout/ops/sgd2_multi.py:264-279`).
- Positions are `torch.randn((num_nodes, 2), dtype=float32) * sqrt(num_nodes)` (`dagua/layout/ops/sgd2_multi.py:2152-2156`).
- Crossing detector weights are initialized after samplers and positions, when `crossings` is active (`dagua/layout/ops/sgd2_multi.py:2158-2169`), therefore it consumes torch RNG after sampler permutations and position initialization.
- Empty graph returns empty float32 `[0,2]` and one-node graph returns zeros (`dagua/layout/ops/sgd2_multi.py:1999-2007`).

Reference adapter:

- Seeds torch and NumPy only (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`).
- Builds `GD2(G_nx)` after seeding? No: seeding occurs before `GD2(G_nx)` in the adapter (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`, `dagua/eval/competitors/sgd2_multi_competitor.py:377`).
- Exact upstream position initialization is not line-citable because `gd2.py` is absent. The dagua archive says `randn * sqrt(N)` (`dagua/layout/_archive/classic/sgd2_multi.py:1562-1565`), and the ops pipeline carries the same initialization (`dagua/layout/ops/sgd2_multi.py:2152-2156`).

Initialization divergences:

- Dagua does not seed NumPy (`dagua/layout/ops/sgd2_multi.py:264-279`); reference adapter seeds NumPy (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`). Any NumPy sampling in upstream criterion functions is aligned only on the reference side, not dagua.
- Reference adapter does not seed Python `random` (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`), while its patched ideal-edge fallback uses `random.sample` (`dagua/eval/competitors/sgd2_multi_competitor.py:93-96`, `dagua/eval/competitors/sgd2_multi_competitor.py:126-127`). This is a concrete RNG bug if upstream uses that path.

## 6. Iteration / convergence

Dagua:

- Default public pipeline: `steps=10000`, `lr=1.0`, `momentum=0.7`, `grad_clamp=4.0`, `batch_size=16` (`dagua/layout/ops/pipelines/sgd2_multi.py:162-170`, `dagua/layout/ops/pipelines/sgd2_multi.py:229-241`).
- Benchmark variants override to `steps=2000`, `lr=0.01` except lr variants, `grad_clamp=5.0`, and optional `batch_size` (`dagua/eval/variants.py:1607-1768`).
- Optimizer is `torch.optim.SGD([positions], lr, momentum, nesterov=True)` (`dagua/layout/ops/sgd2_multi.py:2180`).
- Scheduler is `ReduceLROnPlateau(factor=0.9, patience=20000, min_lr=1e-5)` (`dagua/layout/ops/sgd2_multi.py:2181-2186`).
- EMA decay is `0.5 ** (1 / 100)` and scheduler is stepped every 10 iterations on the EMA loss (`dagua/layout/ops/sgd2_multi.py:1897-1903`).
- Early stop occurs when optimizer lr <= `1e-5` (`dagua/layout/ops/sgd2_multi.py:1903-1905`), but with patience 20000 and only 2000 or 10000 iterations, decay is effectively never reached.
- Loop executes until `steps` or convergence (`dagua/layout/ops/sgd2_multi.py:2204-2208`).

Reference adapter:

- Defaults: `max_iter=2000`, optimizer lr 0.01, stress-only; variants align `max_iter`, lr, grad clamp, criteria, sample sizes (`dagua/eval/competitors/sgd2_multi_competitor.py:345-357`, `dagua/eval/variants.py:1607-1768`).
- It sets `evaluate_interval=max_iter` and `evaluate={"stress"}` if not provided (`dagua/eval/competitors/sgd2_multi_competitor.py:370-374`).
- Exact upstream optimizer, scheduler, and early-stop behavior cannot be verified from current source. The dagua archive comment asserts reference uses `ReduceLROnPlateau(patience=20000)` stepped every 10 iterations (`dagua/layout/_archive/classic/sgd2_multi.py:1581-1590`, `dagua/layout/_archive/classic/sgd2_multi.py:1619-1626`).

## 7. Hyperparameter alignment table

| Parameter | Dagua value | Reference adapter value | Match? | Evidence / notes |
|---|---:|---:|---|---|
| family default criteria | `{"stress":1,"ideal_edge_length":1}` for `classic_sgd2_multi` | Variant default same | Y | Dagua wrapper `dagua/eval/competitors/classic_competitor.py:270-277`; variants `dagua/eval/variants.py:1607-1622` |
| adapter base default criteria | N/A for benchmarked variants | `{"stress":1}` | N/A | Adapter base default `dagua/eval/competitors/sgd2_multi_competitor.py:345-349`; variants override |
| default max iterations in variants | 2000 | 2000 | Y | `dagua/eval/variants.py:1613-1620`, repeated through `dagua/eval/variants.py:1764` |
| public pipeline default steps | 10000 | adapter base 2000 | N, but not benchmarked as direct default | `dagua/layout/ops/pipelines/sgd2_multi.py:162-170`; adapter `dagua/eval/competitors/sgd2_multi_competitor.py:345-349` |
| default variant lr | 0.01 | 0.01 | Y | `dagua/eval/variants.py:1612-1621` |
| lr001 | 0.001 | 0.001 | Y | `dagua/eval/variants.py:1686-1701` |
| lr01 | 0.1 | 0.1 | Y | `dagua/eval/variants.py:1707-1722` |
| momentum | 0.7 | upstream unknown | Unknown | Dagua `dagua/layout/ops/pipelines/sgd2_multi.py:237-240`; adapter does not set momentum unless hidden in upstream optimizer kwargs |
| Nesterov | True | upstream unknown | Unknown | Dagua `dagua/layout/ops/sgd2_multi.py:2180`; adapter passes only optimizer kwargs from variants |
| grad clamp | 5.0 in variants | 5.0 in variants/default | Y at API level | Variants `dagua/eval/variants.py:1614-1620`; adapter default `dagua/eval/competitors/sgd2_multi_competitor.py:375` |
| grad clamp default public | 4.0 | 5.0 adapter default | N, direct-call residual | Dagua `dagua/layout/ops/pipelines/sgd2_multi.py:237-240`; adapter `dagua/eval/competitors/sgd2_multi_competitor.py:375` |
| batch default | 16 public/native | adapter `sample_sizes` defaults 128 | N unless variant sets batch | Dagua `dagua/layout/ops/pipelines/sgd2_multi.py:237-240`; reference `dagua/eval/competitors/sgd2_multi_competitor.py:361-364` |
| batch8 | batch_size 8 | sample_sizes 8 per criterion | Y at size level | `dagua/eval/variants.py:1728-1745` |
| batch128 | batch_size 128 | sample_sizes 128 per criterion | Y at size level | `dagua/eval/variants.py:1751-1768` |
| stress weight | variant-specific | variant-specific | Y | `dagua/eval/variants.py:1607-1768` |
| ideal edge target | 1.0 | 1.0 in patch | Y | Dagua `dagua/layout/ops/sgd2_multi.py:42-45`, `dagua/layout/ops/sgd2_multi.py:888-913`; patch `dagua/eval/competitors/sgd2_multi_competitor.py:121-143` |
| aspect target | 1.0 minor/major ratio | `(1,1)` tuple ratio | Y | Dagua `dagua/layout/ops/sgd2_multi.py:42-45`, `dagua/layout/ops/sgd2_multi.py:1425-1452`; patch `dagua/eval/competitors/sgd2_multi_competitor.py:190-224` |
| crossing detector train steps | 2 | upstream unknown | Unknown | Dagua `dagua/layout/ops/sgd2_multi.py:50-51`, `dagua/layout/ops/sgd2_multi.py:1381-1387` |
| crossing detector lr | 0.01 | upstream unknown | Unknown | Dagua `dagua/layout/ops/sgd2_multi.py:50-51`, `dagua/layout/ops/sgd2_multi.py:2160-2169` |
| scheduler factor | 0.9 | upstream unknown; archive says 0.9 | Unknown/Y inferred | Dagua `dagua/layout/ops/sgd2_multi.py:2055-2057`, `dagua/layout/ops/sgd2_multi.py:2181-2186`; archive `dagua/layout/_archive/classic/sgd2_multi.py:1585-1590` |
| scheduler patience | 20000 | upstream unknown; archive says 20000 | Unknown/Y inferred | Same as above |
| scheduler min lr | 1e-5 | upstream unknown; archive says 1e-5 | Unknown/Y inferred | Same as above |
| EMA half-life | 100 | upstream unknown; archive says 100 | Unknown/Y inferred | Dagua `dagua/layout/ops/sgd2_multi.py:1845-1864`, archive `dagua/layout/_archive/classic/sgd2_multi.py:1592-1624` |
| graph directedness | undirected for distances/edges | undirected NetworkX Graph | Y | Dagua `_build_adjacency` `dagua/layout/ops/sgd2_multi.py:315-340`; reference `dagua/eval/competitors/sgd2_multi_competitor.py:331-339` |
| disconnected handling | finite pairs only, layouts disconnected graphs | rejects disconnected graph | N | Dagua `dagua/layout/ops/sgd2_multi.py:478-483`; reference `dagua/eval/competitors/sgd2_multi_competitor.py:318-326` |
| weighted edges | supported in pipeline/APSP | not passed into `GD2` adapter | N | Dagua `dagua/layout/ops/pipelines/sgd2_multi.py:297-303`, `dagua/layout/ops/sgd2_multi.py:619-623`; reference rows/data ones `dagua/eval/competitors/sgd2_multi_competitor.py:312-317` |
| self loops | ignored by cleaned edges/adjs | skipped when building NetworkX graph | Y | Dagua `dagua/layout/ops/sgd2_multi.py:301-307`; reference `dagua/eval/competitors/sgd2_multi_competitor.py:333-339` |
| multi-edges | unique undirected edge list; adjacency sums duplicate weights in ops graph_utils | NetworkX Graph dedups | Mostly Y unweighted; N weighted/multiplicity | Dagua edges `dagua/layout/ops/sgd2_multi.py:309-312`; graph_utils accumulation `dagua/layout/ops/graph_utils.py:226-260`; reference `dagua/eval/competitors/sgd2_multi_competitor.py:331-339` |

## 8. Edge cases

Self-loops:

- Dagua `_clean_undirected_edges` removes self-loops before edge-based criteria (`dagua/layout/ops/sgd2_multi.py:301-307`).
- Dagua adjacency builder skips self-loops through `build_undirected_adjacency` in graph utils (`dagua/layout/ops/graph_utils.py:260-266`).
- Reference adapter skips self-loops when adding edges to `networkx.Graph` (`dagua/eval/competitors/sgd2_multi_competitor.py:333-339`).
- Alignment: yes for unweighted semantics.

Multi-edges:

- Dagua `_clean_undirected_edges` deduplicates the edge list for ideal length and crossing pools (`dagua/layout/ops/sgd2_multi.py:309-312`).
- Dagua ops use `dagua.layout.ops.graph_utils.build_undirected_adjacency`, which accumulates duplicate-edge weights additively (`dagua/layout/ops/graph_utils.py:226-260`, with rationale note at `dagua/layout/ops/graph_utils.py:216-222`).
- The archived classic distance helper used minimum duplicate weight, not additive (`dagua/layout/_archive/classic/_graph_distances.py:56-64`). This is a real ops-vs-archive divergence for weighted duplicates.
- Reference adapter builds a simple `networkx.Graph`, deduping parallel edges (`dagua/eval/competitors/sgd2_multi_competitor.py:331-339`), and its SciPy connectivity adjacency sums duplicate ones but is only used for reachability (`dagua/eval/competitors/sgd2_multi_competitor.py:312-326`).
- Expected RMSD impact: low for unweighted duplicate edges because shortest-path distances remain 1; medium for weighted duplicate edges because additive vs dedup/min semantics can change Dijkstra distances.

Disconnected components:

- Dagua supports disconnected graphs by building finite stress pairs only; unreachable distances become `inf` and are masked out (`dagua/layout/ops/sgd2_multi.py:448-453`, `dagua/layout/ops/sgd2_multi.py:478-483`).
- Reference adapter rejects disconnected graphs before running upstream (`dagua/eval/competitors/sgd2_multi_competitor.py:318-326`).
- This can affect benchmark OK counts: reference returns `pos=None` for disconnected graphs, so those graphs are not directly comparable.

Weighted edges:

- Dagua accepts `edge_weights` in the public pipeline and validates shape (`dagua/layout/ops/pipelines/sgd2_multi.py:297-303`), computes weighted Dijkstra when weights are present (`dagua/layout/ops/sgd2_multi.py:619-623`), and `graph_utils` uses float32 weights (`dagua/layout/ops/graph_utils.py:254-260`).
- Reference adapter ignores dagua edge weights entirely; it builds `data = np.ones(len(rows))` (`dagua/eval/competitors/sgd2_multi_competitor.py:312-317`) and never passes weights to `GD2`.
- Result: weighted-edge fidelity is not aligned.

Empty graph and one-node graph:

- Dagua public pipeline returns empty or zeros (`dagua/layout/ops/sgd2_multi.py:1999-2007`).
- Reference adapter builds an empty/one-node NetworkX graph and then SciPy shortest paths; exact behavior is adapter-visible only as disconnected rejection for nonfinite distances (`dagua/eval/competitors/sgd2_multi_competitor.py:318-326`). For `n=0`, CSR shape `(0,0)` and shortest_path behavior may be fragile; no explicit guard is present.
- Dagua canonical `s_gd2` fallback separately handles 0/1/no-edge cases (`dagua/layout/ops/pipelines/sgd2_multi.py:57-63`), but that fallback is gated off for variant comparisons.

No non-incident edge pairs:

- Dagua keeps crossing criterion active but returns zero loss for empty batches (`dagua/layout/ops/sgd2_multi.py:1320-1351`, `dagua/layout/ops/sgd2_multi.py:1606-1623`).
- Reference adapter removes crossing-related criteria, and if none remain, falls back to stress-only (`dagua/eval/competitors/sgd2_multi_competitor.py:377-398`).
- This is an objective-level edge-case divergence.

## 9. Numerical precision

Dagua:

- Positions are float32 (`dagua/layout/ops/sgd2_multi.py:2152-2156`).
- Returned positions are detached and cast to float32 (`dagua/layout/ops/sgd2_multi.py:2210`).
- Distance matrix is converted from NumPy to torch float32 (`dagua/layout/ops/sgd2_multi.py:448-453`).
- Edge weights are converted to float32 inside `graph_utils.build_undirected_adjacency` (`dagua/layout/ops/graph_utils.py:254-260`).
- Loss accumulation order follows insertion order of the Python criteria/schedules dict (`dagua/layout/ops/sgd2_multi.py:1801-1827`).

Reference adapter:

- SciPy shortest path returns floating arrays; adapter only uses this for connectivity rejection (`dagua/eval/competitors/sgd2_multi_competitor.py:318-326`).
- Patched stress uses whatever dtype upstream `D`, `W`, and `pos` have, with `PairwiseDistance` (`dagua/eval/competitors/sgd2_multi_competitor.py:147-186`).
- Patched ideal edge length creates target tensor without explicit device/dtype: `_torch.tensor([float(targetLengths[e]) for e in edges])` (`dagua/eval/competitors/sgd2_multi_competitor.py:133-139`). On CPU float32 positions this is probably float32? PyTorch default float dtype is normally float32, but device mismatch would occur if upstream used CUDA positions.
- Patched aspect creates target tensor without device/dtype (`dagua/eval/competitors/sgd2_multi_competitor.py:219-224`).

Precision residuals:

- Dagua clamps aspect ratio to `[1e-6, 1-1e-6]` (`dagua/layout/ops/sgd2_multi.py:1446-1452`); patched reference does not clamp singular ratio before BCE (`dagua/eval/competitors/sgd2_multi_competitor.py:219-224`).
- Dagua uses `torch.linalg.svd`; patched reference uses `torch.svd` (`dagua/layout/ops/sgd2_multi.py:1442-1443`, `dagua/eval/competitors/sgd2_multi_competitor.py:219-224`).
- Dagua stress uses upper-triangle finite pairs and mean over selected unordered pairs (`dagua/layout/ops/sgd2_multi.py:472-483`, `dagua/layout/ops/sgd2_multi.py:858-885`); reference patched no-sample stress flattens all ordered pairs including diagonal (`dagua/eval/competitors/sgd2_multi_competitor.py:159-173`). This is more than numerical if that path is used.

## 10. RNG semantics

Does dagua's torch seed produce the same sequence as reference RNG?

No. Same integer seed does not imply same random sequence across the two implementations.

Evidence:

- Dagua seeds only torch and CUDA (`dagua/layout/ops/sgd2_multi.py:264-279`).
- Dagua position init, cyclic sampler permutations, random fallback sampling, neighborhood negatives, and crossing detector initialization all consume torch RNG (`dagua/layout/ops/sgd2_multi.py:821-855`, `dagua/layout/ops/sgd2_multi.py:1082-1083`, `dagua/layout/ops/sgd2_multi.py:2152-2169`).
- Reference adapter seeds torch and NumPy (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`).
- Reference adapter monkeypatches stress no-sample path to use NumPy random (`dagua/eval/competitors/sgd2_multi_competitor.py:159-169`).
- Reference adapter monkeypatches aspect sample-size path to use NumPy random without replacement (`dagua/eval/competitors/sgd2_multi_competitor.py:204-211`).
- Reference adapter monkeypatches ideal-edge sample-size path to use Python `random.sample` but never seeds Python `random` (`dagua/eval/competitors/sgd2_multi_competitor.py:93-96`, `dagua/eval/competitors/sgd2_multi_competitor.py:126-127`, `dagua/eval/competitors/sgd2_multi_competitor.py:340-344`).

Implications:

- `stress_only`, `default`, `lr*`, and `batch*` residuals can be stochastic even when objective formulas align.
- `with_aspect` has a larger expected RNG mismatch because dagua samples aspect batches by torch cyclic randperm while reference patch samples NumPy without replacement if upstream does not provide explicit samples.
- `with_crossing` has the largest RNG-sensitive surface: sampler order, detector initialization, Adam updates, exact crossing labels around epsilon boundaries, and detector/position loss ordering all matter.

## 11. Edge-case bugs

1. Reference source availability bug: `SGD2MultiRef.available()` requires `/tmp/graph-drawing/gd2.py` (`dagua/eval/competitors/sgd2_multi_competitor.py:23-28`), but current upstream clone lacks `gd2.py`. A fresh clone of `github.com/tiga1231/graph-drawing` also lacks `criteria.py`, although the adapter imports and patches it (`dagua/eval/competitors/sgd2_multi_competitor.py:93-105`). This makes the reference adapter non-reproducible from the named upstream HEAD.
2. Python RNG not seeded in reference adapter: patched ideal edge length uses `_random.sample` (`dagua/eval/competitors/sgd2_multi_competitor.py:93-96`, `dagua/eval/competitors/sgd2_multi_competitor.py:126-127`), but only torch and NumPy are seeded (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`).
3. Dagua ops adjacency duplicate-weight semantics differ from the archived classic distance helper: ops import `build_undirected_adjacency` from `dagua.layout.ops.graph_utils` (`dagua/layout/ops/sgd2_multi.py:23-29`), which sums duplicate weights (`dagua/layout/ops/graph_utils.py:226-260`), while archived `_graph_distances.build_undirected_adjacency` keeps the minimum duplicate weight (`dagua/layout/_archive/classic/_graph_distances.py:56-64`). For weighted multiedges, this is a real shortest-path divergence.
4. Reference adapter computes `dist = shortest_path(...)` and uses it only to reject disconnected graphs (`dagua/eval/competitors/sgd2_multi_competitor.py:318-326`); this looks like dead or partial plumbing because the computed `dist` is not passed to `GD2`.
5. Dagua supports disconnected graphs by masking unreachable pairs (`dagua/layout/ops/sgd2_multi.py:478-483`), while reference adapter returns an error (`dagua/eval/competitors/sgd2_multi_competitor.py:318-326`). This is intentional in adapter behavior but creates missing comparable outputs.
6. Crossing no-pair behavior is objective-changing: dagua zeroes the crossing loss (`dagua/layout/ops/sgd2_multi.py:1320-1351`), but reference strips the criterion and may fall back to stress-only (`dagua/eval/competitors/sgd2_multi_competitor.py:377-398`).
7. Aspect ratio clamp mismatch: dagua clamps ratio to avoid BCE boundary singularities (`dagua/layout/ops/sgd2_multi.py:1446-1452`), while patched reference does not clamp (`dagua/eval/competitors/sgd2_multi_competitor.py:219-224`). This is small but can matter on near-collinear samples.
8. Public default mismatch: direct `layout_sgd2_multi_pipeline()` defaults to 10000 steps, lr 1.0, batch 16, grad clamp 4.0 (`dagua/layout/ops/pipelines/sgd2_multi.py:229-241`), while `sgd2_multi_ref` adapter defaults to max_iter 2000, lr 0.01, sample size 128, grad clamp 5.0 (`dagua/eval/competitors/sgd2_multi_competitor.py:345-375`). The benchmark variants align these, but direct callers are not reference-default aligned.
9. `s_gd2` fallback is a latent non-multicriteria substitution: the public pipeline may return canonical `s_gd2.layout` scaled by 100 for default native hyperparams and no explicit criteria (`dagua/layout/ops/pipelines/sgd2_multi.py:25-100`, `dagua/layout/ops/pipelines/sgd2_multi.py:305-346`). That fallback is not the `sgd2_multi_ref` GD2 engine and could surprise fidelity analysis outside the registered variants.

## 12. Ranked fix list

Ranked by expected RMSD / reproducibility impact for the current sgd2_multi family.

1. Restore or vendor the actual upstream `GD2` reference files used by the adapter.
   - Evidence: adapter requires `/tmp/graph-drawing/gd2.py` and `criteria.py` (`dagua/eval/competitors/sgd2_multi_competitor.py:23-28`, `dagua/eval/competitors/sgd2_multi_competitor.py:93-105`, `dagua/eval/competitors/sgd2_multi_competitor.py:307`), but current upstream HEAD lacks them.
   - Impact: high on reproducibility and diagnosis correctness; without this, future rounds cannot line-cite or replay the true reference.
   - Proposed fix size: M. Add a pinned reference acquisition path, submodule, artifact, or adapter error message with exact commit/source.

2. Align RNG sources or explicitly measure stochastic floor with independent RNGs.
   - Evidence: dagua seeds torch only (`dagua/layout/ops/sgd2_multi.py:264-279`); reference seeds torch and NumPy (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`); patched criteria use NumPy and Python random (`dagua/eval/competitors/sgd2_multi_competitor.py:159-169`, `dagua/eval/competitors/sgd2_multi_competitor.py:204-211`, `dagua/eval/competitors/sgd2_multi_competitor.py:126-127`).
   - Impact: high for all stochastic variants, especially `with_aspect` and `with_crossing`.
   - Proposed fix size: S for seeding Python `random` in reference adapter; M for matching sample streams in dagua.

3. Investigate and align sampler semantics: torch cyclic randperm versus upstream DataLoader / NumPy / Python random.
   - Evidence: dagua `_CyclicSampler` visits every index once per epoch and truncates batch size to total (`dagua/layout/ops/sgd2_multi.py:821-855`); reference adapter supplies `sample_sizes` but upstream loop is missing (`dagua/eval/competitors/sgd2_multi_competitor.py:361-364`, `dagua/eval/competitors/sgd2_multi_competitor.py:399-400`); patched fallback paths use different sampling.
   - Impact: high on Procrustes RMSD because different pair batches drive different optimization basins.
   - Proposed fix size: M after upstream source is restored; cannot safely implement before that.

4. Fix crossing no-pair objective mismatch.
   - Evidence: dagua zeroes crossing loss when no pairs (`dagua/layout/ops/sgd2_multi.py:1320-1351`); reference strips crossing criteria or falls back to stress-only (`dagua/eval/competitors/sgd2_multi_competitor.py:377-398`).
   - Impact: medium to high for small sparse graphs in `sgd2_multi_with_crossing`; current family has partial-match verdict (`eval_output/fidelity_report/report.md:83`).
   - Proposed fix size: S. Make dagua variant wrapper mirror reference stripping behavior for crossing-only/no-pair cases, or make adapter keep zero-loss behavior for comparison.

5. Align weighted/multiedge adjacency semantics.
   - Evidence: ops graph_utils sums duplicate weights (`dagua/layout/ops/graph_utils.py:226-260`); archived classic minimums duplicate weights (`dagua/layout/_archive/classic/_graph_distances.py:56-64`); reference adapter uses simple unweighted NetworkX graph (`dagua/eval/competitors/sgd2_multi_competitor.py:331-339`).
   - Impact: medium on weighted or parallel-edge graphs; low on simple unweighted graphs.
   - Proposed fix size: S to use local min-weight helper inside `sgd2_multi.py`; M if preserving shared graph_utils semantics elsewhere.

6. Align aspect-ratio SVD/clamp and sampler semantics.
   - Evidence: dagua `torch.linalg.svd` with clamp (`dagua/layout/ops/sgd2_multi.py:1442-1452`); patched reference `torch.svd` no clamp (`dagua/eval/competitors/sgd2_multi_competitor.py:219-224`); sampler mismatch described above.
   - Impact: medium for `sgd2_multi_with_aspect`, currently weak-equivalent with median RMSD 0.203 (`eval_output/fidelity_report/report.md:82`).
   - Proposed fix size: S for SVD/clamp toggle; M for sampler alignment.

7. Remove or gate the public `s_gd2` fallback during fidelity comparison.
   - Evidence: fallback is attempted only under strict default-native conditions (`dagua/layout/ops/pipelines/sgd2_multi.py:305-320`) and returns `s_gd2.layout` scaled by 100 (`dagua/layout/ops/pipelines/sgd2_multi.py:92-100`).
   - Impact: low for registered variants, medium for direct callers and future benchmarks that omit explicit criteria.
   - Proposed fix size: S. Add explicit `use_reference_fallback` or disable in `classic_sgd2_multi`.

8. Make direct defaults align or document that benchmark variants are the canonical reference-aligned entrypoint.
   - Evidence: public defaults differ from adapter defaults (`dagua/layout/ops/pipelines/sgd2_multi.py:229-241`, `dagua/eval/competitors/sgd2_multi_competitor.py:345-375`).
   - Impact: low for current variant sweep, medium for user expectations.
   - Proposed fix size: S documentation, M if changing public defaults.

## 13. Recommended Round 22+ fix scope

Recommended one-round bundle:

1. First, make the reference reproducible. Pin or vendor the actual upstream `GD2` source expected by `sgd2_multi_competitor.py`, or update `_SGD2_REPO`/imports to the real installed package. This is prerequisite work because the current adapter imports files not present in the named upstream clone (`dagua/eval/competitors/sgd2_multi_competitor.py:23-28`, `dagua/eval/competitors/sgd2_multi_competitor.py:93-105`, `dagua/eval/competitors/sgd2_multi_competitor.py:307`).
2. In the same round if time permits, seed Python `random` in `SGD2MultiRef.layout_with_variant` next to torch and NumPy (`dagua/eval/competitors/sgd2_multi_competitor.py:340-344`) because patched ideal-edge sampling uses Python random (`dagua/eval/competitors/sgd2_multi_competitor.py:126-127`). This is small, low-risk, and improves repeatability.
3. Re-run only the eight sgd2_multi variants after reference restoration. If source restoration changes reference behavior, regenerate the family verdict before touching dagua math.
4. If residual remains largest in `with_crossing`, align no-pair behavior and sampler ordering around crossing batches first (`dagua/layout/ops/sgd2_multi.py:1320-1351`, `dagua/eval/competitors/sgd2_multi_competitor.py:377-398`).
5. If residual remains largest in default/stress-only after source restoration, compare exact sampled pair streams. The most likely lever is cyclic torch randperm versus upstream DataLoader/random sampling (`dagua/layout/ops/sgd2_multi.py:821-855`, `dagua/eval/competitors/sgd2_multi_competitor.py:361-364`).

Do not start with broad objective rewrites. The variant hyperparameters are mostly aligned in `dagua/eval/variants.py:1607-1768`, and the current verdicts are already weak-equivalent except crossings. The highest-value next round is reference reproducibility plus RNG/sampler verification.

## Verification notes

- This report intentionally did not run `ruff`, `mypy`, or pytest because the task is diagnosis-only and no source code changed.
- The file should exceed 10KB and includes line references throughout.
