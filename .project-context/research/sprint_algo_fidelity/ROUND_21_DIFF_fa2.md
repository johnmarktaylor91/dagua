# Round 21 Diff: `classic_fa2` vs `fa2_ref`

Diagnosis-only adversarial diff for the ForceAtlas2 family. No source changes were made.

## 1. Files Read

### Dagua side

- `dagua/layout/ops/pipelines/fa2.py`: public `FA2Config`, pipeline wiring, and `layout_fa2_pipeline` wrapper. Key refs: `FA2Config` defaults at `dagua/layout/ops/pipelines/fa2.py:18-58`, pipeline op order at `dagua/layout/ops/pipelines/fa2.py:88-124`, wrapper/defaults at `dagua/layout/ops/pipelines/fa2.py:127-217`.
- `dagua/layout/ops/init.py`: FA2 input validation and initialization. Key refs: validation at `dagua/layout/ops/init.py:318-432`, Python-random float32 initialization at `dagua/layout/ops/init.py:633-732`.
- `dagua/layout/ops/preprocess.py`: unique undirected edge preparation, degree/mass, outbound compensation, speed state. Key refs: `FA2PrepareStateConfig` at `dagua/layout/ops/preprocess.py:1203-1207`, state preparation at `dagua/layout/ops/preprocess.py:1210-1335`.
- `dagua/layout/ops/force.py`: monolithic FA2 force step and related reusable FA2 ops. Key refs: mass helper at `dagua/layout/ops/force.py:362-383`, `FA2ForceStepConfig` at `dagua/layout/ops/force.py:1594-1661`, full iteration at `dagua/layout/ops/force.py:1664-2002`, reusable attraction at `dagua/layout/ops/force.py:2005-2143`, reusable gravity at `dagua/layout/ops/force.py:2146-2214`, reusable Barnes-Hut op at `dagua/layout/ops/force.py:2282-2405`, reusable speed op at `dagua/layout/ops/force.py:2682-2831`.
- `dagua/layout/engine.py`: dispatcher path for `algorithm="fa2"` into the pipeline. Key ref: `dagua/layout/engine.py:141-144`.
- `dagua/eval/variants.py`: FA2 variant definitions and dagua/reference parameter mapping. Key refs: default/gravity/scaling/strong/no-outbound/dissuade/linlog/BH/exact variants at `dagua/eval/variants.py:457-735`.
- `dagua/eval/competitors/base.py`: default `variant_param_names` behavior. Key refs: `dagua/eval/competitors/base.py:26-32`, default `layout_with_variant` at `dagua/eval/competitors/base.py:64-91`.
- `dagua/eval/competitors/classic_competitor.py`: dagua classic adapter, defaults, and variant forwarding. Key refs: `_ClassicBase.layout_with_variant` at `dagua/eval/competitors/classic_competitor.py:54-97`, `classic_fa2` spec at `dagua/eval/competitors/classic_competitor.py:164-168`, direct `ClassicFA2.layout` at `dagua/eval/competitors/classic_competitor.py:797-850`.
- `dagua/eval/competitors/fa2_competitor.py`: reference adapter. Key refs: load order at `dagua/eval/competitors/fa2_competitor.py:18-38`, accepted variant params at `dagua/eval/competitors/fa2_competitor.py:83-96`, seeding and graph conversion at `dagua/eval/competitors/fa2_competitor.py:151-185`, engine kwargs and variant application at `dagua/eval/competitors/fa2_competitor.py:186-222`.
- `dagua/layout/_archive/classic/fa2.py`: archived direct translation from `fa2_modified`, not live in the current pipeline but useful provenance. Key refs: archived init/loop at `dagua/layout/_archive/classic/fa2.py:42-198`, exact repulsion/gravity/attraction at `dagua/layout/_archive/classic/fa2.py:365-513`, archived Barnes-Hut at `dagua/layout/_archive/classic/fa2.py:516-768`, speed update at `dagua/layout/_archive/classic/fa2.py:771-840`.

### Reference side

- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2/forceatlas2.py`: live reference class selected by the adapter in this environment. Key refs: constructor defaults/validation at `.../fa2/forceatlas2.py:139-201`, init/mass/RNG/edges at `.../fa2/forceatlas2.py:281-390`, loop/vectorized dispatch at `.../fa2/forceatlas2.py:392-467`, vectorized loop at `.../fa2/forceatlas2.py:468-536`, Cython/loop backend at `.../fa2/forceatlas2.py:538-619`, NetworkX wrapper and `weight_attr` default at `.../fa2/forceatlas2.py:621-705`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2/fa2util.py`: pure Python fallback formulas for the live `fa2` package. Key refs: repulsion at `.../fa2/fa2util.py:88-133`, gravity at `.../fa2/fa2util.py:195-234`, attraction at `.../fa2/fa2util.py:237-294`, exact loop at `.../fa2/fa2util.py:357-393`, attraction loop at `.../fa2/fa2util.py:396-432`, Barnes-Hut region at `.../fa2/fa2util.py:435-525`, speed apply at `.../fa2/fa2util.py:528-604`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2/fa2util.pyx`: compiled Cython source corresponding to the live `.so`. Key refs: double-precision `Node2D` fields at `.../fa2/fa2util.pyx:24-42`, 2D exact force formulas at `.../fa2/fa2util.pyx:154-255`, batch repulsion/gravity/attraction at `.../fa2/fa2util.pyx:507-651`, Barnes-Hut region at `.../fa2/fa2util.pyx:658-837`, speed apply at `.../fa2/fa2util.pyx:844-941`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2/fa2util.cpython-311-x86_64-linux-gnu.so`: actual imported live `fa2.fa2util` backend. Verified via Python import; the `.pyx` above is the source with line refs.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2_modified/forceatlas2.py`: hinted reference package, but not selected first by `fa2_ref` when `fa2` imports. Key refs: constructor defaults and unimplemented flags at `.../fa2_modified/forceatlas2.py:48-83`, init/mass/RNG/edges at `.../fa2_modified/forceatlas2.py:85-144`, main loop at `.../fa2_modified/forceatlas2.py:162-259`, NetworkX adapter at `.../fa2_modified/forceatlas2.py:265-289`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2_modified/fa2util.py`: hinted reference pure Python formulas. Key refs: scalar node fields at `.../fa2_modified/fa2util.py:17-35`, formulas at `.../fa2_modified/fa2util.py:44-185`, attraction at `.../fa2_modified/fa2util.py:188-227`, Barnes-Hut region at `.../fa2_modified/fa2util.py:229-335`, speed apply at `.../fa2_modified/fa2util.py:337-427`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2_modified/fa2util.pxd`: Cython typing for `fa2_modified`. Key refs: double fields and typed function signatures at `.../fa2_modified/fa2util.pxd:18-30`, formula signatures at `.../fa2_modified/fa2util.pxd:36-88`, speed Cython locals at `.../fa2_modified/fa2util.pxd:119-135`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2_modified/fa2util.c`: generated C file skimmed for actual typed fields and compiled symbols. Key refs: C `double` node fields at `.../fa2_modified/fa2util.c:1654-1666`, typed function prototypes at `.../fa2_modified/fa2util.c:2681-2691`.

### Existing analysis

- `eval_output/fidelity_report/report.md`: current FA2 verdict table. Key refs: `fa2_default` weak equivalent RMSD 0.068 at `eval_output/fidelity_report/report.md:20-21`, `fa2_dissuade_hubs` partial match RMSD 0.103 at `eval_output/fidelity_report/report.md:22`, `fa2_exact` strong equivalent RMSD 0.054 at `eval_output/fidelity_report/report.md:23`, other FA2 variants at `eval_output/fidelity_report/report.md:24-30`.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`: sprint context. Key ref: deferred `fa2_dissuade_hubs` item at `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:270-271`.

## 2. Overall Pipeline Structure

### Dagua `classic_fa2`

The live dagua pipeline is a composable ops pipeline:

1. Validate public FA2 inputs with `ValidateFA2Inputs` (`dagua/layout/ops/pipelines/fa2.py:88-95`; validation body at `dagua/layout/ops/init.py:400-432`).
2. Install a `FixedSteps` convergence object, but actual convergence is fixed-count only (`dagua/layout/ops/pipelines/fa2.py:96`).
3. Initialize positions using `random.Random(problem.seed)` and store a `torch.float32` tensor (`dagua/layout/ops/pipelines/fa2.py:97`; init body at `dagua/layout/ops/init.py:707-732`).
4. Prepare unique undirected edges, optional summed weights, degree, mass, mean-mass compensation, and speed scalars (`dagua/layout/ops/pipelines/fa2.py:98-102`; body at `dagua/layout/ops/preprocess.py:1282-1335`).
5. Repeat one monolithic `FA2ForceStep` `steps` times (`dagua/layout/ops/pipelines/fa2.py:103-121`), computing repulsion, gravity, attraction, adaptive speed, and position update in one op (`dagua/layout/ops/force.py:1748-2002`).

The default callable `layout_fa2_pipeline` exposes `steps=100`, `barnes_hut=False`, and `outbound_attraction_distribution=True` (`dagua/layout/ops/pipelines/fa2.py:127-143`). The benchmark adapter overrides classic defaults to `steps=200`, `barnes_hut=True`, `barnes_hut_theta=1.2` (`dagua/eval/competitors/classic_competitor.py:164-168`), and variants then apply the per-variant kwargs (`dagua/eval/competitors/classic_competitor.py:81-97`).

### Reference `fa2_ref`

The adapter claims to support `fa2` or `fa2_modified`, but `_load_forceatlas2()` imports `fa2` first and only falls back to `fa2_modified` on `ImportError` (`dagua/eval/competitors/fa2_competitor.py:31-38`). In this environment, `_load_forceatlas2()` resolves to `fa2.forceatlas2.ForceAtlas2`, so the live reference for benchmark runs is the newer `fa2` package, not the hinted `fa2_modified` package.

The reference adapter flow:

1. Seed Python `random` and NumPy globals when a benchmark seed is provided (`dagua/eval/competitors/fa2_competitor.py:155-162`).
2. Short-circuit graphs with `num_nodes <= 1` to zeros (`dagua/eval/competitors/fa2_competitor.py:163-166`).
3. Convert the dagua graph to a `networkx.Graph`, skipping self-loops and preserving optional weights (`dagua/eval/competitors/fa2_competitor.py:168-185`).
4. Build `ForceAtlas2` kwargs. Adapter defaults are `outboundAttractionDistribution=True`, `edgeWeightInfluence=1.0`, `jitterTolerance=1.0`, `barnesHutOptimize=True`, `barnesHutTheta=1.2`, `scalingRatio=2.0`, `strongGravityMode=False`, `gravity=1.0`, `verbose=False` (`dagua/eval/competitors/fa2_competitor.py:186-196`).
5. Add `seed` to the engine for newer `fa2` (`dagua/eval/competitors/fa2_competitor.py:197-200`).
6. Route `iterations` into layout kwargs and all other variant params into engine kwargs (`dagua/eval/competitors/fa2_competitor.py:201-207`).
7. Filter unsupported kwargs by inspecting the class signature (`dagua/eval/competitors/fa2_competitor.py:209-216`).
8. Run `forceatlas2_networkx_layout` (`dagua/eval/competitors/fa2_competitor.py:218-222`) and copy positions into `torch.float32` output (`dagua/eval/competitors/fa2_competitor.py:224-230`).

Inside live `fa2`, `forceatlas2_networkx_layout` defaults `weight_attr=None`, documents that `None` means unweighted, and passes that value into `networkx.to_scipy_sparse_array(..., weight=weight_attr)` (`.../fa2/forceatlas2.py:621-683`). Live `fa2.forceatlas2()` then validates `iterations >= 1`, initializes speed, initializes nodes and edges, transforms weights, chooses backend, and runs either vectorized or Cython/loop backend (`.../fa2/forceatlas2.py:392-467`). With the installed compiled `.so`, backend `"auto"` uses the loop/Cython path (`.../fa2/forceatlas2.py:455-466`).

### Structural Match

The high-level algorithm is aligned: initialize random positions, compute mass from undirected adjacency, then repeat repulsion, gravity, attraction, adaptive speed, and position update. The exact-mode flow is especially close; this matches the current report where `fa2_exact` is `strong_equivalent` with median RMSD 0.054 (`eval_output/fidelity_report/report.md:23`).

The biggest structural divergences are:

- live reference package ambiguity: `fa2_ref` uses `fa2`, while the archived dagua translation and task hint target `fa2_modified` (`dagua/eval/competitors/fa2_competitor.py:31-38`; `dagua/layout/_archive/classic/fa2.py:1`);
- dagua stores all live tensors as `float32`, while reference Cython stores positions/forces/masses as C `double` (`dagua/layout/ops/init.py:727-731`; `.../fa2/fa2util.pyx:24-42`);
- Barnes-Hut implementation is reimplemented in dagua with NumPy traversal, but live `fa2` Cython uses `Region` with identity leaf exclusion and bucket order from bitmasks (`dagua/layout/ops/force.py:1750-1884`; `.../fa2/fa2util.pyx:658-837`);
- dagua has a `dissuade_hubs` option that the live reference adapter does not expose in `variant_param_names` and `fa2` has no constructor param for it (`dagua/eval/variants.py:632-656`; `dagua/eval/competitors/fa2_competitor.py:83-96`; live constructor at `.../fa2/forceatlas2.py:139-165`).

## 3. Energy / Loss / Objective

ForceAtlas2 here is not implemented as an explicit scalar energy minimization in either side. Both sides are force accumulators plus adaptive displacement.

### Repulsion

Dagua exact repulsion computes all-pairs deltas:

- `delta = pos.unsqueeze(1) - pos.unsqueeze(0)` (`dagua/layout/ops/force.py:1886`);
- `distance_sq = torch.cdist(pos, pos).square()` (`dagua/layout/ops/force.py:1887-1888`);
- valid factor `scaling_ratio * mass_i * mass_j / distance_sq` (`dagua/layout/ops/force.py:1889-1893`);
- force is sum over `delta * factor` (`dagua/layout/ops/force.py:1893`).

Live reference Cython 2D exact repulsion is equivalent in formula:

- `xDist = n1.x - n2.x`, `yDist = n1.y - n2.y` (`.../fa2/fa2util.pyx:154-157`);
- `factor = coefficient * n1.mass * n2.mass / distance2` (`.../fa2/fa2util.pyx:159-160`);
- symmetric updates to `n1.dx/n1.dy` and `n2.dx/n2.dy` (`.../fa2/fa2util.pyx:161-164`).

`fa2_modified` uses the same formula but names `distance = sqrt(...)` and divides by `distance / distance`, equivalent to distance squared (`.../fa2_modified/fa2util.py:50-59`).

Residual divergence: dagua computes a full distance matrix and sums each row in PyTorch order (`dagua/layout/ops/force.py:1886-1893`); reference applies pairwise updates in nested loops (`.../fa2/fa2util.pyx:515-522`). With `float32` dagua and C `double` reference, the formula matches but rounding/summation order does not.

### Gravity

Dagua normal gravity:

- `distance = torch.linalg.vector_norm(pos, dim=1)` (`dagua/layout/ops/force.py:1902`);
- factor `mass * gravity / distance` for nonzero distance (`dagua/layout/ops/force.py:1903-1906`);
- contribution `-pos * factor` (`dagua/layout/ops/force.py:1906`).

Live reference normal gravity:

- Cython computes `distance = sqrt(n.x * n.x + n.y * n.y)` (`.../fa2/fa2util.pyx:210-211`);
- factor `n.mass * g / distance` (`.../fa2/fa2util.pyx:213-215`);
- subtracts `n.x * factor` and `n.y * factor` (`.../fa2/fa2util.pyx:215-216`).

Dagua strong gravity:

- checks `valid = (pos[:, 0] != 0) & (pos[:, 1] != 0)` (`dagua/layout/ops/force.py:1897-1900`);
- factor `scaling_ratio * mass * gravity` (`dagua/layout/ops/force.py:1897-1900`).

Live `fa2` strong gravity:

- checks `if n.x != 0.0 or n.y != 0.0` (`.../fa2/fa2util.pyx:219-224`).

This is a real bug relative to live `fa2`: dagua and `fa2_modified` use the stricter `and` condition (`dagua/layout/ops/force.py:1897-1900`; `.../fa2_modified/fa2util.py:114-121`), while live `fa2` uses `or` (`.../fa2/fa2util.pyx:219-224`). This explains why `fa2_strong_gravity` is one of the worst FA2 variants in the current report, RMSD 0.177 (`eval_output/fidelity_report/report.md:30`). In random `[0, 1]` starts exact zeros are rare except single-node or axis-crossing after updates, so the impact is graph/trajectory-dependent rather than universal.

### Attraction

Dagua linear attraction:

- per undirected edge `delta = pos[source] - pos[target]` (`dagua/layout/ops/force.py:1909-1912`);
- base factor `-outbound_att_compensation` for linear mode (`dagua/layout/ops/force.py:1920-1926`);
- divide by source mass when outbound distribution is enabled (`dagua/layout/ops/force.py:1928-1929`);
- optional extra divide for `dissuade_hubs` (`dagua/layout/ops/force.py:1930-1931`);
- apply transformed weights (`dagua/layout/ops/force.py:1932-1938`);
- scatter add source `delta * factor`, target `-delta * factor` (`dagua/layout/ops/force.py:1940-1952`).

Live reference linear attraction:

- `xDist = n1.x - n2.x`, `yDist = n1.y - n2.y` (`.../fa2/fa2util.pyx:227-230`);
- factor `-coefficient * e` or `-coefficient * e / n1.mass` (`.../fa2/fa2util.pyx:231-235`);
- symmetric updates (`.../fa2/fa2util.pyx:235-238`);
- `apply_attraction` chooses weight `1`, `edge.weight`, or `pow(edge.weight, edgeWeightInfluence)` (`.../fa2/fa2util.pyx:599-607`, and ND/linlog branches at `.../fa2/fa2util.pyx:622-651`).

Dagua linlog attraction:

- clamps distance to `1e-6` and uses `-outbound_comp * log1p(distance) / distance` (`dagua/layout/ops/force.py:1913-1919`).

Live `fa2` linlog:

- computes `distance = sqrt(...)`;
- only applies if `distance > 0`;
- `log_factor = log(1.0 + distance) / distance`;
- factor includes edge weight and optional mass division (`.../fa2/fa2util.pyx:241-255`).

The linlog formula is aligned for nonzero distances, but dagua's clamp creates a nonzero attraction for coincident endpoints where the reference emits no attraction (`dagua/layout/ops/force.py:1913-1919`; `.../fa2/fa2util.pyx:241-255`). This is an edge-case divergence and can matter when random collisions or deterministic identical positions occur. `fa2_linlog` has RMSD 0.154 (`eval_output/fidelity_report/report.md:26`).

### Barnes-Hut Repulsion

Dagua Barnes-Hut builds a per-iteration nested `BarnesHutNode` class, converts `pos` and `mass` to NumPy arrays, creates a tree by mass-center quadrants, and either applies leaf exact forces or aggregate force if `(size^2 / dist_sq) < theta^2` (`dagua/layout/ops/force.py:1750-1884`).

Live `fa2` Cython `Region` stores `mass`, `massCenter`, `size`, `nodes`, `subregions`, and builds buckets with bitmasks (`.../fa2/fa2util.pyx:658-681`, `.../fa2/fa2util.pyx:736-775`). It accepts a region if `distance * theta > self.size` (`.../fa2/fa2util.pyx:790-799`), equivalent to `size^2 / distance^2 < theta^2`, then uses `linRepulsion_region_2d` (`.../fa2/fa2util.pyx:167-175`).

The acceptance formula is equivalent, but implementation details differ:

- dagua leaf nodes store `mass_value=0.0`, `mass_center=(0,0)` and exact leaf contributions come from `indices` (`dagua/layout/ops/force.py:1770-1778`, `dagua/layout/ops/force.py:1830-1847`);
- live `fa2` leaf `Region` stores the node's actual mass and mass center, but excludes identity with `self.nodes[0] is not n` before applying region force (`.../fa2/fa2util.pyx:691-700`, `.../fa2/fa2util.pyx:781-788`);
- dagua uses `pending.extend(reversed(node.children))` to emulate traversal order (`dagua/layout/ops/force.py:1881`);
- live `fa2` recursively iterates `for subregion in self.subregions` in build order (`.../fa2/fa2util.pyx:797-799`), with bucket order derived from bitmask `[x>=center, y>=center]` (`.../fa2/fa2util.pyx:748-755`).

The order mismatch is a plausible residual for `fa2_default` / `fa2_barnes_hut`, both weak-equivalent at RMSD 0.068/0.069 (`eval_output/fidelity_report/report.md:20-21`).

## 4. Force / Gradient Computation

Both implementations are imperative force accumulators; there is no autograd gradient in FA2.

Dagua accumulates a full `force` tensor each step (`dagua/layout/ops/force.py:1748`), adds repulsion (`dagua/layout/ops/force.py:1750-1894`), gravity (`dagua/layout/ops/force.py:1896-1907`), attraction (`dagua/layout/ops/force.py:1909-1952`), then applies adaptive displacement (`dagua/layout/ops/force.py:1954-1999`).

Live `fa2` loop resets node force fields, applies Barnes-Hut or exact repulsion, applies gravity, applies attraction, then calls `adjustSpeedAndApplyForces` (`.../fa2/forceatlas2.py:554-601`). The Cython backend updates mutable `Node2D` double fields in place (`.../fa2/fa2util.pyx:24-42`, `.../fa2/fa2util.pyx:507-651`).

Primary force-computation differences:

- dagua exact repulsion is vectorized all-pairs with row-wise sum (`dagua/layout/ops/force.py:1886-1893`); reference exact repulsion is pairwise mutation loop (`.../fa2/fa2util.pyx:515-522`);
- dagua attraction is scatter-add tensor accumulation (`dagua/layout/ops/force.py:1940-1952`); reference attraction loops edges in sparse nonzero order (`.../fa2/forceatlas2.py:378-388`, `.../fa2/fa2util.pyx:599-607`);
- dagua includes `dissuade_hubs` as a second source-mass division (`dagua/layout/ops/force.py:1930-1931`); live reference does not have this param;
- dagua does not implement `adjustSizes`, `normalizeEdgeWeights`, `invertedEdgeWeightsMode`, `dim`, or backend selection in the public pipeline (`dagua/layout/ops/pipelines/fa2.py:127-143`; live reference constructor at `.../fa2/forceatlas2.py:139-165`).

## 5. Initialization

Dagua:

- special-cases `N=0` to empty float32 zeros and `N=1` to one zero position (`dagua/layout/ops/init.py:707-720`);
- for `N>1`, uses `rng = random.Random(problem.seed)` (`dagua/layout/ops/init.py:722`);
- draws `[rng.random() for _ in range(position_dim)]` per node (`dagua/layout/ops/init.py:723-726`);
- stores `torch.tensor(..., dtype=torch.float32)` on `problem.edge_index.device` (`dagua/layout/ops/init.py:727-731`).

Live `fa2`:

- uses `rng = random.Random(self.seed)` (`.../fa2/forceatlas2.py:347-348`);
- for dim 2 draws `n.x = rng.random(); n.y = rng.random()` per node (`.../fa2/forceatlas2.py:358-362`);
- stores Cython `double` fields in `Node2D` (`.../fa2/fa2util.pyx:24-42`);
- does not accept `iterations=0`; it raises if `iterations < 1` (`.../fa2/forceatlas2.py:421-422`).

`fa2_modified`:

- uses global `random.random()` rather than `random.Random(self.seed)` because it has no seed constructor param (`.../fa2_modified/forceatlas2.py:124-127`);
- the dagua adapter seeds global Python and NumPy before reference calls to handle this older package (`dagua/eval/competitors/fa2_competitor.py:155-162`).

The random sequence itself matches live `fa2` for `N>1`, because both use Python's Mersenne Twister via `random.Random(seed)`, drawing x then y per node (`dagua/layout/ops/init.py:722-726`; `.../fa2/forceatlas2.py:347-362`). The dtype boundary differs immediately: dagua truncates to float32, reference keeps double throughout and only the adapter output is copied to float32 (`dagua/eval/competitors/fa2_competitor.py:224-227`).

## 6. Iteration / Convergence

Dagua:

- accepts `steps >= 0`; zero steps produce initialized positions after validation/prep and no force repeats (`dagua/layout/ops/pipelines/fa2.py:84-86`, `dagua/layout/ops/pipelines/fa2.py:103-121`);
- benchmark variants use `steps=200` (`dagua/eval/variants.py:461-469`, repeated through `dagua/eval/variants.py:483-731`);
- no convergence threshold; fixed count only via `Repeat(n=resolved.steps)` (`dagua/layout/ops/pipelines/fa2.py:103-121`).

Reference:

- live `fa2.forceatlas2()` requires `iterations` to be a positive integer (`.../fa2/forceatlas2.py:421-422`);
- initializes `speed=1.0`, `speedEfficiency=1.0` (`.../fa2/forceatlas2.py:423-425`);
- loops exactly `range(iterations)` (`.../fa2/forceatlas2.py:551-554`);
- no convergence stop, only fixed count.

Adaptive speed formula is aligned:

- dagua computes swinging, effective traction, estimated optimal jitter, min/max jitter, speed efficiency, target speed, speed rise cap, and factor at `dagua/layout/ops/force.py:1954-1999`;
- live Cython computes the same at `.../fa2/fa2util.pyx:844-941`;
- `fa2_modified` equivalent is at `.../fa2_modified/fa2util.py:337-427`.

Subtle difference: live `fa2` and pure Python fallback guard the jitter calculation for empty/zero-traction cases (`.../fa2/fa2util.py:565-570`; Cython at `.../fa2/fa2util.pyx:885-890`). Dagua special-cases empty positions before speed update (`dagua/layout/ops/force.py:1743-1746`), but if `N>0` and `total_effective_traction == 0`, dagua still computes `jt` via a formula whose inner term becomes zero (`dagua/layout/ops/force.py:1958-1969`). This is usually equivalent to `jitterTolerance * minJT`, but it is a separate edge-case path.

## 7. Hyperparameter Alignment Table

| Param / behavior | Dagua default / variant value | Reference default / variant value | Match? | Notes |
| --- | --- | --- | --- | --- |
| `steps` / `iterations` | Pipeline default 100 (`dagua/layout/ops/pipelines/fa2.py:48`); benchmark/spec variants use 200 (`dagua/eval/competitors/classic_competitor.py:164-168`, `dagua/eval/variants.py:461-472`) | Adapter default layout kwargs 100 (`dagua/eval/competitors/fa2_competitor.py:201`); variants use 200 (`dagua/eval/variants.py:471-477`) | Y in benchmark variants | Direct `layout_fa2_pipeline` default differs from benchmark default, but pairwise variants align. |
| `gravity` | 1.0 (`dagua/layout/ops/pipelines/fa2.py:49`) | Adapter 1.0 (`dagua/eval/competitors/fa2_competitor.py:194`); live constructor 1.0 (`.../fa2/forceatlas2.py:154-157`) | Y | Variants gravity0/gravity2 align at `dagua/eval/variants.py:486-527`. |
| `scaling_ratio` / `scalingRatio` | 2.0 (`dagua/layout/ops/pipelines/fa2.py:50`) | 2.0 (`dagua/eval/competitors/fa2_competitor.py:192`; `.../fa2/forceatlas2.py:154`) | Y | scaling1/scaling4 variants align at `dagua/eval/variants.py:533-577`. |
| `linlog` / `linLogMode` | False default (`dagua/layout/ops/pipelines/fa2.py:51`) | False live default (`.../fa2/forceatlas2.py:142-144`) | Mostly Y | Live `fa2` implements linlog (`.../fa2/fa2util.pyx:241-255`); `fa2_modified` asserts `linLogMode == False` (`.../fa2_modified/forceatlas2.py:69-71`). Adapter load order makes live `fa2` relevant. |
| `strong_gravity` / `strongGravityMode` | False default (`dagua/layout/ops/pipelines/fa2.py:52`) | False (`dagua/eval/competitors/fa2_competitor.py:193`; `.../fa2/forceatlas2.py:155`) | Param Y, formula N | Formula axis condition differs: dagua `and` (`dagua/layout/ops/force.py:1897-1900`), live `fa2` `or` (`.../fa2/fa2util.pyx:219-224`). |
| `outbound_attraction_distribution` / `outboundAttractionDistribution` | Pipeline default True (`dagua/layout/ops/pipelines/fa2.py:53`) | Live constructor default False (`.../fa2/forceatlas2.py:141-143`), but adapter default True (`dagua/eval/competitors/fa2_competitor.py:186-188`) | Y in adapter/variants | Direct package default differs, benchmark adapter aligns to dagua. |
| `dissuade_hubs` | False default; True variant (`dagua/layout/ops/pipelines/fa2.py:54`, `dagua/eval/variants.py:632-656`) | No accepted live constructor param (`.../fa2/forceatlas2.py:139-165`); adapter param set excludes it (`dagua/eval/competitors/fa2_competitor.py:83-96`) | N | This variant is intentionally not equivalent; current report partial_match RMSD 0.103 (`eval_output/fidelity_report/report.md:22`). |
| `edge_weight_influence` / `edgeWeightInfluence` | 1.0 (`dagua/layout/ops/pipelines/fa2.py:55`) | 1.0 (`dagua/eval/competitors/fa2_competitor.py:188`; `.../fa2/forceatlas2.py:145`) | Y | Dagua variants do not sweep it. Formula aligned for 0/1/other (`dagua/layout/ops/force.py:1932-1938`; `.../fa2/fa2util.pyx:599-607`). |
| `barnes_hut` / `barnesHutOptimize` | Pipeline default False (`dagua/layout/ops/pipelines/fa2.py:56`); classic spec/variants default True (`dagua/eval/competitors/classic_competitor.py:164-168`) | Live constructor and adapter default True (`.../fa2/forceatlas2.py:149-151`; `dagua/eval/competitors/fa2_competitor.py:190`) | Y in benchmark default | Exact variant aligns to False at `dagua/eval/variants.py:713-732`. |
| `barnes_hut_theta` / `barnesHutTheta` | 1.2 (`dagua/layout/ops/pipelines/fa2.py:57`) | 1.2 (`dagua/eval/competitors/fa2_competitor.py:191`; `.../fa2/forceatlas2.py:150-151`) | Y | Validation positive both sides (`dagua/layout/ops/init.py:404-405`; `.../fa2/forceatlas2.py:175-176`). |
| `jitter_tolerance` / `jitterTolerance` | 1.0 in config (`dagua/layout/ops/pipelines/fa2.py:58`) | 1.0 (`dagua/eval/competitors/fa2_competitor.py:189`; `.../fa2/forceatlas2.py:149`) | Y | Not exposed through `layout_fa2_pipeline` wrapper args, so variants cannot tune it (`dagua/layout/ops/pipelines/fa2.py:127-143`). |
| `adjustSizes` | Not exposed; effectively False | False default (`.../fa2/forceatlas2.py:143-145`) | Y for current variants | Dagua ignores `node_sizes` (`dagua/layout/ops/pipelines/fa2.py:192`). |
| `nodeSize` / size attr | Not exposed | `fa2_modified` has `nodeSize=1.0` (`.../fa2_modified/forceatlas2.py:61-66`); live `fa2` has size support via wrappers/constructor docs (`.../fa2/forceatlas2.py:98-107`) | Y by omission | Current adapter does not enable sizes. |
| `normalizeEdgeWeights` | Not exposed | False default (`.../fa2/forceatlas2.py:145-147`) | Y by omission | Reference-only capability unused. |
| `invertedEdgeWeightsMode` | Not exposed | False default (`.../fa2/forceatlas2.py:146-148`) | Y by omission | Reference-only capability unused. |
| `dim` | Hard-coded 2 (`dagua/layout/ops/init.py:633-643`) | 2 default (`.../fa2/forceatlas2.py:157-159`) | Y | Dagua only handles `[N,2]`. |
| `backend` | Not applicable; torch/NumPy implementation | `"auto"` default, selects compiled Cython when `.so` present (`.../fa2/forceatlas2.py:159-160`, `.../fa2/forceatlas2.py:455-466`) | N | Dagua does not use reference Cython loop/summation. |
| `seed` | Wrapper default 42 (`dagua/layout/ops/pipelines/fa2.py:131-132`) | Adapter passes seed if provided (`dagua/eval/competitors/fa2_competitor.py:197-200`); live default None (`.../fa2/forceatlas2.py:161-163`) | Y under benchmark seeded calls | Adapter also seeds globals for old package (`dagua/eval/competitors/fa2_competitor.py:155-162`). |

## 8. Edge Cases

### Self-loops

Dagua drops self-loops before unique edge/mass construction (`dagua/layout/ops/preprocess.py:1287-1293`). Therefore self-loops do not affect edges, degree, or mass.

Reference adapter also skips self-loops when building NetworkX (`dagua/eval/competitors/fa2_competitor.py:173-184`). Live `fa2` itself would warn that self-loops inflate mass but exclude edges if they are present in the adjacency matrix (`.../fa2/forceatlas2.py:333-345`), but the adapter prevents that path.

Conclusion: aligned for benchmark adapter input; not aligned if calling live `fa2` directly on a matrix with diagonal entries.

### Multi-edges / duplicate directed edges

Dagua collapses unique undirected pairs with `torch.unique` and sums weights if provided (`dagua/layout/ops/preprocess.py:1293-1309`). If no weights are provided, duplicates do not increase degree or attraction.

Reference adapter builds a simple `nx.Graph` and calls `add_edge`; duplicate edges overwrite the same edge and do not create multiplicity (`dagua/eval/competitors/fa2_competitor.py:171-184`). If weighted duplicate edges are added, NetworkX keeps the last value, not a sum. This is a divergence for weighted multi-edge input: dagua sums duplicate directed/parallel weights, reference adapter overwrites by last insertion.

Unweighted duplicate behavior is mostly aligned: both reduce to one undirected pair.

### Disconnected components

Both sides apply gravity toward origin to every node and do not separately pack connected components. Dagua gravity is at `dagua/layout/ops/force.py:1896-1907`; reference gravity is invoked every loop at `.../fa2/forceatlas2.py:583-586` and implemented at `.../fa2/fa2util.pyx:210-224`. Disconnected components are handled implicitly by repulsion/gravity, not by component logic.

### Weighted edges

Dagua accepts optional `edge_weights`, filters self-loops, and sums duplicate undirected weights (`dagua/layout/ops/preprocess.py:1300-1309`). It applies `edge_weight_influence`: 0 maps all to one, 1 uses raw/summed weights, other uses power (`dagua/layout/ops/force.py:1932-1938`).

Reference adapter copies dagua weights into NetworkX edge attributes named `"weight"` (`dagua/eval/competitors/fa2_competitor.py:176-182`) but calls `forceatlas2_networkx_layout` without `weight_attr` (`dagua/eval/competitors/fa2_competitor.py:219-222`). Live `fa2` documents `weight_attr=None` as unweighted and passes it through to `networkx.to_scipy_sparse_array` (`.../fa2/forceatlas2.py:621-683`). `fa2_modified` has the same default shape (`.../fa2_modified/forceatlas2.py:265-283`). Therefore weighted-edge parity is not just at risk: for live `fa2_ref`, the adapter ignores graph edge weights unless a variant or future adapter change passes `weight_attr="weight"`.

This is potentially high impact for weighted graph tests: dagua may use supplied weights while reference may run unweighted unless the package default chooses `"weight"` internally.

### Empty graph

Dagua accepts `num_nodes=0` and returns empty positions through init and force short-circuit (`dagua/layout/ops/init.py:707-713`, `dagua/layout/ops/force.py:1743-1746`). Reference adapter short-circuits `graph.num_nodes <= 1` to zeros (`dagua/eval/competitors/fa2_competitor.py:163-166`). Aligned at adapter level.

### Single-node graph

Dagua returns a single zero before RNG (`dagua/layout/ops/init.py:714-720`). Reference adapter also returns a single zero (`dagua/eval/competitors/fa2_competitor.py:163-166`). Aligned at adapter level.

### Zero iterations

Dagua accepts `steps=0` (`dagua/layout/ops/pipelines/fa2.py:84-86`) and returns the initialized layout after no repeats. Live `fa2` rejects `iterations < 1` (`.../fa2/forceatlas2.py:421-422`). Current variants use 200, so no benchmark impact unless a future zero-step variant is introduced.

## 9. Numerical Precision

Dagua uses `torch.float32` for initialized positions (`dagua/layout/ops/init.py:727-731`), mass/degree/weights (`dagua/layout/ops/preprocess.py:1303-1325`), and force buffers (`dagua/layout/ops/force.py:1748`). Barnes-Hut converts float32 positions/masses to NumPy arrays and accumulates `force_np` in float64, but converts back to `pos.dtype` (`dagua/layout/ops/force.py:1751-1754`, `dagua/layout/ops/force.py:1884`).

Live reference Cython uses double fields for `Node2D` positions, forces, old forces, mass, and size (`.../fa2/fa2util.pyx:24-42`). `fa2_modified` Cython typing is also double (`.../fa2_modified/fa2util.pxd:18-30`) and generated C confirms double fields (`.../fa2_modified/fa2util.c:1654-1666`).

Boundary effects:

- initial Python RNG draws are double precision in reference but rounded to float32 in dagua before the first force calculation (`dagua/layout/ops/init.py:727-731`; `.../fa2/forceatlas2.py:358-362`; `.../fa2/fa2util.pyx:24-42`);
- exact repulsion summation order differs: dagua vectorized row sums versus reference pairwise mutation (`dagua/layout/ops/force.py:1886-1893`; `.../fa2/fa2util.pyx:515-522`);
- attraction summation order differs: dagua scatter-add versus reference edge loop (`dagua/layout/ops/force.py:1940-1952`; `.../fa2/fa2util.pyx:599-607`);
- speed totals use float32 tensor reductions in dagua before converting to Python floats (`dagua/layout/ops/force.py:1954-1957`), while reference totals are accumulated in C double (`.../fa2/fa2util.pyx:846-865`);
- final reference adapter converts double coordinates to `torch.float32` only at the end (`dagua/eval/competitors/fa2_competitor.py:224-227`).

This precision split is the most plausible floor for residual `fa2_exact` RMSD 0.054 despite formula alignment (`eval_output/fidelity_report/report.md:23`).

## 10. RNG Semantics

Dagua's current FA2 initializer does not use torch RNG. It uses Python `random.Random(problem.seed)` (`dagua/layout/ops/init.py:722-726`). Therefore the specific question "does dagua's torch seed produce same sequence as reference's RNG?" is: no, torch seed is irrelevant for FA2 initialization unless some outer code creates `problem.seed` from a torch-seeded process.

For seeded benchmark runs, dagua and live `fa2` both use Python `random.Random(seed)` and draw x/y per node in the same order (`dagua/layout/ops/init.py:722-726`; `.../fa2/forceatlas2.py:347-362`). This matches RNG sequence before dtype conversion.

For `fa2_modified`, reference uses module-global `random.random()` (`.../fa2_modified/forceatlas2.py:124-127`), and the adapter seeds global Python RNG before constructing the engine (`dagua/eval/competitors/fa2_competitor.py:155-162`). That should match sequence if no intervening Python-random draws occur. Live `fa2` is safer because the engine owns an isolated `random.Random(self.seed)` (`.../fa2/forceatlas2.py:347-348`).

Potential remaining RNG risks:

- if `seed=None`, dagua defaults to 42 in the wrapper (`dagua/layout/ops/pipelines/fa2.py:131-132`) and classic adapter `_layout_seed` returns 42 (`dagua/eval/competitors/classic_competitor.py:29-42`), while live reference constructor default is `seed=None` (`.../fa2/forceatlas2.py:161-163`) unless the adapter passes a seed (`dagua/eval/competitors/fa2_competitor.py:197-200`);
- benchmark harness likely supplies a runtime seed, but direct calls with `seed=None` are not semantically aligned.

## 11. Edge-Case Bugs

1. **Live reference package mismatch.** The task hint says `fa2_modified`, but adapter load order selects `fa2` first (`dagua/eval/competitors/fa2_competitor.py:31-38`). The live package implements linlog and uses `or` for strong-gravity nonzero tests (`.../fa2/fa2util.pyx:219-224`, `.../fa2/fa2util.pyx:241-255`), while `fa2_modified` asserts linlog false and uses `and` (`.../fa2_modified/forceatlas2.py:69-71`, `.../fa2_modified/fa2util.py:114-121`). Any dagua code written against `fa2_modified` can be subtly wrong for actual `fa2_ref`.
2. **Strong gravity axis condition is wrong for live `fa2`.** Dagua skips strong gravity unless both x and y are nonzero (`dagua/layout/ops/force.py:1897-1900`); live `fa2` skips only when all coordinates are zero (`.../fa2/fa2util.pyx:219-224`). This is a wrong condition for axis-aligned nodes.
3. **`dissuade_hubs` variant has no reference equivalent.** Dagua applies an extra divide by source mass (`dagua/layout/ops/force.py:1930-1931`), but reference adapter accepted params exclude it (`dagua/eval/competitors/fa2_competitor.py:83-96`) and live constructor has no param (`.../fa2/forceatlas2.py:139-165`). Current sprint notes already flag this as low ROI but real (`.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:270-271`).
4. **Weighted duplicate semantics diverge.** Dagua sums duplicate undirected weights (`dagua/layout/ops/preprocess.py:1303-1309`); NetworkX simple graph edge insertion overwrites/merges at adapter level (`dagua/eval/competitors/fa2_competitor.py:171-184`). Impact depends on graph suite weighted duplicates.
5. **Reference adapter ignores weights in live `fa2_ref`.** It writes `weight` attributes (`dagua/eval/competitors/fa2_competitor.py:176-182`) but calls `forceatlas2_networkx_layout(nx_graph, **layout_kwargs)` without `weight_attr` (`dagua/eval/competitors/fa2_competitor.py:219-222`). Live `fa2` says `weight_attr=None` means unweighted and passes it to NetworkX sparse conversion (`.../fa2/forceatlas2.py:621-683`), so reference is unweighted while dagua can be weighted.
6. **Linlog coincident-edge clamp differs.** Dagua clamps zero distance to `1e-6` and applies an attraction factor (`dagua/layout/ops/force.py:1913-1919`); live reference applies no linlog attraction when `distance == 0` (`.../fa2/fa2util.pyx:241-255`).
7. **Zero iterations accepted in dagua, rejected in live reference.** Dagua allows `steps=0` (`dagua/layout/ops/pipelines/fa2.py:84-86`); live reference rejects `iterations < 1` (`.../fa2/forceatlas2.py:421-422`).
8. **Classic FA2 variant warnings are noisy/misleading.** `ClassicFA2` does not define `variant_param_names`, so `_ClassicBase.layout_with_variant` will warn all FA2 variant params as unrecognized while still applying them (`dagua/eval/competitors/base.py:32`; `dagua/eval/competitors/classic_competitor.py:81-97`; missing override at `dagua/eval/competitors/classic_competitor.py:797-803`). This is not RMSD-impacting but can obscure real param mistakes.
9. **Dagua's Barnes-Hut is aligned to old `fa2_modified` comments, not necessarily live `fa2` Cython.** Archived file says direct translation from `fa2_modified` (`dagua/layout/_archive/classic/fa2.py:1`) and strong gravity comment explicitly references the old `and` condition (`dagua/layout/_archive/classic/fa2.py:421-427`), while live `fa2` changed semantics in some places (`.../fa2/fa2util.pyx:219-224`).
10. **Float32 trajectory floor.** Not a bug if intentional for GPU, but it prevents bit-level parity with C double reference (`dagua/layout/ops/init.py:727-731`; `.../fa2/fa2util.pyx:24-42`).

## 12. Ranked Fix List

1. **Run dagua FA2 internal state in float64 for fidelity mode.**
   Expected RMSD impact: high for residual exact-mode floor; likely affects every FA2 variant.
   Evidence: dagua initializes and computes in float32 (`dagua/layout/ops/init.py:727-731`; `dagua/layout/ops/preprocess.py:1303-1325`; `dagua/layout/ops/force.py:1748-1999`), while live reference uses C double (`.../fa2/fa2util.pyx:24-42`, `.../fa2/fa2util.pyx:844-941`).
   Proposed fix: add a dtype/fidelity config for FA2 initialization, prepared masses, force tensors, and output conversion. Keep default float32 for GPU performance unless fidelity mode is active.
   Size estimate: M (touch init/preprocess/force/pipeline tests).

2. **Align strong-gravity nonzero condition to live `fa2` (`or`, not `and`) or pin reference to `fa2_modified`.**
   Expected RMSD impact: high for `fa2_strong_gravity` (current RMSD 0.177 at `eval_output/fidelity_report/report.md:30`), low elsewhere.
   Evidence: dagua `valid = (x != 0) & (y != 0)` (`dagua/layout/ops/force.py:1897-1900`), live reference `if n.x != 0.0 or n.y != 0.0` (`.../fa2/fa2util.pyx:219-224`).
   Proposed fix: if the target is actual adapter behavior, change dagua to `|`; if the target is `fa2_modified`, change `_load_forceatlas2()` order or explicitly import `fa2_modified`.
   Size estimate: S.

3. **Resolve `fa2_ref` package target explicitly.**
   Expected RMSD impact: high for linlog/strong-gravity interpretability; may change reference baselines.
   Evidence: adapter uses `fa2` first (`dagua/eval/competitors/fa2_competitor.py:31-38`) despite task/reference hint preferring `fa2_modified`; live `fa2` implements features `fa2_modified` lacks (`.../fa2/forceatlas2.py:139-165`; `.../fa2_modified/forceatlas2.py:69-71`).
   Proposed fix: choose one target: either rename current comparator to `fa2_ref` = live `fa2` and add `fa2_modified_ref`, or invert load order to match the sprint hint. Update reports so "reference" is reproducible.
   Size estimate: S/M depending on benchmark manifest updates.

4. **Make Barnes-Hut traversal and tree construction match live `fa2` exactly.**
   Expected RMSD impact: medium for `fa2_default` / `fa2_barnes_hut` weak-equivalent variants (RMSD 0.068/0.069 at `eval_output/fidelity_report/report.md:20-21`).
   Evidence: dagua custom tree/traversal at `dagua/layout/ops/force.py:1750-1884`; live Cython region/bucket/traversal at `.../fa2/fa2util.pyx:658-837`.
   Proposed fix: port live `fa2` 2D `Region` bucket order and leaf semantics into a testable helper, or call the archived translation only after updating it from live `fa2`. Include single-iteration BH golden tests.
   Size estimate: M/L.

5. **Weight handling fix: pass `weight_attr="weight"` to reference or ignore weights in dagua for FA2 fidelity pairs.**
   Expected RMSD impact: high on weighted graphs, zero on unweighted suite items.
   Evidence: adapter sets weights on NetworkX edges (`dagua/eval/competitors/fa2_competitor.py:176-182`) but does not pass `weight_attr` to the layout call (`dagua/eval/competitors/fa2_competitor.py:219-222`); live `fa2` treats `weight_attr=None` as unweighted (`.../fa2/forceatlas2.py:621-683`); dagua applies edge weights when present (`dagua/layout/ops/force.py:1932-1938`).
   Proposed fix: pass `weight_attr="weight"` explicitly or strip dagua weights in paired tests.
   Size estimate: S.

6. **Remove or remap the `dissuade_hubs` pair.**
   Expected RMSD impact: medium only for the partial-match `fa2_dissuade_hubs` variant (RMSD 0.103 at `eval_output/fidelity_report/report.md:22`).
   Evidence: dagua has extra source-mass division (`dagua/layout/ops/force.py:1930-1931`); reference has no accepted param (`dagua/eval/competitors/fa2_competitor.py:83-96`).
   Proposed fix: either mark this variant intentionally incomparable, add a custom reference shim that applies the same formula, or drop it from dagua-vs-reference scoring.
   Size estimate: S.

7. **Match reference zero-distance linlog behavior.**
   Expected RMSD impact: low/medium for linlog, mostly edge-case.
   Evidence: dagua clamp at `dagua/layout/ops/force.py:1913-1919`; reference `if distance > 0` at `.../fa2/fa2util.pyx:241-255`.
   Proposed fix: compute linlog factor only where distance > 0 instead of clamping for FA2 fidelity mode.
   Size estimate: S.

8. **Add `ClassicFA2.variant_param_names`.**
   Expected RMSD impact: none, diagnostics impact medium.
   Evidence: base default is empty (`dagua/eval/competitors/base.py:32`); FA2 class has no override (`dagua/eval/competitors/classic_competitor.py:797-803`) while variants pass many params (`dagua/eval/variants.py:457-735`).
   Proposed fix: define names matching `layout_fa2_pipeline` kwargs.
   Size estimate: XS.

9. **Align weighted multi-edge semantics.**
   Expected RMSD impact: low unless weighted duplicate graphs are in the fidelity suite.
   Evidence: dagua sums duplicate weights (`dagua/layout/ops/preprocess.py:1303-1309`); adapter uses `nx.Graph.add_edge` (`dagua/eval/competitors/fa2_competitor.py:171-184`).
   Proposed fix: either make dagua keep last duplicate weight in fidelity mode, or make reference use a multigraph/explicit adjacency with summed weights.
   Size estimate: S/M.

10. **Decide `steps=0` parity.**
    Expected RMSD impact: none for current variants.
    Evidence: dagua accepts zero steps (`dagua/layout/ops/pipelines/fa2.py:84-86`); live reference rejects zero iterations (`.../fa2/forceatlas2.py:421-422`).
    Proposed fix: add validation parity in paired benchmark paths only, or leave as public API difference.
    Size estimate: XS.

## 13. Recommended Round 22+ Fix Scope

Recommended one-round bundle, ordered for maximum signal without turning into a broad rewrite:

1. **First decide and document the reference package.** The current adapter uses live `fa2`, not `fa2_modified` (`dagua/eval/competitors/fa2_competitor.py:31-38`). Round 22 should either explicitly pin `fa2_ref` to `fa2_modified` or update dagua parity targets to live `fa2`. Without this, strong-gravity and linlog "fixes" can move toward the wrong reference.
2. **If keeping live `fa2`, fix strong gravity to `or`.** This is the cleanest likely improvement for `fa2_strong_gravity` and a single formula mismatch (`dagua/layout/ops/force.py:1897-1900`; `.../fa2/fa2util.pyx:219-224`).
3. **Add a fidelity dtype path for FA2 exact mode.** This targets the residual floor in `fa2_exact`, the strongest currently aligned variant (`eval_output/fidelity_report/report.md:23`). Implement only enough dtype plumbing for FA2 and tests; do not generalize all ops.
4. **Fix reference weight passing or mark weighted FA2 pairs separately.** The adapter currently omits `weight_attr`, and live `fa2` treats that as unweighted (`dagua/eval/competitors/fa2_competitor.py:176-182`, `dagua/eval/competitors/fa2_competitor.py:219-222`, `.../fa2/forceatlas2.py:621-683`).
5. **Mark `fa2_dissuade_hubs` as intentionally incomparable or add a reference shim.** It has no native live-reference parameter (`dagua/eval/variants.py:632-656`; `dagua/eval/competitors/fa2_competitor.py:83-96`), so chasing it as a normal divergence will waste effort.

I would defer Barnes-Hut traversal surgery until after the dtype/reference-target fixes. BH is probably responsible for the weak-equivalent residual in default/BH variants, but it is more invasive and harder to verify than the strong-gravity, dtype, and adapter-target issues.

## Assumptions

- I treated the live benchmark reference as `fa2.forceatlas2.ForceAtlas2` because the project adapter imports `fa2` before `fa2_modified` and the local environment resolves that import successfully (`dagua/eval/competitors/fa2_competitor.py:31-38`). I still read and cited `fa2_modified` because the task explicitly requested it and dagua's archived implementation is documented as a `fa2_modified` translation (`dagua/layout/_archive/classic/fa2.py:1`).
- I did not run fidelity benchmarks because the task was diagnosis-only and requested one markdown report. Verification was limited to source inspection and file creation.

## Dead Code / Removable Candidates

- No source was changed, so no new dead code was created.
- `dagua/layout/_archive/classic/fa2.py` appears to be a frozen reference translation, not live pipeline code. It is useful for provenance but should not be used as the current parity oracle unless `fa2_ref` is pinned to `fa2_modified`.
