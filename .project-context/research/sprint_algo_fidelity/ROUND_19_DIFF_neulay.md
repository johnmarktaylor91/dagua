# Round 19 Diff: NeuLay

Diagnosis-only adversarial diff for the active Dagua NeuLay pipeline against the
cloned upstream NeuLay reference at `/home/jtaylor/projects/_references/NeuLay`.

## Sources Read

- Reference old NeuLay script: `/home/jtaylor/projects/_references/NeuLay/old_code/NeuLay-2.py`
- Reference force-directed baseline: `/home/jtaylor/projects/_references/NeuLay/old_code/FDL.py`
- Reference PyG notebook JSON: `/home/jtaylor/projects/_references/NeuLay/NeuLay_pyg.ipynb`
- Reference README: `/home/jtaylor/projects/_references/NeuLay/README.md`
- Dagua ops: `dagua/layout/ops/neulay.py`
- Dagua pipeline: `dagua/layout/ops/pipelines/neulay.py`
- Dagua competitor adapter: `dagua/eval/competitors/neulay_competitor.py`

## Executive Summary

Dagua's active NeuLay implementation most closely follows the older script
`NeuLay-2.py`: sparse normalized adjacency, `100 -> 100 -> 3` GCN features,
Tanh after the first GCN, RMSprop at `lr=0.01`, Gaussian local repulsion, and a
direct refinement phase initialized from the GCN output. The high-level
two-phase structure is correct.

The largest fidelity gaps are:

1. **Output dimension defaults differ**: reference uses `dim = 3`; Dagua
   defaults to `dim = 2`.
2. **KD-tree query radius differs**: old reference queries radius `4` absolute;
   Dagua queries `4 * radius = 1.6` with the default `radius = 0.4`.
3. **Training budgets differ for old script**: old reference runs up to 40,000
   GCN epochs and then up to 1,000,000 direct epochs; Dagua defaults to 2,000
   GCN steps and 18,000 direct steps.
4. **The notebook is a materially different PyG variant**: Adam, optional
   `GCNConv` / `GATConv` / `GraphNet`, normal latent initialization, full
   all-pairs repulsion, fixed magnitude `10`, and mean-window early stopping.
5. **Dagua competitor install assumption is false for this clone**: the cloned
   repo has no packaging metadata, so `pip install -e` fails.

## 1. Overall Flow

### Old-code reference

`NeuLay-2.py` implements two phases:

- GCN reparameterization: instantiate `LayoutNet`, train it with RMSprop, and
  stop on a rolling loss-window criterion (`NeuLay-2.py:222-274`).
- Direct refinement: detach the GCN output into a learnable parameter `w`, wrap
  it with `LayoutLinear`, and continue RMSprop on direct coordinates
  (`NeuLay-2.py:276-314`).

The direct-only baseline in `FDL.py` has only the second phase: a bias-free
`nn.Linear(N, dim)` trained with RMSprop (`FDL.py:140-199`).

### PyG notebook reference

The notebook `ResGCN.train()` also has two phases:

- `train_gcn()` runs if `gcn_steps > 0` and there are GCN layers
  (`NeuLay_pyg.ipynb:386-404`, `NeuLay_pyg.ipynb:413-425`).
- It copies `self()` into `fine_pos`, then calls `train_fine()`
  (`NeuLay_pyg.ipynb:398-405`, `NeuLay_pyg.ipynb:428-440`).

### Dagua

Dagua's pipeline follows the same high-level shape:

- `build_neulay_pipeline()` seeds RNG, prepares state, runs `NeuLayRunGCNPhase`,
  prepares the direct optimizer, repeats direct steps/convergence checks, and
  finalizes positions (`dagua/layout/ops/pipelines/neulay.py:66-91`).
- `NeuLayRunGCNPhase` trains `_ResGCN` when `use_gcn and gcn_steps > 0`; otherwise
  it falls back to random direct initialization (`dagua/layout/ops/neulay.py:467-523`).
- `NeuLayPrepareDirectOptimizer` wraps the GCN output/random positions in a
  learnable `nn.Parameter` and attaches RMSprop (`dagua/layout/ops/neulay.py:563-570`).

Verdict: **flow matches the two-phase idea**, especially the old-code script.

## 2. GCN Architecture

### Old-code reference

The old script architecture is:

- Input is sparse identity `x` (`NeuLay-2.py:174-178`).
- `LayoutNet` starts with a learnable `weight1` of shape `[N, hidden_dim_1]`,
  Xavier initialized with gain `N ** (1 / dim)` (`NeuLay-2.py:114-126`).
- `GCN1`: hidden_dim_1 to hidden_dim_2 through `A_norm @ (X @ W)`
  (`NeuLay-2.py:92-109`, `NeuLay-2.py:127`).
- Activation: `Tanh` after `GCN1` (`NeuLay-2.py:130`, `NeuLay-2.py:145-149`).
- `GCN2`: hidden_dim_2 to hidden_dim_3 (`NeuLay-2.py:133`, `NeuLay-2.py:151`).
- Concatenate `[x, gnn1, gnn2]`, then final `weight2` maps
  `hidden_dim_1 + hidden_dim_2 + hidden_dim_3` to `output_dim`
  (`NeuLay-2.py:135`, `NeuLay-2.py:154-160`).
- Concrete instantiation: `output_dim=dim`, `hidden_dim_1=100`,
  `hidden_dim_2=100`, `hidden_dim_3=3` (`NeuLay-2.py:222-223`).

### PyG notebook reference

The notebook architecture is more flexible:

- `ResGCN` takes `feat_dims`, optional `GCNConv`, `GATConv`, or `GraphNet`, and
  an optional normalized adjacency path (`NeuLay_pyg.ipynb:316-363`).
- Latent parameter shape is `[n, feat_dims[0]]`; initialization is normal with
  `std = n ** (1 / feat_dims[0])` (`NeuLay_pyg.ipynb:348-352`).
- Projection is `nn.Linear(sum(feat_dims[:-1]), feat_dims[-1])` unless only one
  dimension is supplied (`NeuLay_pyg.ipynb:368-369`).
- Example paper/notebook runs use `feat_dims=[100,100,DIMENSIONS]` for
  GraphNet/GCN/GAT and `feat_dims=[10,10,DIMENSIONS]` for some smaller GCN
  tests (`NeuLay_pyg.ipynb:614-617`, `NeuLay_pyg.ipynb:848-849`).

### Dagua

Dagua's active `_ResGCN` matches the old script more than the notebook:

- `_HIDDEN = 100`; `_GCN2_OUT = 3` (`dagua/layout/ops/neulay.py:165-170`).
- `weight1` is `[num_nodes, 100]` with Xavier gain
  `N ** (1 / dim)` (`dagua/layout/ops/neulay.py:179-187`).
- `gcn1` is `100 -> 100`; `gcn2` is `100 -> 3`
  (`dagua/layout/ops/neulay.py:189-190`).
- It uses `torch.tanh` after `gcn1`, concatenates `[h0, h1, h2]`, and maps with
  `weight2` to `dim` (`dagua/layout/ops/neulay.py:192-203`).

Diff:

- Dagua's default output `dim=2` (`dagua/layout/ops/neulay.py:59`,
  `dagua/layout/ops/pipelines/neulay.py:26`, `dagua/layout/ops/pipelines/neulay.py:104`)
  differs from both old script and notebook examples, which set dimensions to
  `3` (`NeuLay-2.py:173-178`, `NeuLay_pyg.ipynb:546`, `NeuLay_pyg.ipynb:813`).
- Dagua does not implement notebook variants `GCNConv`, `GATConv`, or `GraphNet`
  (`NeuLay_pyg.ipynb:196-205`, `NeuLay_pyg.ipynb:614-617`).

## 3. Energy / Loss Function

### Old-code reference

Constants:

- `radius = .4`
- `magnitude = 100 * N ** (1/3) * radius`
- `k = 1`
  (`NeuLay-2.py:181-184`)

Training loss:

- Elastic edge term: `(k / 2) * sum(||X_i - X_j||^2)` over the upper-triangular
  adjacency edge list (`NeuLay-2.py:59`, `NeuLay-2.py:186-194`).
- Repulsion term: `magnitude * sum(exp(-Dist / 4 / radius**2))`, where `Dist`
  is squared distance for KD-tree query pairs (`NeuLay-2.py:77-90`,
  `NeuLay-2.py:191-194`, `NeuLay-2.py:250-255`).

Reported full energy:

- Same elastic term.
- Full all-pairs Gaussian repulsion over `r2_len`, including self-pairs because
  the diagonal is not masked (`NeuLay-2.py:199-209`).

`FDL.py` uses the same force-directed energy constants and terms for its
direct-only baseline (`FDL.py:99-127`).

### PyG notebook reference

Constants:

- `MIN_REPUL_DIST = 1e-3`, but the inverse-distance version is commented out.
- `radius = .4`
- `magnitude = 10`
  (`NeuLay_pyg.ipynb:212-214`)

Loss:

- `Repulsion(X)` computes full all-pairs Gaussian repulsion with
  `magnitude * sum(exp(-r / 4 / radius**2))`; self-pairs are not removed
  (`NeuLay_pyg.ipynb:216-226`).
- Dense `Elastic(X, A)` is `trace(X.T @ L @ X)` with `L = D - A`
  (`NeuLay_pyg.ipynb:229-233`).
- Edgelist elastic is `sum(||X_i - X_j||^2) / 2`
  (`NeuLay_pyg.ipynb:239-243`).

### Dagua

Dagua constants/defaults:

- `radius=0.4`, `magnitude=None`, `magnitude_scale_base=100.0`, and
  `query_radius_factor=4.0` (`dagua/layout/ops/neulay.py:59-67`).
- If magnitude is `None`, Dagua computes
  `100 * N ** (1/3) * radius` (`dagua/layout/ops/neulay.py:383-390`).

Dagua loss:

- KD-tree repulsion uses `magnitude * exp(-sq_dist / (4 * radius * radius)).sum()`
  over cached pairs (`dagua/layout/ops/neulay.py:277-285`).
- GCN/direct elastic terms collapse self/parallel opposite directions into
  unique undirected pairs, then compute `0.5 * sum(||X_i - X_j||^2)`
  (`dagua/layout/ops/neulay.py:484-497`, `dagua/layout/ops/neulay.py:636-649`).
- GCN and direct phases use identical elastic + KD-tree repulsion
  (`dagua/layout/ops/neulay.py:498-506`, `dagua/layout/ops/neulay.py:650-658`).

Diff:

- Dagua matches the old script's adaptive magnitude formula, not the notebook's
  fixed `magnitude=10`.
- Dagua matches the old script's local KD-tree training repulsion structure,
  not the notebook's all-pairs training repulsion.
- Dagua's training loss does not include full all-pairs self-pair repulsion;
  old-code training also uses KD-tree pairs, but its `energy()` output metric
  does include all-pairs/self-pairs.
- Dagua has no `k` multiplier setting; it is effectively fixed at `k=1`,
  matching the old default.

## 4. Optimizer + LR

### Old-code reference

- GCN phase optimizer: `torch.optim.RMSprop(net.parameters(), lr=0.01)`
  (`NeuLay-2.py:227`).
- Direct phase optimizer: `torch.optim.RMSprop(net1.parameters(), lr=0.01)`
  (`NeuLay-2.py:276-279`).
- No LR schedule is applied.
- `FDL.py` also uses RMSprop at `lr=0.01` (`FDL.py:143-148`).

### PyG notebook reference

- GCN optimizer: Adam over all model parameters at `lr / 2`
  (`NeuLay_pyg.ipynb:371`).
- Fine optimizer: Adam over `fine_pos` at `lr`
  (`NeuLay_pyg.ipynb:373-375`).
- Example notebook calls use `lr=1e-1`, giving GCN Adam `0.05` and fine Adam
  `0.1` (`NeuLay_pyg.ipynb:614-617`, `NeuLay_pyg.ipynb:848-849`).
- No LR schedule is visible.

### Dagua

- GCN phase uses RMSprop at `optimizer_lr=0.01`
  (`dagua/layout/ops/neulay.py:76-89`, `dagua/layout/ops/neulay.py:474`).
- Direct phase uses RMSprop at configured `lr`, default `0.01`
  (`dagua/layout/ops/neulay.py:59-67`, `dagua/layout/ops/neulay.py:566-568`).
- No LR schedule is applied.

Verdict: Dagua matches the old script, not the notebook.

## 5. Early Stopping / Patience

### Old-code reference

- `difference(r) = (max(r) - min(r)) / max(r)` (`NeuLay-2.py:213-214`).
- GCN loss window uses `patience = 10` and threshold
  `0.0001 * sqrt(N)` (`NeuLay-2.py:239-240`, `NeuLay-2.py:263-270`).
- Direct phase reuses the same rolling window and stops at
  `1e-8 * sqrt(N)` (`NeuLay-2.py:298-306`).
- Imported `EarlyStopping` instances are created with `patience=10` and
  `patience=3`, but they are not actually used in the loops
  (`NeuLay-2.py:233-235`).
- `FDL.py` uses `patience = 5` and `1e-8 * sqrt(N)` for direct-only FDL
  (`FDL.py:153-155`, `FDL.py:179-189`).

### PyG notebook reference

- Early stopping is not a patience counter; it compares mean loss changes over
  windows (`small_window=32`, `big_window=1000`) and checks
  `dl_small / dl_big < stop_delta_ratio`
  (`NeuLay_pyg.ipynb:456-472`).
- `train()` defaults to `early_stop_check_steps=100`, `min_steps=100`,
  `gcn_stop_threshold=2e-2`, `fdl_stop_threshold=5e-3`
  (`NeuLay_pyg.ipynb:386-390`).
- Experiment cells override thresholds to `5e-3` and `2e-3` or `2e-4`
  (`NeuLay_pyg.ipynb:546-557`, `NeuLay_pyg.ipynb:813-824`).

### Dagua

- Dagua's `_relative_window_difference()` matches the old script's rolling
  relative max/min formula (`dagua/layout/ops/neulay.py:258-263`).
- GCN config defaults: `patience=10`, `relative_tolerance=1e-4`
  (`dagua/layout/ops/neulay.py:70-89`).
- Direct config defaults: `patience=10`, `relative_tolerance=1e-8`
  (`dagua/layout/ops/neulay.py:92-128`).
- GCN and direct thresholds multiply tolerance by `sqrt(num_nodes)`
  (`dagua/layout/ops/neulay.py:509-512`, `dagua/layout/ops/neulay.py:700-703`).

Verdict: Dagua matches the old script's active early stopping, not the notebook.

## 6. Initial Coordinates

### Old-code reference

- GCN phase initializes `weight1`, each GCN layer weight, and `weight2` with
  Xavier uniform using gain `N ** (1 / dim)` (`NeuLay-2.py:99`,
  `NeuLay-2.py:125`, `NeuLay-2.py:135`).
- Direct phase initializes from the detached GCN output, not a fresh random
  tensor (`NeuLay-2.py:276-279`).
- Direct-only FDL baseline initializes `nn.Linear(N, dim, bias=False)` with
  Xavier uniform gain `N ** (1 / dim)` (`FDL.py:67-70`, `FDL.py:140-148`).

### PyG notebook reference

- Latent tensor and fine-position tensor are normal initialized with
  `std = n ** (1 / feat_dims[0])` (`NeuLay_pyg.ipynb:348-352`,
  `NeuLay_pyg.ipynb:373-375`).
- After GCN, `fine_pos.data = self()` (`NeuLay_pyg.ipynb:398-404`).

### Dagua

- GCN initialization uses Xavier uniform with gain `N ** (1 / dim)`, matching
  old code (`dagua/layout/ops/neulay.py:179-197`).
- Direct phase clones the GCN output into an `nn.Parameter`
  (`dagua/layout/ops/neulay.py:563-568`).
- If `use_gcn=False` or `gcn_steps=0`, Dagua uses `_initial_positions()`, a fresh
  Xavier-uniform `[N, dim]` tensor (`dagua/layout/ops/neulay.py:250-255`,
  `dagua/layout/ops/neulay.py:518-523`).

Diff:

- Dagua direct-only initialization matches the old `FDL.py` distribution in
  spirit, but not exactly the same module path (`nn.Linear` weight vs standalone
  tensor).
- Dagua does not match the notebook's normal latent/fine initialization.

## 7. Latent Dimensions

The task prompt says Dagua uses `_LATENT_DIM=10`. In the active files requested
for this diff, no `_LATENT_DIM` constant exists. The active Dagua model uses
`_HIDDEN = 100` and `_GCN2_OUT = 3` (`dagua/layout/ops/neulay.py:165-170`).

Reference dimensions:

- Old script: `hidden_dim_1=100`, `hidden_dim_2=100`, `hidden_dim_3=3`,
  `dim=3` (`NeuLay-2.py:173-178`, `NeuLay-2.py:222-223`).
- Notebook examples: primary GraphNet/GCN/GAT variants use
  `feat_dims=[100,100,DIMENSIONS]`; some smaller tests use
  `feat_dims=[10,10,DIMENSIONS]` (`NeuLay_pyg.ipynb:614-617`,
  `NeuLay_pyg.ipynb:660-663`, `NeuLay_pyg.ipynb:848-849`).

Verdict: active Dagua is aligned to the old script's `100,100,3` internal
architecture. The likely stale `_LATENT_DIM=10` concern does not apply to the
active files read for this task.

## 8. Pair Sampling for Repulsion

### Old-code reference

- `c_kdtree(x, r)` builds a SciPy `cKDTree` from `x.detach().numpy()` and calls
  `query_pairs(r, output_type='ndarray')` (`NeuLay-2.py:77-79`).
- Training refreshes pairs every 5 epochs with absolute radius `4`
  (`NeuLay-2.py:250-253`, `NeuLay-2.py:286-289`).
- Distances are squared pair distances from the cached pairs
  (`NeuLay-2.py:85-90`).
- The direct-only FDL baseline uses the same refresh interval and absolute
  radius `4` (`FDL.py:63-76`, `FDL.py:164-168`).

### PyG notebook reference

- The notebook imports `torch_geometric.nn.pool.radius`, but the visible loss
  uses full all-pairs repulsion and does not call KD-tree or radius sampling
  inside `train_gcn()` / `train_fine()` (`NeuLay_pyg.ipynb:201-226`,
  `NeuLay_pyg.ipynb:413-440`).

### Dagua

- `_query_pairs()` uses SciPy `cKDTree` on `pos.detach().cpu().numpy()`
  (`dagua/layout/ops/neulay.py:266-274`).
- Pair refresh interval defaults to 5 for both GCN and direct
  (`dagua/layout/ops/neulay.py:82-89`, `dagua/layout/ops/neulay.py:105-115`).
- Query radius is stored as `query_radius_factor * radius`; defaults are
  `4.0 * 0.4 = 1.6` (`dagua/layout/ops/neulay.py:53-67`,
  `dagua/layout/ops/neulay.py:407-409`).

Major diff: **old reference uses absolute query radius `4`; Dagua uses
`4 * radius = 1.6`**. If the old script is the primary fidelity target, Dagua
is under-sampling repulsive pairs.

## 9. Number of Training Steps

### Old-code reference

- GCN phase upper bound: `range(40000)` (`NeuLay-2.py:244`).
- Direct phase upper bound: `range(epoch, 1000000)` (`NeuLay-2.py:280`).
- FDL direct-only baseline upper bound: `range(500000)` (`FDL.py:157`).
- These are high ceilings with early stopping expected to terminate before the
  ceiling.

### PyG notebook reference

- `ResGCN.train()` defaults are `gcn_steps=200`, `fdl_steps=2000`
  (`NeuLay_pyg.ipynb:386-390`).
- Experiment constants use `MAX_GCN_STEPS = int(2e3)` and
  `MAX_FDL_STEPS = int(2e4)` (`NeuLay_pyg.ipynb:546-552`,
  `NeuLay_pyg.ipynb:813-819`).

### Dagua

- Pipeline defaults: `steps=20_000`, `gcn_steps=2_000`, so direct steps are
  `18_000` when `use_gcn=True`
  (`dagua/layout/ops/pipelines/neulay.py:22-30`,
  `dagua/layout/ops/pipelines/neulay.py:64`).
- Public wrapper uses the same defaults (`dagua/layout/ops/pipelines/neulay.py:96-108`).
- Competitor adapter uses `steps=20_000`, `gcn_steps=2_000`, `use_gcn=True`
  (`dagua/eval/competitors/neulay_competitor.py:130-135`).

Diff:

- Dagua matches the notebook's experiment-scale total budget approximately
  (`2k + 20k`), but not the old script's `40k + up to 1M`.
- Dagua interprets `steps` as total budget, not direct/fine steps. Notebook
  names `MAX_FDL_STEPS=20k`, separate from `MAX_GCN_STEPS=2k`.

## 10. RNG

### Old-code reference

- No explicit seed is set in `NeuLay-2.py` or `FDL.py`.
- Randomness is PyTorch module initialization via Xavier; NumPy creates the
  rolling zero windows but not the coordinates (`NeuLay-2.py:99`,
  `NeuLay-2.py:125`, `NeuLay-2.py:135`, `NeuLay-2.py:239-240`).

### PyG notebook reference

- No explicit seed is visible in the read cells.
- Randomness is PyTorch normal initialization for latent and fine positions
  (`NeuLay_pyg.ipynb:348-352`, `NeuLay_pyg.ipynb:373-375`).

### Dagua

- `NeuLaySeedRNG` calls `torch.manual_seed(problem.seed)`,
  `np.random.seed(problem.seed)`, and `torch.cuda.manual_seed_all()` when CUDA is
  available (`dagua/layout/ops/neulay.py:288-311`).
- Public wrapper default seed is `42` (`dagua/layout/ops/pipelines/neulay.py:96-108`).

Diff:

- Dagua is deterministic by default; upstream scripts/notebook are not seeded in
  the checked-in code.
- This is good for Dagua tests but is not bit-fidelity to the reference unless a
  reference harness also sets the same torch seed before construction.

## 11. Hyperparameter Alignment Table

| Hyperparameter | Old `NeuLay-2.py` | PyG notebook | Dagua active | Fidelity verdict |
|---|---:|---:|---:|---|
| Output dim | `3` (`NeuLay-2.py:174`) | `3` (`NeuLay_pyg.ipynb:546`) | `2` default (`dagua/layout/ops/pipelines/neulay.py:26`) | Mismatch |
| Hidden / latent dims | `100,100,3` (`NeuLay-2.py:222-223`) | Often `100,100,3`; sometimes `10,10,3` (`NeuLay_pyg.ipynb:614`, `NeuLay_pyg.ipynb:848`) | `100,100,3` internal (`dagua/layout/ops/neulay.py:168-190`) | Matches old |
| Activation | Tanh after first GCN (`NeuLay-2.py:130`, `NeuLay-2.py:148`) | PyG modules/GraphNet use internal ReLU for GraphNet (`NeuLay_pyg.ipynb:32-33`) | Tanh after first GCN (`dagua/layout/ops/neulay.py:201`) | Matches old |
| Adjacency normalization | Symmetric `D(A+I)D` (`NeuLay-2.py:67-73`) | Row-like `diag(1/(.1+deg))A` for custom path; PyG layers otherwise (`NeuLay_pyg.ipynb:180-184`, `NeuLay_pyg.ipynb:196-205`) | Symmetric `D(A+I)D` (`dagua/layout/ops/neulay.py:206-247`) | Matches old |
| Repulsion radius | `0.4` (`NeuLay-2.py:182`) | `0.4` (`NeuLay_pyg.ipynb:213`) | `0.4` (`dagua/layout/ops/neulay.py:59-67`) | Match |
| Repulsion magnitude | `100*N^(1/3)*radius` (`NeuLay-2.py:183`) | `10` (`NeuLay_pyg.ipynb:214`) | `100*N^(1/3)*radius` (`dagua/layout/ops/neulay.py:383-390`) | Matches old |
| Pair query radius | Absolute `4` (`NeuLay-2.py:250-253`) | Full all-pairs | `4*radius=1.6` default (`dagua/layout/ops/neulay.py:407-409`) | Mismatch vs old |
| Pair refresh | Every 5 epochs (`NeuLay-2.py:250`, `NeuLay-2.py:286`) | None visible | Every 5 steps (`dagua/layout/ops/neulay.py:82-89`, `dagua/layout/ops/neulay.py:631-634`) | Matches old interval |
| GCN optimizer | RMSprop `0.01` (`NeuLay-2.py:227`) | Adam `lr/2`, examples `0.05` (`NeuLay_pyg.ipynb:371`, `NeuLay_pyg.ipynb:614`) | RMSprop `0.01` (`dagua/layout/ops/neulay.py:474`) | Matches old |
| Direct optimizer | RMSprop `0.01` (`NeuLay-2.py:278`) | Adam `lr`, examples `0.1` (`NeuLay_pyg.ipynb:375`, `NeuLay_pyg.ipynb:614`) | RMSprop `0.01` default (`dagua/layout/ops/neulay.py:566-568`) | Matches old |
| GCN max steps | `40000` (`NeuLay-2.py:244`) | `2000` experiments (`NeuLay_pyg.ipynb:549`) | `2000` default (`dagua/layout/ops/pipelines/neulay.py:24`) | Matches notebook |
| Direct max steps | Up to `1000000` from `epoch` (`NeuLay-2.py:280`) | `20000` experiments (`NeuLay_pyg.ipynb:552`) | `steps - gcn_steps = 18000` (`dagua/layout/ops/pipelines/neulay.py:64`) | Near notebook, mismatch old |
| GCN early stop | max/min window 10, `1e-4*sqrt(N)` (`NeuLay-2.py:239-270`) | mean-window ratio, default `2e-2` (`NeuLay_pyg.ipynb:386-390`, `NeuLay_pyg.ipynb:456-472`) | max/min window 10, `1e-4*sqrt(N)` (`dagua/layout/ops/neulay.py:86-89`, `dagua/layout/ops/neulay.py:509-512`) | Matches old |
| Direct early stop | same window, `1e-8*sqrt(N)` (`NeuLay-2.py:298-306`) | mean-window ratio, defaults/experiments vary | same window, `1e-8*sqrt(N)` (`dagua/layout/ops/neulay.py:119-128`, `dagua/layout/ops/neulay.py:700-703`) | Matches old |
| Seed | None visible | None visible | Default `42` and explicit torch/numpy seeding (`dagua/layout/ops/neulay.py:307-310`, `dagua/layout/ops/pipelines/neulay.py:100`) | Mismatch but desirable |

## 12. Ranked Fix List

1. **Expose old-code fidelity mode defaults**: set NeuLay fidelity defaults to
   `dim=3`, GCN max `40000`, direct max old-code-compatible, and absolute
   `query_radius=4`. This is the highest-impact diff against the checked-in
   script.
2. **Separate `steps` from `fdl_steps` semantics**: Dagua currently treats
   `steps` as total budget, but notebook/reference naming treats FDL/direct
   steps as a separate post-GCN budget.
3. **Fix KD-tree radius semantics**: add a config path for absolute query
   radius, or set `query_radius_factor=10.0` when `radius=0.4` to match the
   old script's absolute `4`.
4. **Make output dimensionality explicit in competitor variants**: the adapter's
   variant names omit `dim` (`dagua/eval/competitors/neulay_competitor.py:67`);
   add it if benchmarking against 3D reference layouts.
5. **Decide reference target**: old script and PyG notebook disagree on
   optimizer, magnitude, repulsion sampling, architecture variants, and early
   stopping. Dagua should name modes like `old_code` vs `pyg_notebook` rather
   than mixing defaults.
6. **Optional PyG variant support**: if paper-fidelity means the notebook,
   implement `Adam`, normal latent/fine initialization, mean-window early stop,
   and optional `GCNConv` / `GraphNet` behavior behind explicit parameters.
7. **Reference energy reporting**: if parity metrics compare reported energy,
   add an all-pairs `energy()` metric matching `NeuLay-2.py:199-209`, separate
   from the KD-tree training loss.

## 13. Installation Feasibility

The cloned reference repo is not directly pip-installable:

- `find` found no `setup.py`, `setup.cfg`, `pyproject.toml`, requirements file,
  or environment file under `/home/jtaylor/projects/_references/NeuLay`.
- `python -m pip install -e /home/jtaylor/projects/_references/NeuLay --dry-run`
  failed with:

```text
ERROR: file:///home/jtaylor/projects/_references/NeuLay does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
```

The competitor adapter expects an importable package named `neulay` or `NeuLay`
with `layout_neulay` or `layout` entry point
(`dagua/eval/competitors/neulay_competitor.py:30-58`). The clone provides only
scripts/notebook plus data, so the adapter cannot load it as-is.

Dependency check in the current environment:

- Installed: `torch`, `torch_geometric`, `torch_scatter`, `networkx`, `pandas`,
  `scipy`.
- Missing: `pytorchtools`, which `NeuLay-2.py` and `FDL.py` import
  (`NeuLay-2.py:8`, `FDL.py:8`). Those imported `EarlyStopping` objects are not
  active in the old-code loops, but the import still prevents direct execution
  without a shim or dependency.

Recommended install path for benchmarking is not `pip install -e`; create a
small local wrapper module that imports/adapts the cloned scripts or ports the
needed functions into a callable harness with explicit graph input.

## 14. Recommended Round 20 Fix Scope

Recommended Round 20 scope should be narrow and old-code targeted:

1. Add explicit NeuLay config parameters for `dim`, `gcn_steps`, `fdl_steps` or
   `linear_steps`, and absolute `query_radius`.
2. Preserve current Dagua defaults for existing users, but add a named
   `old_code_fidelity=True` or equivalent path that uses:
   `dim=3`, `gcn_steps=40000`, direct ceiling compatible with old script,
   RMSprop `lr=0.01`, radius `0.4`, magnitude `100*N^(1/3)*radius`,
   pair refresh `5`, absolute query radius `4`, GCN early tolerance `1e-4`,
   direct early tolerance `1e-8`.
3. Add regression tests for parameter resolution and loss components rather than
   long-running optimization parity.
4. Update the competitor adapter separately to reflect that this clone is not an
   importable upstream package; do not block algorithm fixes on packaging.

## Assumptions

- I treated `old_code/NeuLay-2.py` as the primary fidelity target because the
  prompt explicitly says the original paper code was cloned and the old-code
  script is the main implementation.
- I treated `NeuLay_pyg.ipynb` as a second reference variant because its
  optimizer, initialization, and loss differ materially from `NeuLay-2.py`.
- I did not use `dagua/layout/_archive/classic/neulay.py` for the Dagua-side
  diff because the task's Dagua-side input list was limited to the active ops,
  active pipeline, and competitor adapter.

## Verification

- Created this diagnosis document only.
- No implementation files were modified.
- No tests were run because this was a read-only diagnosis task.
