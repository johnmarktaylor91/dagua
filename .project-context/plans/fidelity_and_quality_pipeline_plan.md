# Fidelity + Quality/Runtime Analysis Pipelines -- Unified Plan v4

Date: 2026-04-09
Branch: feat/bench-and-aesthetics
Status: Revised after THREE rounds of adversarial review by Codex and Claude Explore.

Reviews folded in:
- Round 1: .project-context/plans/fidelity_quality_codex_review.md, fidelity_quality_claude_review.md
- Round 2: .project-context/plans/fidelity_quality_round2_codex.md, fidelity_quality_round2_claude.md
- Round 3: .project-context/plans/fidelity_quality_round3_codex.md, fidelity_quality_round3_claude.md

User decisions (2026-04-09): see "User decisions" below.

---

## User decisions (all open questions closed)

1. Fidelity B2: expand `QUALITY_METRICS` to the full set (sampled_stress,
   crossing_rate, overlap_count, edge_straightness_mean_deg,
   depth_spearman_rho).
2. Fidelity D1: raise `PAIRWISE_SAMPLE_SIZE` from 10 to 30.
3. QR seed budget: evaluate ALL successful seeds per stochastic engine
   (up to 60).
4. QR dagua surface: compare every competitor against the SINGLE engine
   literally named `dagua` in the manifest.
5. QR runtime cost: multiprocessing + on-disk cache.
6. QR inline Pareto: two tables per family (`sampled_stress` AND
   `crossing_rate`).
7. QR plots: PNG plots ON by default. `--no-plots` disables.
8. QR cyclic families: skip from `dag_consistency` and
   `depth_spearman_rho` rank tables.
9. QR family threshold: emit scorecards only for families with
   `coverage_ratio >= 0.5` and `>= 3` graphs.
10. Dagua default = single engine named `dagua` in the manifest.

---

## CRITICAL FINDINGS

### CF1: `within_rmsd` is POOLED orig+reimpl, not within-original
`scripts/fidelity_analysis.py:1816-1817` pools `pairwise_orig +
pairwise_reimpl`. Must be within-original only. Group A1.

### CF2: Report is LaTeX/PDF, not markdown
`scripts/generate_fidelity_report.py:1, :894, :914, :959`. Full rewrite
required. Cleanup2.

### CQ1: Quality metrics are NOT in `results.json`
`BenchmarkRecord` at `scripts/run_benchmark.py:162-210` stores only
status, runtime, positions_path, sizes. Both pipelines must recompute
metrics from positions.

Corollary: `fidelity_analysis.py:814-822` already calls `quick()` per
layout. Expanding `QUALITY_METRICS` at line 38 is free on the loading
side for metrics already in `quick()`.

### CQ2: Stochastic-at-eval metrics break reproducibility
- `count_overlaps_detailed()` at `dagua/metrics.py:397`: `torch.randperm(m)[:200]` no seed.
- `sampled_crossing_rate()` at `:582-583`: `torch.randint()` no seed.
- `count_crossings()` at `:1554` delegates to sampled for E > 500.
- `sampled_stress()` at `:476` is ALREADY deterministic (`:510` docstring,
  `_deterministic_sample_indices()` at `:531`, `:543`).
- Other stochastic metrics exist (`neighborhood_preservation`,
  `angular_resolution`, `cluster_separation`) but only in `full()`, NOT
  in the QR metric surface. Not in scope.

### CQ3: `graph_rel_best` needs rank-primary, clamped-secondary
Per-graph RANK is the primary ordering (scale-immune). `rel_best` is
secondary with a clamp + floor for pathological near-zero cases. See
QR-CORE section.

### CQ4: Coverage denominator must be `covered / scheduled`
Engines capped by `max_nodes` look under-covered otherwise. The
benchmark scheduler writes `skipped` rows with `skip_reason` for
`max_nodes` violations at `scripts/run_benchmark.py:2126+`, so the
implementation can derive "scheduled" directly from `results.json` rows
with status in `{ok, error, skipped, timeout, running}`.

### CQ5: `validate_sync()` is a real hard gate in fidelity
`scripts/fidelity_analysis.py:2479-2499` imports `validate_sync` from
`validate_benchmark_integrity` and calls `sys.exit(1)` if
`desync_count > 10`. Cleanup1 removes the hard gate.

### CQ6: FIX-S uses the EXISTING `stable_seed()` helper
`scripts/fidelity_analysis.py:463-477` -- SHA-256 based, signature
`stable_seed(*parts: str) -> int`. Already used at `:1519, :1770, :1775`
AND at `scripts/fidelity_recompute_verdicts.py:26, :42, :149` (four
consumers total, not three). Python's built-in `hash()` is
process-randomized and must NOT be used.

### CQ7: `aspect_ratio_deviation` is a derived metric
`dagua/metrics.py:415` has raw `aspect_ratio` only. Define
`aspect_ratio_deviation(positions) = |log(max(raw_aspect_ratio, eps))|`
in the shared helper. Fidelity uses raw; QR uses derived.

### CQ8: Pareto ideal corner is (1.0, 0.0)
x = `median_runtime_rel_fastest` (min 1.0), y = `median_rel_best`
(min 0.0). Ideal corner is (1.0, 0.0) -- bottom-left is NOT the ideal.

### CQ9: `positions.h5` is stale; benchmark writes only `.pt` files
`eval_output/variant_bench_full/positions.h5` last modified 2026-03-31
(9 days old, 1.4GB, ~400k records). Current benchmark writes only
`.pt` files to `positions/` (917,951 files as of 2026-04-09 14:16).
`scripts/run_benchmark.py` has ZERO h5py references.

Implications:
- No concurrency conflict between QR-CORE readers and the running
  benchmark writer (which doesn't touch h5).
- But h5 data is STALE for all new records. The pipeline MUST fall back
  to `.pt` for any record_key the h5 doesn't have.
- Operational flow: after benchmark completes, run
  `scripts/consolidate_positions_hdf5.py` (~3 hours for ~900k records,
  one-time cost) to refresh the h5 store. Then analysis loads are 75x
  faster (Codex confirmed at `scripts/consolidate_positions_hdf5.py:6`).
- Or skip consolidation: the pipeline works directly from `.pt` files
  with slower analysis loads.

### CQ10: QR-IO path semantics
`positions_file` in `results.json` is ALREADY a relative path INCLUDING
the `positions/` prefix -- see `scripts/run_benchmark.py:87`
(`POSITION_DIRNAME = "positions"`), `:420-427`
(`position_relative_path` prepends `POSITION_DIRNAME`), and `:1462`
(`positions_file = str(relative_path)`). Fidelity already loads via
`path = input_dir / record.positions_file` at `:799`.

The loader MUST accept `input_dir: Path` (benchmark root) and call
`input_dir / positions_file`. It must NOT accept a separate
`positions_dir` argument -- that would double the prefix.

### CQ11: Canonical rejection reason enum
Both v2 and v3 invented rejection strings that didn't match the actual
code. v4 uses the EXACT existing enum from `fidelity_analysis.py`:

```python
# From load_layout() at :785-806 and validate_positions() at :722-752:
REJECTION_REASONS = (
    # load_layout direct returns
    "missing_positions_file",   # positions_file is None (line 786)
    "h5_load_failure",          # HDF5 read raised (line 797)
    "load_failure",             # torch.load raised (line 803) -- covers .pt missing
    "not_tensor",               # loaded object is not a torch.Tensor (line 805)
    # validate_positions returns (lines 740-751)
    "tensor_not_2d",            # positions.ndim != 2
    "tensor_not_xy",            # positions.shape[1] != 2
    "too_few_nodes",            # positions.shape[0] < MIN_VALID_NODE_COUNT
    "node_count_mismatch",      # positions.shape[0] != expected_nodes
    "contains_nan",             # torch.isnan().any()
    "contains_inf",             # torch.isinf().any()
)
```

Note: there is NO distinct `h5_missing_key` code -- when the key is
missing in HDF5, `load_layout` silently falls through to the `.pt`
branch. There is NO distinct `pt_missing` code either -- a missing `.pt`
file raises `FileNotFoundError`, caught as `load_failure`.

The canonical enum is used by:
- `pipeline_io.load_position_tensor()` return type
- Group E's `rejection_breakdown` bucket keys
- The markdown report's failures section

### CQ12: FID-C Tier 1 does NOT need node-index sorting
Position tensors are stored row-indexed by graph node index (verified at
`scripts/run_benchmark.py:1453-1461` + `:182`). Both orig and reimpl
tensors are in the same order. Direct `torch.equal(orig_pos, reimpl_pos)`
works. The v3 spec added a `sort_idx` that was over-engineering.

---

## STALE ITEMS REMOVED

- v1 B2 BH NaN propagation: already fixed in
  `apply_bh_correction()` at `:1976-1990`.
- v1 F2 pre-flight validation: already exists in `run_analysis()` at
  `:2469+`.
- v1 risk note "fidelity only reads pre-computed metrics": false,
  `:814` already calls `quick()`.

## CITATION CORRECTIONS

| Wrong | Corrected |
|---|---|
| `MAX_PROCRUSTES_SEEDS_PER_SIDE` | `PAIRWISE_SAMPLE_SIZE` (`:58`) |
| BH correction "at line ~2617" | `apply_bh_correction()` at `:1960`; multipletests at `:1986` |
| `apply_deferred_bhcorrection` | `apply_bh_correction` |
| Family threshold "85%" | `>=0.90` strong/weak; `>0.50` divergent (`:2259-2266`) |
| "77% complete" in v2 budget | True: log shows 77.6%, manifest `completed_scope` is stale at ~810k, actual ok rows in results.json are ~914k. Use ~914k as the QR recomputation base. |
| `overlap_count` NOT in quick() | `overlap_count` IS in `quick()` at `metrics.py:1222` |
| `stable_seed` has 3 callers | 4 callers: fidelity_analysis `:1519, :1770, :1775` + fidelity_recompute_verdicts `:26, :42, :149` |
| `positions_dir / positions_path` in loader | `input_dir / positions_file` (positions_file already includes the "positions/" prefix) |
| v3 invented rejection strings | Use the canonical enum from `load_layout` + `validate_positions` verbatim |
| FID-C Tier 1 "canonical node ordering" | Direct `torch.equal(orig_pos, reimpl_pos)`; no sorting |

---

## PART 0: SHARED HELPERS (QR-IO task -- Wave 0 root)

### Shared module: `dagua/eval/pipeline_io.py`

Internal evaluation infrastructure. NOT re-exported from
`dagua.eval.__all__`.

**Wave 0 scope** (narrowed per Codex R3 HIGH #3): QR-IO ships ONLY:
- `stable_seed` (moved from fidelity_analysis.py)
- `load_position_tensor` (extracted from load_layout)
- `validate_positions` (moved from fidelity_analysis.py)
- `aspect_ratio_deviation` (new derived metric)
- `open_h5_for_worker` (worker initializer helper)

QR-IO does NOT ship `compute_quick_metrics` or `compute_sampled_metrics`
in Wave 0. Those depend on FIX-S signatures (`seed` parameter on
`quick()` and stochastic metric functions) which don't exist yet. They
are added in Wave 2 as part of QR-CORE after FIX-S has landed and
`quick()` accepts `seed`.

```python
# dagua/eval/pipeline_io.py (Wave 0 version)

import hashlib
import math
from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import h5py


def stable_seed(*parts: str) -> int:
    """Process-stable 32-bit seed from SHA-256 of string parts.

    Moved from scripts/fidelity_analysis.py:463. Existing callers
    (fidelity_analysis.py:1519, :1770, :1775 and
    fidelity_recompute_verdicts.py:26, :42, :149) must be updated to
    import from here.
    """
    joined = "::".join(parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8], 16)


def validate_positions(
    positions: torch.Tensor,
    expected_nodes: int,
    *,
    min_valid_nodes: int = 3,
) -> Optional[str]:
    """Validate a loaded layout tensor. Returns rejection reason or None.

    Reasons match the canonical enum in CQ11.
    """
    if positions.ndim != 2:
        return "tensor_not_2d"
    if positions.shape[1] != 2:
        return "tensor_not_xy"
    if positions.shape[0] < min_valid_nodes:
        return "too_few_nodes"
    if positions.shape[0] != expected_nodes:
        return "node_count_mismatch"
    if torch.isnan(positions).any().item():
        return "contains_nan"
    if torch.isinf(positions).any().item():
        return "contains_inf"
    return None


def load_position_tensor(
    *,
    record_key: str,
    positions_file: Optional[str],  # e.g. "positions/graph__engine.pt", may be None
    input_dir: Path,                # benchmark root (NOT the positions/ subdir)
    h5_file: Optional["h5py.File"] = None,  # worker-local handle, or None
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Raw position loader. Returns (tensor, reason).

    Rejection reasons match the canonical enum:
        missing_positions_file, h5_load_failure, load_failure, not_tensor

    Lookup order:
      1. If h5_file is provided AND record_key is in h5_file, read from HDF5.
      2. Otherwise (or HDF5 read failed), fall back to .pt file at
         input_dir / positions_file.

    Validation (shape, NaN, inf) is NOT done here -- call
    validate_positions() separately after loading.
    """
    # 1. Early rejection
    if positions_file is None:
        return None, "missing_positions_file"

    # 2. HDF5 first
    if h5_file is not None and record_key and record_key in h5_file:
        try:
            arr = h5_file[record_key][:]
            tensor = torch.from_numpy(arr).to(dtype=torch.float32)
            return tensor, None
        except Exception:
            return None, "h5_load_failure"
        # NOTE: on h5_load_failure we return immediately; the fallback
        # is only triggered when the key is MISSING from HDF5, not when
        # a read error occurs. This matches current load_layout
        # behavior at fidelity_analysis.py:796-797.

    # 3. .pt fallback
    pt_path = input_dir / positions_file
    try:
        tensor = torch.load(pt_path, map_location="cpu")
    except Exception:
        return None, "load_failure"
    if not isinstance(tensor, torch.Tensor):
        return None, "not_tensor"
    return tensor.detach().to(dtype=torch.float32, device="cpu"), None


def open_h5_for_worker(h5_path: Path) -> Optional["h5py.File"]:
    """Open HDF5 for a worker process. Call from multiprocessing
    initializer, store result in a worker-local global.

    Returns None if the file doesn't exist so the pipeline gracefully
    falls through to .pt loading.
    """
    if not h5_path.exists():
        return None
    import h5py
    return h5py.File(h5_path, "r")


def aspect_ratio_deviation(positions: torch.Tensor) -> float:
    """Derived metric: |log(max(aspect_ratio, eps))|.

    Raw aspect_ratio is width/height with no monotone "better"
    direction around 1.0. The log transform makes this lower-better
    with a zero at ratio=1.0.
    """
    from dagua.metrics import aspect_ratio
    raw = aspect_ratio(positions)
    ratio = float(raw.get("aspect_ratio", 1.0))
    if ratio <= 1e-9:
        return float("inf")
    return abs(math.log(ratio))
```

**Refactor step (part of QR-IO task):**
- Move `stable_seed` from `fidelity_analysis.py:463-477` to pipeline_io.
- Move `validate_positions` from `fidelity_analysis.py:722-752` to
  pipeline_io.
- Update imports in:
  - `fidelity_analysis.py`: add `from dagua.eval.pipeline_io import stable_seed, validate_positions, load_position_tensor`
  - `fidelity_recompute_verdicts.py`: add
    `from dagua.eval.pipeline_io import stable_seed`
- Refactor `load_layout()` at `:755-834` to call the new
  `load_position_tensor()` for the load path, then `validate_positions()`,
  then its own metric computation. Preserve the function attributes
  `_h5_file`, `_positions_cache`, `_skip_metrics` as local behavior (not
  moved to pipeline_io).

**Equivalence requirement**: QR-IO must be a pure refactor on the
fidelity side. Run `scripts/fidelity_analysis.py` against a small
fixture before AND after the refactor, diff the outputs, assert
zero-delta.

### FIX-S seeding (in `dagua/metrics.py`) -- Wave 1 FID-S task

Three functions add `seed: int | None = None` keyword parameters:

1. **`count_overlaps_detailed(pos, node_sizes, *, seed=None)`**
   - `:397`: replace `torch.randperm(m)[:200]` with:
     ```python
     gen = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
     cell_nodes = cell_nodes[torch.randperm(m, generator=gen)[:200]]
     ```
   - `torch.randperm` accepts `generator` kwarg in torch >= 1.0. Verify
     project's torch version at `pyproject.toml` supports this.

2. **`sampled_crossing_rate(pos, edge_index, n_samples=..., *, seed=None)`**
   - `:582-583`: pass a seeded generator to `torch.randint`.

3. **`count_crossings(pos, edge_index, *, seed=None)`**
   - `:1554+`: forward `seed` into the `sampled_crossing_rate`
     delegation.

4. **`quick(pos, edge_index, *, seed=None, ...)`**
   - `:1166-1232`: add `seed: int | None = None` parameter.
   - `:1222`: call `count_overlaps_detailed(pos, ns, seed=seed)`.
   - No other quick() callee is stochastic.

5. **Tests** at `tests/test_metric_seeding.py`:
   - **Reproducibility**: `count_overlaps_detailed(pos, ns, seed=42) == count_overlaps_detailed(pos, ns, seed=42)` (two calls, identical output).
   - **Stochasticity preserved**: on a graph with overlap ambiguity,
     5 calls to `count_overlaps_detailed(pos, ns, seed=None)` must
     produce at least two different results (the `seed=None` branch
     uses the global RNG and is therefore stochastic).
   - **Distinct seeds**: `seed=1 != seed=2` on ambiguous graph.
   - Same three tests for `sampled_crossing_rate`.
   - Same for `count_crossings` on E > 500 (to trigger the sampled branch).
   - `quick(pos, ei, ns, seed=42)` reproducibility on the
     `overlap_count` field only.

---

## PART 1: FIDELITY PIPELINE FIX-LIST

### Group A: Procrustes statistics (CRITICAL/HIGH; atomic)

**A1 -- Fix the pooled within distribution.**
- File: `fidelity_analysis.py:1814-1833`
- Change: `within_rmsd = [c.procrustes_rmsd for c in pairwise_orig]` ONLY.
- Add diagnostic field `within_reimpl_rmsd` from `pairwise_reimpl` for
  reporting; do NOT feed it into the equivalence test.
- CSV: `within_rmsd_mean/std` stay as within-original; add
  `reimpl_rmsd_mean/std` as diagnostic.

**A2 -- Add Procrustes-specific TOST equivalence test.**
- File: `fidelity_analysis.py`, insert after `:1828`
- For each factor in `[0.5, 1.0, 1.5, 2.0]`:
  - `std_within_orig = np.std(within_orig_rmsd, ddof=1)` if `len >= 2`
    else a tiny floor
  - `margin = factor * std_within_orig`
  - `pvalue = tost_pvalue(within_orig_rmsd, between_rmsd, margin)`
  - Store `procrustes_tost_margin_{label}`,
    `procrustes_tost_pvalue_{label}_raw`, `..._bh`
- Mirror `metric_test_columns()` at `:1109` with
  `procrustes_tost_columns()`.
- Add to `per_graph_fieldnames()` at `:1149`.
- Initialize in row dict near `:1657`.
- Add to BH bucket dict at `:2529-2534`.

**A3 -- Add Procrustes two-sided Mann-Whitney U.**
- File: `:1824`
- Add `mannwhitneyu(between_rmsd, within_orig_rmsd, alternative="two-sided")`.
- New columns `procrustes_mannwhitney_pvalue_raw` and `_bh`.

**A4 -- BH-correct the existing one-sided procrustes p-value.**
- File: `:2529-2534` + `apply_bh_correction()` at `:1960`
- Add a `procrustes_one_sided` bucket.

**A5 -- Delete the backwards verdict heuristic; replace with TOST
routing.**
- File: `:2155-2172`
- **DELETE** (this is the "absence of evidence" bug):
  ```python
  elif wb_pval >= 0.05:
      row["verdict"] = "strong_equivalent"
  elif wb_pval >= 0.01 and rmsd_ratio < 1.5:
      row["verdict"] = "weak_equivalent"
  ```
- **REPLACE** with TOST-based routing (uses columns from A2):
  ```python
  def _pass(col):
      v = _safe_float(row.get(col))
      return math.isfinite(v) and v < 0.05

  procrustes_tost_1x_pass = _pass("procrustes_tost_pvalue_1x_bh")
  procrustes_tost_2x_pass = _pass("procrustes_tost_pvalue_2x_bh")
  metric_tost_1x_pass_rate = _compute_metric_tost_pass_rate(row, "1x")

  if procrustes_tost_1x_pass and metric_tost_1x_pass_rate >= 0.8:
      row["verdict"] = "strong_equivalent"
  elif procrustes_tost_2x_pass and metric_tost_1x_pass_rate >= 0.5:
      row["verdict"] = "weak_equivalent"
  elif procrustes_tost_2x_pass:
      row["verdict"] = "partial_match"
  else:
      row["verdict"] = "divergent"
  ```

**Atomicity**: A1+A2+A3+A4+A5 MUST land in ONE Codex task. Any subset
leaves the verdict logic broken.

### Group B: Metric tests (HIGH)

**B1 -- Two-sided Welch t-test per metric.**
- File: `fidelity_analysis.py:1535` (`add_metric_tests_to_row`)
- Add `scipy.stats.ttest_ind(orig, reimpl, equal_var=False, alternative="two-sided")`
  per metric.
- New columns `metric_welch_pvalue_{name}_raw` and `_bh`.
- Mirror in `scripts/fidelity_recompute_verdicts.py:161+`.
- Update `metric_test_columns()` at `:1122` and BH buckets at `:2529-2534`.

**B2 -- Expand `QUALITY_METRICS`** (user answer).
- File: `fidelity_analysis.py:38`
- New value:
  ```python
  QUALITY_METRICS: tuple[str, ...] = (
      "aspect_ratio",
      "dag_consistency",
      "edge_length_cv",
      "edge_straightness_mean_deg",
      "depth_spearman_rho",
      "overlap_count",              # quick() already computes; needs FIX-S for determinism
  )
  SAMPLED_QUALITY_METRICS: tuple[str, ...] = (
      "sampled_stress",             # already deterministic
      "crossing_rate",              # needs FIX-S
  )
  ```
- `load_layout()` at `:814-822` filters `quick()` output against
  `QUALITY_METRICS`. Widening the tuple is sufficient for the
  quick-metrics side.
- Depends on FIX-S landing first for `overlap_count` reproducibility.

**B2b -- Add sampled metrics to fidelity metric computation.**
- File: `fidelity_analysis.py:814-822`
- After the existing `quick()` call, also invoke:
  ```python
  from dagua.eval.pipeline_io import stable_seed
  layout_seed = stable_seed(record.graph_name, variant_id, side, str(record.seed or 0))
  sampled = compute_sampled_metrics(
      positions, edge_index, num_nodes=int(node_sizes.shape[0]),
      seed=layout_seed,
  )
  metrics.update({k: v for k, v in sampled.items() if k in SAMPLED_QUALITY_METRICS})
  ```
- The `compute_sampled_metrics` helper is added to pipeline_io in Wave 2
  (after FIX-S lands). B2b is therefore sequenced in Wave 2.
- Runtime cost per layout: ~10-50ms on top of existing quick(). At
  ~914k ok records that's ~2-12 hours for fidelity metric expansion
  alone. Make it opt-in via a `--with-sampled-metrics` CLI flag on
  `fidelity_analysis.py` (default: on).

**B3 -- Update `fidelity_add_metrics.py` + `fidelity_recompute_verdicts.py`.**
- File: `scripts/fidelity_add_metrics.py:38` -- import `QUALITY_METRICS`
  from `fidelity_analysis` instead of hardcoding.
- File: `scripts/fidelity_recompute_verdicts.py:46, :161` -- know about
  the expanded metric columns for test recomputation.

### Group C: Deterministic comparator (HIGH; depends on FID-B for Tier 3)

**C1 -- Three-tier deterministic comparator.**
- File: `fidelity_analysis.py:1884-1942` + `:2173-2191`
- Tier 1 -- raw position equality:
  ```python
  if orig_pos.shape == reimpl_pos.shape and torch.equal(orig_pos, reimpl_pos):
      verdict = "identical"
  ```
  The saved `.pt` tensors are stored in node-index row order
  (`run_benchmark.py:1453-1461, :182`). No sort needed.
- Tier 2 -- numeric near-equality after procrustes alignment
  (rotation + centering, NO scale normalization):
  ```python
  aligned_reimpl = procrustes_align(reimpl_pos, orig_pos, scale=False)
  if torch.allclose(orig_pos, aligned_reimpl, atol=1e-6, rtol=1e-4):
      verdict = "geometric_equivalent"
  ```
  Note: the existing `fidelity_procrustes()` at `:837-892` DOES include
  scale normalization. Add a `scale=False` path or a new
  `procrustes_align_rigid()` helper that does centering + rotation only.
- Tier 3 -- metric near-equality:
  ```python
  all_close = all(
      math.isclose(orig_metrics[m], reimpl_metrics[m], abs_tol=1e-6, rel_tol=1e-4)
      for m in QUALITY_METRICS
      if m in orig_metrics and m in reimpl_metrics
  )
  if all_close:
      verdict = "metric_equivalent"
  ```
  Iterates over `QUALITY_METRICS` -- depends on B2's expansion. Must
  land AFTER FID-B.
- Otherwise: existing procrustes/displacement heuristics with explicit
  rejection reasons.

**C2 -- Docstring update on deterministic routing.**
- File: `:2079, :2105, :2173`
- Severity: LOW.

### Group D: Procrustes math (MEDIUM)

**D1 -- Raise `PAIRWISE_SAMPLE_SIZE` from 10 to 30** (user answer).
- File: `:58`
- Affects pairwise sampling at `:1769, :1774`. Not procrustes-specific.

**D2 -- Carry `variant_id`, `reflected`, `max_node_displacement` into
`pairwise_similarity.csv`.**
- File: `PairwiseComparison` at `:169`; CSV writer at `:1788`.

**D3 -- Threshold normalization** (DEFERRED to follow-up after real
data).

### Group E: Failure causes preserved (HIGH)

**E1 -- Stop discarding non-`ok` records.**
- File: `ResultRecord` at `:106`; `build_variant_groups` at `:400`;
  `process_group` at `:1458, :1704, :1711`
- Extend `ResultRecord`:
  ```python
  error_message: str | None = None    # status == "error"
  skip_reason: str | None = None       # status == "skipped"
  ```
  (Both fields already exist in `results.json` per `run_benchmark.py:178-180, :190-191`.)
- In `process_group`, accumulate a `rejection_breakdown` dict with
  keys from the canonical enum (CQ11) PLUS scheduling-level reasons
  (`orig_error`, `orig_timeout`, `orig_skipped`, `reimpl_error`,
  `reimpl_timeout`, `reimpl_skipped`, `too_few_seeds`).
- Surface counts in `per_graph_detail.csv` and in the markdown
  failures section.

### Group F: Cleanup + report rewrite

**Cleanup1 -- Remove the `validate_sync()` hard gate.**
- File: `fidelity_analysis.py:2479-2499`
- Current: raises `sys.exit(1)` when `desync_count > 10`.
- Change: log the same message, write a telemetry file at
  `<output_dir>/data/validate_sync_telemetry.json`, do NOT exit.
- Severity: HIGH.

**Cleanup2 -- Markdown report rewrite** (CF2 critical bug).
- File: full rewrite of `scripts/generate_fidelity_report.py`
- Drop LaTeX entirely. Emit `eval_output/fidelity_report/report.md`.
- Inline:
  - Executive summary table (one row per algorithm family with verdict counts).
  - Failures section listing divergent / partial_match variants with
    one-line reasons sourced from `rejection_breakdown`.
- Read `per_seed_detail.csv` AND `pairwise_similarity.csv` for forensic
  detail.
- Surface new columns: `within_orig_rmsd_mean`, `reimpl_rmsd_mean`,
  `between_rmsd_mean`, `procrustes_tost_pvalue_1x_bh`, Welch columns,
  expanded metric columns.
- Update methodology text (rotation-only claim wrong, metric count now
  correct after B2, seed-count claim wrong).
- Dependencies: A (new procrustes columns), B (new metric columns),
  E (rejection_breakdown).

**Cleanup3 -- `family_tost_pass_rate()` NaN edge cases.**
- File: `generate_fidelity_report.py:279`
- Reuse finalized pass flag from `algorithm_summary.csv`.

**Cleanup4 -- Remove duplicate `pdflatex` in shell driver.**
- File: `run_fidelity_pipeline.sh:14`
- Drop after Cleanup2 lands.

**Cleanup5 -- Defer `compare_*.py` deletion.**
- Leave in place; `_final_run.py` and `_overnight.py` still reference
  them.

**Cleanup6 -- `merge_fidelity_csvs.py` README fix.**
- File: `:144`
- Preserve merged README, append merge note.

**Cleanup7 -- Wire `validate_fidelity_output.py` into shell driver.**
- File: `scripts/run_fidelity_pipeline.sh`
- Add post-analysis validator call.

### Group G: Family verdict docstring

**G1 -- Document threshold rule.**
- File: `fidelity_analysis.py:2257-2266`
- Docstring update only.

---

## PART 2: QUALITY/RUNTIME PIPELINE (NEW)

### Files

- `scripts/quality_runtime_analysis.py`
- `scripts/generate_quality_runtime_report.py`
- `scripts/run_quality_runtime_pipeline.sh`

All three import from `dagua/eval/pipeline_io.py`.

### Consolidation prerequisite

`positions.h5` is stale (2026-03-31). Two operational paths:

**Path A (recommended)**: run
`scripts/consolidate_positions_hdf5.py` after benchmark completes, then
run the analysis pipelines. One-time ~3 hour consolidation cost, then
75x faster analysis loads.

**Path B**: skip consolidation; QR-CORE loads directly from `.pt`
files. Slower but no consolidation step. QR-IO's loader already
supports this -- when `h5_file=None` or the record_key is missing from
HDF5, it falls back to `.pt`.

The v4 plan assumes Path A for the final run but tests work under
Path B with a small fixture h5 file.

### Data loading

- Read `results.json` into `records_df` with ALL statuses (running, ok,
  error, skipped, timeout). **Must read all statuses** -- filtering to
  `ok` only regresses the coverage denominator (Codex R3 dispatch risk).
- Join `manifest.json` graph metadata + tags + engine summaries.
- Use `dagua.eval.variants.algorithm_family()` for `engine_family`.
- Derive `graph_family`, `graph_size_token`, `graph_size_bucket`.
- Run `validate_sync()` as TELEMETRY only (reports to
  `validate_sync_telemetry.json`). No hard gate.
- Per row, call `load_position_tensor(record_key=..., positions_file=..., input_dir=..., h5_file=worker_h5)`.
  Keep the rejection reason in `records_df` for coverage accounting.

### Multiprocessing

```python
# scripts/quality_runtime_analysis.py

_worker_h5: "h5py.File | None" = None

def _worker_init(h5_path: Path):
    global _worker_h5
    from dagua.eval.pipeline_io import open_h5_for_worker
    _worker_h5 = open_h5_for_worker(h5_path)

def _worker_compute(task):
    global _worker_h5
    from dagua.eval.pipeline_io import (
        load_position_tensor, validate_positions, stable_seed,
        compute_quick_metrics_seeded, compute_sampled_metrics_seeded,
        aspect_ratio_deviation,
    )
    tensor, reason = load_position_tensor(
        record_key=task.record_key,
        positions_file=task.positions_file,
        input_dir=task.input_dir,
        h5_file=_worker_h5,
    )
    if tensor is None:
        return (task, None, reason)
    shape_reason = validate_positions(tensor, task.num_nodes)
    if shape_reason is not None:
        return (task, None, shape_reason)

    seed = stable_seed(task.graph_name, task.engine_name, str(task.layout_seed or 0))
    quick_metrics = compute_quick_metrics_seeded(
        tensor, task.edge_index, task.node_sizes,
        seed=seed, metric_filter=QR_QUICK_METRICS,
    )
    sampled_metrics = compute_sampled_metrics_seeded(
        tensor, task.edge_index, task.num_nodes,
        seed=seed,
        stress_sources=task.config.stress_sources,
        stress_targets=task.config.stress_targets,
        crossing_samples=task.config.crossing_samples,
    )
    quick_metrics["aspect_ratio_deviation"] = aspect_ratio_deviation(tensor)
    return (task, {**quick_metrics, **sampled_metrics}, None)

def run(input_dir, output_dir, workers):
    h5_path = input_dir / "positions.h5"
    with multiprocessing.Pool(
        workers,
        initializer=_worker_init,
        initargs=(h5_path,),
    ) as pool:
        for result in pool.imap_unordered(_worker_compute, tasks):
            ...
```

Key points:
- `h5_path` is a `Path`, picklable for initargs. The `h5py.File`
  handle is opened inside each worker and lives in a worker-local
  global.
- `compute_quick_metrics_seeded()` and `compute_sampled_metrics_seeded()`
  are added to pipeline_io in QR-CORE (Wave 2), not in QR-IO (Wave 0).
  They depend on FIX-S signatures.

### Seed budget (user answer)

- All successful seeds per stochastic engine (up to 60).
- Multiprocessing pool: `workers = max(1, os.cpu_count() - 2)`.
- **Realistic base**: ~914k ok rows. Each row: quick (~5ms) + sampled (~30ms).
  Sequential: ~9.5 hours. Parallel (8 cores): ~1.5-3 hours first run.
  Cached: minutes for re-runs.
- CLI flag `--max-nodes-for-sampled-metrics=5000` to skip sampled metrics
  on xlarge graphs (prevent one huge graph from eating the budget).

### Cache key strategy

Cache directory: `eval_output/quality_runtime_report/cache/`. Key is a
SHA-256 digest of:

- `record_key` (graph::engine::seed)
- metric name
- profile (`quick` or `sampled`)
- sampling config tuple `(stress_sources, stress_targets, crossing_samples)`
- `dagua/metrics.py` file content SHA-256
- FIX-S version tag (bump when seed policy changes)

CLI flags `--no-cache`, `--cache-dir PATH`, `--cache-invalidate`. Cache
format: one JSON file per `(record_key, profile)` pair; atomic write via
temp + rename.

**Known coarseness**: hashing the whole `metrics.py` file is broader
than necessary (every metric edit invalidates the entire cache). This
is acceptable for v1; the `--cache-invalidate` flag is the safety net
for finer-grained control. Transitive dependencies on other modules
(`dagua/utils.py`) are NOT in the hash; document that manual bust is
needed if those modules change.

### Metric sets

```python
# In scripts/quality_runtime_analysis.py

QR_QUICK_METRICS = {
    "edge_length_cv",              # already deterministic
    "dag_consistency",             # deterministic
    "depth_spearman_rho",          # deterministic
    "overlap_count",               # FIX-S seeded
    "edge_straightness_mean_deg",  # deterministic
    # aspect_ratio_deviation computed separately via pipeline_io helper
}

QR_SAMPLED_METRICS = {
    "sampled_stress",              # already deterministic
    "crossing_rate",               # FIX-S seeded
    "edge_crossings",              # FIX-S seeded (E > 500 branch)
}
```

### Graph family derivation

Tag-first with the EXPANDED tag set preserving real benchmark tags:

```
# Structural
hub-spoke -> hub_spoke
compound -> compound
tree -> tree
dependency -> dependency
small-world -> small_world
scale-free -> scale_free
community -> community
bipartite -> bipartite
grid|mesh|lattice -> grid
neural-net -> neural_net
geometric|spatial -> geometric

# Workload tags (from dagua/eval/graphs.py tag usage)
linear-shallow -> linear_shallow
linear-deep -> linear_deep
skip-light -> skip_light
skip-heavy -> skip_heavy
diamond -> diamond
nested-shallow -> nested_shallow
nested-deep -> nested_deep
mixed-width -> mixed_width
large-sparse -> large_sparse
large-dense -> large_dense
wide-layer|wide-parallel -> wide_layer

# Generic
erdos-renyi|random -> random
clustered -> clustered
cyclic -> cyclic
else -> misc
```

Specific tags win over generic. First match wins.

Size buckets: tiny (<20), small (20-99), medium (100-999),
large (1000-9999), xlarge (>=10000).

### Graph-relative ranking

**Primary**: per-graph rank. Secondary: clamped rel_best.

```python
def score_engines_on_graph(df, metric, higher_is_better):
    ascending = not higher_is_better
    ranked = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    ranked["graph_rank"] = range(1, len(ranked) + 1)

    best = ranked[metric].iloc[0]
    # Typical scale for normalization: use the family-wide distribution median
    # to avoid the near-zero explosion. Capped, not clamped.
    typical_scale = max(abs(ranked[metric].median()), 1e-3)

    def rel_best(value):
        if higher_is_better:
            gap = best - value
            denom = max(abs(best), typical_scale)
        else:
            gap = value - best
            denom = max(best, typical_scale)
        raw = gap / denom if denom > 0 else 0.0
        # Clamp to bound aggregation pathology. 10.0 chosen so a "10x worse
        # than the best engine" engine still sorts correctly; anything above
        # is clamped and the rank-based primary ordering carries the
        # differentiation.
        return min(max(raw, 0.0), 10.0)

    ranked["rel_best"] = ranked[metric].apply(rel_best)
    fastest = ranked["runtime_seconds"].min()
    ranked["runtime_rel_fastest"] = ranked["runtime_seconds"] / max(fastest, 1e-6)
    return ranked
```

**Known limitations** (documented, not fixed in v1):
- Clamp at 10.0 flattens any "10x+ worse" engine into the same rel_best
  value. The rank column carries true ordering; the report prints both
  rank and rel_best so the user can see when the clamp is active.
- Floor of 1e-3 on `typical_scale` flattens rel_best for families where
  all engines score near-zero. The per-family percentile printout
  (below) flags these cases.

For `dag_consistency` and `depth_spearman_rho`, rank-only mode: compute
`rel_best` for display but sort by rank only.

### Coverage denominator (CQ4)

```python
def compute_engine_coverage(records_df, graph_family_name):
    family_rows = records_df[records_df["graph_family"] == graph_family_name]

    scheduled = {}
    covered = {}
    for row in family_rows.itertuples():
        # Any row existing means the benchmark scheduled this (engine, graph).
        # Scheduler writes skipped rows for max_nodes caps, so the absence
        # of a row means the engine was not scheduled on this graph.
        scheduled.setdefault(row.engine_name, set()).add(row.graph_name)
        if row.status == "ok":
            covered.setdefault(row.engine_name, set()).add(row.graph_name)

    return {
        engine: {
            "graphs_scheduled": len(sched),
            "graphs_covered": len(covered.get(engine, set())),
            "coverage_ratio": len(covered.get(engine, set())) / max(len(sched), 1),
        }
        for engine, sched in scheduled.items()
    }
```

**Critical**: QR-CORE must iterate `records_df` INCLUDING non-ok rows,
or the scheduled set will be wrong.

Family-level context columns: `graphs_in_family_total` (from manifest)
and `graphs_in_family_available` (graphs where at least one engine
completed). These are for CONTEXT only; the engine-level coverage ratio
uses `covered / scheduled`.

Family inclusion gate (user answer): emit scorecard when
`graphs_in_family_available >= 3` AND the engine has
`coverage_ratio >= 0.5`.

### Pareto front

Per `(graph_family, metric_name)`:
- `x = median_runtime_rel_fastest`, `y = median_rel_best`.
- Both axes minimize. Ideal corner is **(1.0, 0.0)**.
- Dominance pruning with `epsilon = 1e-9`.
- Roles: `best_quality` (min y), `fastest` (min x),
  `balanced` (min Euclidean distance to (1.0, 0.0)),
  `dagua_anchor` (the literal `dagua` engine's point).

Inline Pareto (user answer): two tables per family -- `sampled_stress`
AND `crossing_rate`. Others to sidecar CSV.

PNG plots (user answer): ON by default. Log-x when runtime spread > 10x.
`--no-plots` disables.

Skip cyclic families (user answer): for `dag_consistency` and
`depth_spearman_rho`, exclude families whose name contains `"cyclic"`
OR where the family-level median of `dag_consistency` is < 0.5.

### Insights for dagua default

Per-metric thresholds with empirical percentiles printed alongside as
sanity check.

```python
THRESHOLDS = {
    "dag_consistency":            {"steal_abs": 0.05, "premium_abs": 0.10},
    "depth_spearman_rho":         {"steal_abs": 0.05, "premium_abs": 0.10},
    "edge_straightness_mean_deg": {"steal_abs": 3.0,  "premium_abs": 5.0},
    "overlap_count":              {"steal_abs": 5,    "premium_abs": 20},
    "sampled_stress":             {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1e-3},
    "edge_length_cv":             {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1e-3},
    "crossing_rate":              {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1e-4},
    "edge_crossings":             {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1.0},
    "aspect_ratio_deviation":     {"steal_abs": 0.10, "premium_abs": 0.30},
}
# All metrics additionally require runtime ratio <= 1.25 (steal) or <= 2.0 (premium).
```

Coverage filters: `graphs_covered_by_engine >= 3`,
`coverage_ratio_for_engine >= 0.5`, both dagua and competitor present.

Per-family diagnostic output: print p25/p50/p75 of each metric in the
report so the user can eyeball whether thresholds are reasonable for
that family. Future revision can switch to percentile-based insight
triggers.

### Output shape

`eval_output/quality_runtime_report/report.md`:
```
# Quality/Runtime Analysis
## Dataset Snapshot      (total / ok / error / skipped / timeout / running)
## Coverage              (per family: total / available / dagua_coverage_ratio)
## Family Scorecards     (one leader table per family + 1-3 bullets)
## Dagua Default Insights
## Best-of-Breed Configs
## Artifact Index
```

Sidecar CSVs:
- `analysis_records_snapshot.csv`
- `family_metric_summary.csv`
- `family_<family>__metric_<metric>__topk.csv`
- `family_<family>__metric_<metric>__pareto.csv`
- `family_<family>__summary.csv`
- `dagua_default_insights.csv`
- `best_of_breed_configs.csv`
- `artifact_index.csv`
- `validate_sync_telemetry.json`
- `family_<family>__metric_<metric>__pareto.png` (when plots ON)

### Test strategy

**`tests/test_pipeline_io.py`** (QR-IO):
- `stable_seed` multi-process reproducibility (spawn two subprocesses, assert
  identical return for the same input tuple).
- `load_position_tensor`:
  - HDF5 present, key present -> tensor, None
  - HDF5 present, key missing -> falls through to .pt; returns tensor
    if .pt present, else `load_failure`
  - **HDF5 present, key present, read raises** -> None, `h5_load_failure`
    (no fallback, matches current behavior)
  - HDF5 absent, .pt present -> tensor, None
  - Both absent -> None, `load_failure`
  - `positions_file=None` -> None, `missing_positions_file`
  - Corrupt .pt -> None, `load_failure`
  - Non-tensor .pt -> None, `not_tensor`
  - **HDF5 precedence over .pt** when both exist: assert the HDF5 tensor
    is returned (not the .pt tensor) to verify HDF5 is consulted first.
- `validate_positions` catches all 6 reasons.
- `aspect_ratio_deviation` known values.

**`tests/test_metric_seeding.py`** (FIX-S):
- `count_overlaps_detailed(pos, ns, seed=42)` called twice returns
  identical output.
- `count_overlaps_detailed(pos, ns, seed=None)` called 5 times on a
  graph with overlap ambiguity produces at least two different results.
- Same for `sampled_crossing_rate` and `count_crossings` (E > 500).
- Distinct seeds produce distinct results.
- `quick(pos, ei, ns, seed=42)` reproducible on `overlap_count` field.

**`tests/test_fidelity_procrustes.py`** (Group A):
- **Fixture 1 (known-good)**: 10 orig + 10 reimpl from same
  `N(pos_mean, 0.01)`. A1 + TOST at 1x should PASS.
- **Fixture 2 (known-bad)**: 10 orig from `N(pos_mean, 0.01)`, 10 reimpl
  from `N(pos_mean + 0.2, 0.01)`. A1 + TOST at 1x should FAIL; two-sided
  MWU should reject.
- **Fixture 3 (pooled-within regression)**: 10 orig from `N(pos_mean, 0.01)`,
  10 reimpl from `N(pos_mean + 0.15, 0.05)` -- same systematic offset
  but with HIGHER reimpl noise. Under the old pooled-within baseline,
  `within_rmsd` gets inflated by the reimpl noise, the between < within
  test passes (falsely accepting the reimpl). Under A1's fix (within =
  orig-only), the baseline is tight and the bias is detected.
  - Expected v3 verdict on this fixture: "strong_equivalent" (buggy)
  - Expected v4 verdict on this fixture: "divergent" (correct)
  - Test asserts that the v4 comparator returns "divergent".
- Procrustes TOST column population verified.
- A5 verdict routing exercised with all four outcomes.

**`tests/test_fidelity_deterministic.py`** (C1):
- Tier 1: identical tensors -> "identical"
- Tier 2: rotated near-identical -> "geometric_equivalent"
- Tier 3: different positions, matching metrics -> "metric_equivalent"
- No tier passes -> fallback verdict with explicit reason

**`tests/test_quality_runtime_analysis.py`** (QR-CORE):
- Fixture with all 5 statuses (running, ok, error, skipped, timeout).
- Fixture with negative `depth_spearman_rho`.
- Fixture with `overlap_count == 0` (zero-ideal case).
- Fixture with `sampled_stress` values near `1e-3` (tests the clamp).
- Asymmetric deterministic-orig / stochastic-reimpl.
- Coverage denominator uses scheduled-for-engine (fixture includes
  rows with `skipped` status from `max_nodes` cap).
- Pareto front dominance + role labels.
- Insight extraction with all 4 types.
- End-to-end smoke test.

---

## EXECUTION SEQUENCE (v4 DAG)

```
Wave 0a: QR-IO (single task, serial)
  └─> Wave 0b: FID-D, FID-G (parallel, both edit fidelity_analysis.py in different regions)
        └─> Wave 1: FID-S, FID-A, FID-E (parallel)
              └─> Wave 2: FID-B, QR-CORE (parallel; QR-CORE adds compute_*_metrics_seeded to pipeline_io)
                    └─> Wave 3: FID-C, FID-CLEANUP, QR-REPORT (parallel)
```

Changes from v3:
- **Wave 0 split into 0a + 0b** to avoid merge conflicts on
  `fidelity_analysis.py`. QR-IO is the heavy refactor and must land
  first. FID-D (1-line constant change) and FID-G (docstring) can then
  go in parallel on top of it.
- **QR-IO scope narrowed** to loader + seed + validate + aspect_ratio
  + open_h5. The `compute_quick_metrics_seeded` and
  `compute_sampled_metrics_seeded` helpers are moved to Wave 2 as part
  of QR-CORE (after FIX-S lands and `quick()` has a `seed` parameter).
- **Total waves: 5** (was 4 in v3). Realistic: QR-IO ~30 min
  + 0b ~15 min + Wave 1 ~30 min + Wave 2 ~40 min (QR-CORE is larger)
  + Wave 3 ~30 min = ~2.5 hours total dispatch time.

### Per-task boundaries

- **QR-IO** (Wave 0a): Build `dagua/eval/pipeline_io.py`. Move
  `stable_seed` and `validate_positions` from fidelity_analysis.
  Extract `load_position_tensor`. Add `open_h5_for_worker` and
  `aspect_ratio_deviation`. Refactor `load_layout()` to call the new
  `load_position_tensor()` helper. Update imports in
  `fidelity_recompute_verdicts.py`. Preserve behavior (equivalence
  test: run fidelity on a small fixture, diff output before/after).
  Tests in `tests/test_pipeline_io.py`.

- **FID-D** (Wave 0b): Raise `PAIRWISE_SAMPLE_SIZE` to 30. Extend
  `PairwiseComparison` dataclass (add `reflected` + `max_node_displacement`
  columns). Update CSV writer. Tests.

- **FID-G** (Wave 0b): Docstring update on family verdict threshold.
  Trivial.

- **FID-S** (Wave 1): Add `seed: int | None = None` parameters to
  `count_overlaps_detailed`, `sampled_crossing_rate`, `count_crossings`,
  and `quick` in `dagua/metrics.py`. Tests in
  `tests/test_metric_seeding.py`.

- **FID-A** (Wave 1): A1+A2+A3+A4+A5 atomically. Tests in
  `tests/test_fidelity_procrustes.py`.

- **FID-E** (Wave 1): Extend `ResultRecord` with `error_message` +
  `skip_reason`. Build `rejection_breakdown` in `process_group`. Tests.

- **FID-B** (Wave 2): B1 Welch + B2 metric expansion + B2b sampled
  metric integration + B3 update downstream scripts. Depends on FID-S
  (for seeded metrics) AND adds `compute_sampled_metrics_seeded` to
  pipeline_io (shared with QR-CORE).

- **QR-CORE** (Wave 2): `scripts/quality_runtime_analysis.py` +
  adding `compute_quick_metrics_seeded` + `compute_sampled_metrics_seeded`
  + `compute_pareto_front` + `extract_dagua_insights` helpers to
  pipeline_io. Depends on FID-S. Tests in
  `tests/test_quality_runtime_analysis.py`.

- **FID-C** (Wave 3): Three-tier deterministic comparator. Depends on
  FID-B (Tier 3 uses expanded QUALITY_METRICS). Tests in
  `tests/test_fidelity_deterministic.py`.

- **FID-CLEANUP** (Wave 3): Cleanup1 (validate_sync gate), Cleanup2
  (markdown rewrite), Cleanup3-7. Depends on FID-A, FID-B, FID-E columns.

- **QR-REPORT** (Wave 3): `scripts/generate_quality_runtime_report.py`
  + `run_quality_runtime_pipeline.sh` + golden file tests. Depends on
  QR-CORE.

### Post-implementation

1. Verify all tests green locally.
2. Run fidelity pipeline on a small fixture; diff against pre-refactor
   baseline; assert zero delta on QR-IO refactor.
3. Smoke-test both pipelines against the in-progress
   `variant_bench_full/` (currently 77.6%).
4. When benchmark finishes:
   - Run `scripts/consolidate_positions_hdf5.py` (~3 hours).
   - Run fidelity pipeline (minutes).
   - Run QR pipeline (~1.5-3 hours first run, cached after).

---

## OPEN QUESTIONS

All closed.

---

## RISKS

1. **FID-A atomicity**: A1+A2+A3+A4+A5 one task. Codex diff will be
   ~600-900 lines.
2. **B2/B2b depends on FIX-S**: if FIX-S is late, ship B2 with only
   deterministic metrics (`edge_straightness_mean_deg`,
   `depth_spearman_rho`) and defer stochastic metrics to follow-up.
3. **QR metric recompute cost**: ~914k ok rows × quick + sampled.
   Multiprocessed + cached: 1.5-3 hours first run. Document this.
4. **Cache coarseness**: whole-module hash is broad; transitive deps
   (`dagua/utils.py`) not covered. `--cache-invalidate` is the safety net.
5. **QR-IO refactor must preserve behavior**: run equivalence test
   before + after on a small fixture. Non-negotiable.
6. **Consolidation timing**: running `consolidate_positions_hdf5.py`
   takes ~3 hours on ~900k files. Build into the post-benchmark
   workflow, don't make the user rediscover it.
7. **Merge conflict risk**: resolved in v4 by splitting Wave 0. If
   future waves add more tasks touching fidelity_analysis.py, maintain
   single-task-per-wave discipline on that file.
8. **`graph_rel_best` clamp**: documented as a known coarseness, not a
   principled bound. v1 of QR ships with the clamp; v2 revisits if it
   hides real differences.

---

## FILE INVENTORY

New:
- `dagua/eval/pipeline_io.py`
- `scripts/quality_runtime_analysis.py`
- `scripts/generate_quality_runtime_report.py`
- `scripts/run_quality_runtime_pipeline.sh`
- `tests/test_pipeline_io.py`
- `tests/test_metric_seeding.py`
- `tests/test_fidelity_procrustes.py`
- `tests/test_fidelity_deterministic.py`
- `tests/test_quality_runtime_analysis.py`
- `tests/fixtures/quality_runtime/`

Modified:
- `dagua/metrics.py` (FIX-S)
- `scripts/fidelity_analysis.py` (A, B, C, D, E, G + QR-IO refactor)
- `scripts/generate_fidelity_report.py` (full rewrite)
- `scripts/run_fidelity_pipeline.sh` (wire validator, drop pdflatex)
- `scripts/validate_fidelity_output.py` (wired into shell driver)
- `scripts/fidelity_recompute_verdicts.py` (Welch mirror + stable_seed import)
- `scripts/fidelity_add_metrics.py` (import QUALITY_METRICS)
- `scripts/merge_fidelity_csvs.py` (README fix)

NOT deleted: `compare_classic.py`, `compare_reimpl_vs_original.py`.

---

## REVIEW HISTORY

- **v1**: Initial synthesis from 4 research passes.
- **v2**: Round 1 reviews + user answers + FIX-S + QR-IO module.
- **v3**: Round 2 reviews. Fixed stable_seed, validate_sync cleanup,
  Pareto axes, graph_rel_best, coverage denominator, aspect_ratio_deviation,
  cache key, Phase 2 DAG.
- **v4** (this): Round 3 reviews. Fixes:
  - **Path join bug** in pipeline_io: `input_dir / positions_file`
    (positions_file already includes the "positions/" prefix).
  - **Canonical rejection reason enum** matching the actual code
    (`missing_positions_file`, `h5_load_failure`, `load_failure`,
    `not_tensor`, `tensor_not_2d`, `tensor_not_xy`, `too_few_nodes`,
    `node_count_mismatch`, `contains_nan`, `contains_inf`). No
    invented strings.
  - **QR-IO scope narrowed** for Wave 0: loader + seed + validate +
    aspect_ratio + h5 helper. The seeded metric helpers move to QR-CORE
    in Wave 2 because they depend on FIX-S.
  - **Wave 0 split into 0a (QR-IO serial) + 0b (FID-D + FID-G parallel)**
    to avoid merge conflicts on fidelity_analysis.py.
  - **FID-C Tier 1 simplified**: direct `torch.equal(orig_pos, reimpl_pos)`
    with documented node-index-order assumption. No `sort_idx`.
  - **stable_seed has 4 consumers**, not 3. Both `fidelity_analysis.py`
    AND `fidelity_recompute_verdicts.py` update imports.
  - **positions.h5 staleness**: document the 9-day-old h5 file and the
    consolidation prerequisite. Pipeline works without consolidation
    via `.pt` fallback; consolidation is an optimization.
  - **Cleanup1 is real**: `validate_sync` IS called at
    `fidelity_analysis.py:2479-2499` as `sys.exit(1)` when
    `desync_count > 10`. Task is not a no-op.
  - **QR-CORE must read all statuses**: explicit risk note.
  - **Test fixtures expanded**: HDF5 precedence test, HDF5-read-error
    test (no fallback), known-bad pooled-within regression with
    specific parameter values (offset=0.15, reimpl_std=0.05).
  - **`graph_rel_best` documentation**: clamp at 10.0 is documented as
    a known coarseness, not principled. Rank is primary, rel_best is
    secondary display.
