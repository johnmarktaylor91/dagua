NEW SESSION: Read this file first, then CLAUDE.md and AGENTS.md.

## Mission

THIS IS THE LAST RUN. Fix EVERY remaining code difference between dagua's
reimplementations and their references, re-benchmark, re-analyze, produce
final fidelity report. No prioritization, no "medium priority" deferrals.
Fix everything.

User directive (verbatim): "can you please stop the current run, match
EVERY LAST THING, make sure the full pipeline beginning to end is ready
to rerun those specific changes without deleting other progress or wasting
time, then kick it off again? Please be 100% sure you are doing everything
you can possibly do to make this the last time!!!"

## What's Already Done

- 97 algorithm families benchmarked (510K+ evals)
- 77 families are strong_equivalent/identical -- DONE, don't touch
- 20 reimpl variants have code fixes applied and were re-benchmarked
- positions.h5 has fresh positions for all 20 variants (consolidated)
- Existing CSVs at eval_output/fidelity_report/data/ have correct data
  for the 77 unchanged families (from overnight run with mirror-aware
  Procrustes)
- The 20 changed families have BAD data in CSVs (from overnight run
  where H5 was desynced -- positions were empty)

## Fixes Already Applied (verified in source)

1. FA2 linlog: `-(torch.log1p(distance) / distance).squeeze(1)` (fa2.py:487)
2. SGD2 multi LR: `lr: float = 1.0` (sgd2_multi.py:1065)
3. SGD2 multi BCE: `reduction='sum'` (sgd2_multi.py:886)
4. SGD2 multi LR schedule: ExponentialLR gamma=0.993, step every iter (sgd2_multi.py:1141-1171)
5. t-SNE early exag: `early_exaggeration_steps = 250` (tsnet.py:371)
6. t-SNE binary search: `range(100)` (tsnet.py:184)
7. NeuLay GCN: 3-layer residual (100->100->3, skip concat 203->2) (neulay.py:372-435)
8. NeuLay magnitude: `100.0 * N^(1/3) * radius` (neulay.py:562-564)
9. NeuLay optimizer: RMSprop, _GNN_LR=0.01 (neulay.py:28,350,472)
10. Procrustes: mirror-aware, best-of-two rotations (fidelity_analysis.py:885-891)
11. _safe_float: handles empty CSV strings (fidelity_analysis.py:2003-2011)

## Remaining Fixes to Apply

### SGD2 Multi (dagua/layout/classic/sgd2_multi.py)

**S1. Batch sampling: random with replacement -> full-epoch sweep**
- Current: `torch.randint(0, total, (batch_size,), device=device)` (~line 491)
- Reference: iterates ALL pairs every epoch in sequential order (DataLoader-style)
- Fix: create a permutation of all pairs at epoch start, iterate in batch chunks,
  reshuffle when exhausted

**S2. Weight function: include self-loops (d=0)**
- Current: filters `positive_distances > 0` (~line 322)
- Reference: includes all pairs including d=0 with epsilon handling
- Fix: remove the `> 0` filter, add epsilon to denominator instead

**S3. Distance handling for disconnected graphs**
- Current: replaces inf distances with `max_distance + 1.0` (~line 290-295)
- Reference: rejects disconnected graphs entirely
- Fix: add a check that raises or warns on disconnected graphs, matching
  reference behavior. For benchmark purposes this is fine since test graphs
  include disconnected ones that should be skipped.

**S4. Epsilon usage**
- Current: `_EPS = 1.0e-6` applied to `distances.square() + _EPS` (~line 20, 322)
- Reference: `eps=0.01` controls LR floor
- Fix: verify these are different parameters (distance stability vs LR floor).
  If _EPS is only for numerical stability and doesn't affect results, leave it.
  If it changes the weight function shape, match reference value.

### NeuLay (dagua/layout/classic/neulay.py)

**N1. Early stopping formula**
- Current: two rolling windows (_SHORT_STOP_WINDOW=32, _LONG_STOP_WINDOW=1000)
  with ratio thresholds (_SHORT_STOP_RATIO=5e-4, _LONG_STOP_RATIO=1e-4) (~line 276-316)
- Reference: different window metrics and N-scaling
- Fix: read the upstream NeuLay reference at /tmp/neulay_pkg/neulay/core.py
  (or pip show neulay to find it), copy the exact early stopping logic

**N2. Initialization distribution**
- Current: `torch.randn(...) * scale` where scale=sqrt(N) (~line 194)
- Reference: may use different distribution or scale
- Fix: read upstream init code, match exactly

**N3. Coordinate centering**
- Current: centers positions after every optimization step (~line 364)
- Reference: allows natural drift without forced centering
- Fix: remove the per-step centering. Center only at the end if needed
  for output normalization.

**N4. Self-loop handling**
- Current: explicitly removes self-loops in _clean_edge_index() (~line 155-173)
- Reference: may handle differently
- Fix: check reference, match behavior

### FA2 (dagua/layout/classic/fa2.py)

**F1. Gravity target**
- Current: pulls toward centroid of current positions (~line 174-180)
- Reference: pulls toward origin (0,0)
- NOTE: Other FA2 variants (non-linlog) already pass as strong_equivalent.
  This means either the gravity difference doesn't matter OR it's correct.
  CHECK: if all non-linlog FA2 variants are strong, this difference is
  acceptable. Only fix if it causes fa2_linlog to fail.

### t-SNE (dagua/layout/classic/tsnet.py)

**T1. No remaining differences found.** The audit confirmed tsnet.py matches
sklearn's implementation. The two previously fixed issues (early_exag, binary
search) were the only diffs.

## Pipeline for the Final Run

### Step 1: Apply all remaining fixes above
- Read each file, make the changes
- Clear pycache: `find dagua scripts -name '__pycache__' -type d -exec rm -rf {} +`
- Verify fixes in source (grep for key patterns)
- Lint: `ruff check dagua/layout/classic/sgd2_multi.py dagua/layout/classic/neulay.py`

### Step 2: Purge ONLY the changed variants from results.json
Use the safe purge script:
```bash
python scripts/safe_purge_variants.py \
  classic_sgd2_multi_batch128 classic_sgd2_multi_batch8 \
  classic_sgd2_multi_default classic_sgd2_multi_lr001 \
  classic_sgd2_multi_lr01 classic_sgd2_multi_stress_only \
  classic_sgd2_multi_with_aspect classic_sgd2_multi_with_crossing \
  classic_neulay_default classic_neulay_lr001 classic_neulay_lr05 \
  classic_neulay_no_gcn classic_neulay_radius02 classic_neulay_radius08 \
  --confirm
```
DO NOT purge fa2_linlog or tsnet variants -- those fixes are already
benchmarked and have correct positions in H5.

### Step 3: Re-benchmark ONLY SGD2 + NeuLay (14 engines)
```bash
python scripts/run_benchmark.py --resume --variants \
  --output-dir eval_output/variant_bench_full \
  --workers 4 --seeds 30 --timeout 120 \
  --engines classic_sgd2_multi_batch128,classic_sgd2_multi_batch8,classic_sgd2_multi_default,classic_sgd2_multi_lr001,classic_sgd2_multi_lr01,classic_sgd2_multi_stress_only,classic_sgd2_multi_with_aspect,classic_sgd2_multi_with_crossing,classic_neulay_default,classic_neulay_lr001,classic_neulay_lr05,classic_neulay_no_gcn,classic_neulay_radius02,classic_neulay_radius08
```

### Step 4: Consolidate new .pt files into H5
```bash
python scripts/consolidate_positions_hdf5.py \
  --input eval_output/variant_bench_full \
  --output eval_output/variant_bench_full/positions.h5
```

### Step 5: Validate benchmark integrity
```bash
python scripts/validate_benchmark_integrity.py \
  --data-dir eval_output/variant_bench_full
```
Must pass. If it fails, fix the desync before proceeding.

### Step 6: Run fidelity analysis on changed families only
Create filtered results.json with only the 20 changed families (SGD2 +
NeuLay + FA2 linlog + t-SNE -- all 20 because the existing CSVs have
bad data for all 20 from the desynced overnight run):
```bash
# Filter script already exists at /tmp/filtered_bench/
# Re-create it with updated results.json after purge + re-benchmark
```
Run analysis on filtered input:
```bash
python scripts/fidelity_analysis.py \
  --input /tmp/filtered_bench \
  --output /tmp/fidelity_changed
```

### Step 7: Merge into existing CSVs
```bash
python scripts/merge_fidelity_csvs.py \
  --existing eval_output/fidelity_report/data \
  --partial /tmp/fidelity_changed \
  --output eval_output/fidelity_report/data \
  --families fa2_linlog,neulay_default,neulay_lr001,neulay_lr05,neulay_no_gcn,neulay_radius02,neulay_radius08,sgd2_multi_batch128,sgd2_multi_batch8,sgd2_multi_default,sgd2_multi_lr001,sgd2_multi_lr01,sgd2_multi_stress_only,sgd2_multi_with_aspect,sgd2_multi_with_crossing,tsnet_default,tsnet_perp5,tsnet_perp50,tsnet_steps200,tsnet_steps2000
```

### Step 8: Recompute verdicts
```bash
python scripts/fidelity_recompute_verdicts.py \
  --data eval_output/fidelity_report/data
```

### Step 9: Validate output with delta comparison
```bash
python scripts/validate_fidelity_output.py \
  --data eval_output/fidelity_report/data \
  --previous /tmp/fidelity_previous
```
MUST show changes for the 20 families. If output is identical, STOP --
something is wrong.

### Step 10: Generate PDF + verdict breakdown
```bash
python scripts/generate_fidelity_report.py \
  --data eval_output/fidelity_report/data \
  --output eval_output/fidelity_report
```

### Step 11: Commit everything

## Time Estimates

- Step 1 (fixes): 30 min
- Step 2 (purge): 2 min
- Step 3 (benchmark 14 engines): ~2 hrs
- Step 4 (consolidate): 30 min
- Step 5 (validate): 1 min
- Step 6 (analysis ~1800 groups): ~1-2 hrs
- Step 7 (merge): 1 min
- Step 8 (recompute): 12 min
- Step 9-10 (validate + PDF): 2 min
- **Total: ~4-5 hours**

## Critical Rules

1. DO NOT re-run unchanged families. The 77 unchanged families have
   correct data in the existing CSVs.
2. DO NOT purge positions.h5 manually. Use safe_purge_variants.py.
3. After benchmark, ALWAYS consolidate .pt -> H5 before analysis.
4. After analysis, ALWAYS run validate_fidelity_output.py with --previous.
5. If ANY validation step fails, STOP and investigate. Do not proceed.
6. Fix ALL differences, not just "critical" ones. No prioritization.

## Enforcement Scripts (use them)

- `scripts/validate_benchmark_integrity.py` -- results.json/H5 sync check
- `scripts/validate_fidelity_output.py` -- output sanity + delta comparison
- `scripts/safe_purge_variants.py` -- atomic purge of both data stores
- `scripts/merge_fidelity_csvs.py` -- merge partial into existing CSVs

## Key Files

| File | What |
|------|------|
| dagua/layout/classic/sgd2_multi.py | SGD2 reimpl (needs S1-S4 fixes) |
| dagua/layout/classic/neulay.py | NeuLay reimpl (needs N1-N4 fixes) |
| dagua/layout/classic/tsnet.py | t-SNE reimpl (DONE, no remaining diffs) |
| dagua/layout/classic/fa2.py | FA2 reimpl (DONE, check F1 if linlog still fails) |
| scripts/fidelity_analysis.py | Main analysis (has integrity gate) |
| scripts/fidelity_recompute_verdicts.py | Fast verdict recomputer |
| scripts/generate_fidelity_report.py | PDF report (compact version) |
| eval_output/fidelity_report/data/ | Current CSVs (77 families correct, 20 need replacing) |
| eval_output/variant_bench_full/ | Benchmark data (results.json + positions/) |

## Previous Verdict Breakdown (baseline to compare against)

74 strong_equivalent, 11 weak_equivalent, 2 partial_match, 10 divergent

The 20 non-strong families are the ones we're fixing. After this run,
the number of strong families should INCREASE. If it doesn't, the fixes
didn't work and further investigation is needed.
