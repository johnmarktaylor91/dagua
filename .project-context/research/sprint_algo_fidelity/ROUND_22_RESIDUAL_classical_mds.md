# Round 22 Residual: `classic_classical_mds` vs `igraph_mds`

Date: 2026-04-30

## Outcome

No source fix was committed for `classical_mds`.

The requested baseline and after measurements both ran cleanly on the specified
five-graph, three-seed subset, and both reported median RMSD `0.000000`.
The intended fix qualified only under the opt-in-fidelity-mode criterion, but
the source edits could not be kept stable in this shared worktree.

## Attempted Fix Bundle

I followed the smaller staged scope recommended in
`ROUND_21_DIFF_classical_mds.md` rather than implementing DLA component merge:

- RNG measurement correctness for `igraph_mds`, based on the recommendation
  that igraph's disconnected MDS merge uses `RNG_UNIF()` while the adapter did
  not seed it (`ROUND_21_DIFF_classical_mds.md:354-361`,
  `ROUND_21_DIFF_classical_mds.md:412-419`).
- Opt-in igraph-compatible connected embedding semantics: largest algebraic
  eigenvalues, `sqrt(abs(lambda))`, and reversed output columns
  (`ROUND_21_DIFF_classical_mds.md:362-366`,
  `ROUND_21_DIFF_classical_mds.md:94-132`).
- Opt-in two-node raw special case matching igraph's `[0, 0]`, `[1, 1]`
  behavior (`ROUND_21_DIFF_classical_mds.md:367-369`,
  `ROUND_21_DIFF_classical_mds.md:214-220`).

## Blocker

Multiple concurrent Codex processes were active in the same checkout for other
Round 22 families. After applying the `classical_mds` source edits, the modified
files were repeatedly rewritten back within seconds. A direct timestamp check
confirmed `dagua/layout/ops/embed.py` changed after a `sleep 2` without any
local command targeting that file.

Because the source edits were externally reverted, the new regression test file
became inconsistent with the actual source and was removed before finishing.
No `classical_mds` source changes were left behind and no commit was made.

## Measurement

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_classical_mds igraph_mds \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/classical_mds/baseline
```

After command:

```bash
python scripts/algo_fidelity_live_compare.py classic_classical_mds igraph_mds \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/classical_mds/after
```

Both runs:

- graphs: `5`
- median: `0.000000`
- p25: `0.000000`
- p75: `0.000000`
- p95: `0.000000`
- worst: `parallel_multiedge_bundle 0.000000`

## Test Results

The initial family selector passed while the source edits were present:

```text
pytest tests/test_layout/ -x --tb=short -q -k "classical_mds"
...                                                                      [100%]
3 passed, 249 deselected in 0.20s
```

Later pytest runs were blocked by unrelated concurrent work:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
ERROR tests/test_layout/test_fa2_fidelity.py
ImportError: cannot import name '_FA2_REFERENCE_PACKAGE_ORDER'
```

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

An additional import-time blocker appeared from concurrent FMMM edits:

```text
NameError: name '_GALAXY_CHOICE_HIGHER' is not defined
```

## Recommended Next Step

Re-run this round in an isolated worktree or after the other Round 22 family
agents finish. The smallest viable patch is still the staged opt-in bundle
above; full igraph DLA component merge should remain out of scope for that
small round.
