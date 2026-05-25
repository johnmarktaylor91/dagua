<task>
R32 GEM deep port — the user has explicitly requested this. "Leave no stone unturned. NOTHING deferred."

## Background

gem (Frick/Ludwig/Mehldau 1995) currently has 3 variants strong_equivalent in the 100-seed report (RMSD 0.13-0.22 vs ogdf_gem). Within stochastic floor at 1x margin. But the R31 SUMMARY noted "architectural floor with init bit-exact": post-init divergences remain.

R30 codex (commit aba48d6) implemented `_glibc_rand_values` + `_ogdf_runner_initial_positions` for bit-exact INIT. Post-init algorithm work was deferred as "1000s of lines of OGDF C++."

User said: "I don't see why it's so hard to read the c++ code."

## Your job

Read OGDF GEMLayout.cpp end-to-end and port the remaining divergent semantics to dagua's GEM. Specifically:

### Phase 1: Read

- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp` (full)
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h` (full)
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/Random*.cpp` (for RNG)
- `/home/jtaylor/projects/dagua/dagua/layout/ops/gem.py` (full)
- `/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/gem.py` (full)

### Phase 2: Identify divergences post-init

The init is already bit-exact (R30 work). Find the iteration loop divergences:
- Node update order: how does OGDF permute nodes per iteration? (probably linked list / Skiplist / RNG-driven permutation)
- Cooling schedule: temperature, oscillation, rotation thresholds
- Connected-component handling: how does OGDF split + lay out + pack components
- Final geometry transforms: axis-align, center, scale

### Phase 3: Implement under fidelity_mode

Add `fidelity_mode="ogdf"` to gem pipeline. When enabled:
- Use OGDF-bit-exact permutation order (port the data structure)
- Match cooling/temperature decay exactly
- Match component packing logic
- Match final geometry transforms

### Phase 4: Verify

```bash
python scripts/algo_fidelity_live_compare.py classic_gem ogdf_gem --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_32/gem/post_impl
```

Expected: gem variants currently strong_equivalent at 0.13-0.22 -> push toward 0.05-0.10 or lower under fidelity_mode.

## Scope

- DO NOT TOUCH: render/styles, cluster sprint, existing benchmark outputs
- Multi-commit OK; each commit small.
- This may be a multi-hour codex effort. Acceptable.

## Output

`eval_output/algo_fidelity/round_32/gem/SUMMARY.md` with:
- List of divergences identified
- Each ported divergence with file:line on both sides
- Before/after RMSD on bounded subset
- Any remaining residuals (be honest if some are truly impossible)
</task>

<completeness_contract>
Port AT LEAST the node-permutation order (highest-impact item per R20 codex's diagnosis). Component packing + geometry transforms if time permits.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Read deeply.
</default_follow_through_policy>
