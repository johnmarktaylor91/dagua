<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 20 ADVERSARIAL FIX for **neulay**.

## SPEC

Your spec is `.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_neulay.md`.
Read it end-to-end. 7-item ranked fix list.

The reference is the cloned `csabath95/NeuLay` repo at
`/home/jtaylor/projects/_references/NeuLay/`. The active dagua impl
matches the **old-code** variant (`old_code/NeuLay-2.py`).

Apply top 3 fixes from the ranked list:

1. **Old-code fidelity defaults**: when matching `old_code/NeuLay-2.py`,
   set `dim=3` (was 2), absolute query_radius=4 (currently scales with
   radius). Either expose a `fidelity_mode='old_code'` config flag or
   change defaults to match old code.
2. **Separate `steps` from `fdl_steps` semantics**: dagua treats `steps`
   as total budget; reference treats FDL/direct steps as separate
   post-GCN budget. Distinguish them in pipeline + competitor variant
   semantics.
3. **Fix KD-tree radius semantics**: add config path for absolute
   query radius (or set `query_radius_factor=10.0` to match old script's
   absolute radius=4 with default radius=0.4).

The neulay competitor adapter at `dagua/eval/competitors/neulay_competitor.py`
expects upstream `neulay`/`NeuLay` package. The cloned repo doesn't
have packaging metadata, but the diff doc may suggest installation
options. If installable: try it, document. If not: the fidelity
target is the cloned source, compared against dagua's classic_neulay
output (single-seed since multi-seed comparison currently can't run
without the upstream package).

## Process

1. Read `ROUND_19_DIFF_neulay.md` fully (especially sections 1, 2, 13, 14).
2. Baseline (cached comparison or direct port):
   ```
   python scripts/algo_fidelity_live_compare.py classic_neulay neulay \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_20/neulay/baseline
   ```
   This may return mostly errors if upstream package unavailable.
   Document.
3. Apply top 3 fixes.
4. Tests: `pytest tests/test_layout/ -x --tb=short -q -k "neulay"`.
5. Re-measure (or note source_unavailable).
6. COMMIT criterion: median improves OR (if upstream unavailable)
   the fidelity-mode toggle is added cleanly with documentation.
7. Commit `feat(fidelity): round 20 neulay -- <short>` if met.

## Scope

**Allowed**:
- `dagua/layout/ops/neulay.py`
- `dagua/layout/ops/pipelines/neulay.py`
- `eval_output/algo_fidelity/round_20/neulay/**`
- `.project-context/research/sprint_algo_fidelity/ROUND_20_*neulay*.md`
- `tests/test_layout/test_*neulay*.py`

**Out of scope**:
- Other family pipelines/ops
- The `dagua/eval/competitors/neulay_competitor.py` (don't touch the adapter)

## Verification
- pytest neulay tests pass
- live_compare attempts at least

ONE commit only IF improvement OR clean fidelity-mode addition.
</task>

<scope_constraints>neulay files only.</scope_constraints>
