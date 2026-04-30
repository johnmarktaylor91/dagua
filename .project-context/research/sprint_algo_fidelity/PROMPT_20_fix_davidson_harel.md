<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 20 ADVERSARIAL FIX for **davidson_harel**.

## SPEC

Your spec is the diff document at:
`.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_davidson_harel.md`

Read it end-to-end. The "Ranked Fix List" section has 5+ items. Apply
the top 3-4 highest-impact levers as a SINGLE bundle:

1. **Add fine-tuning phase**: igraph runs `maxiter` annealing rounds + `fineiter` fine-tuning rounds. Add `fineiter` to dagua pipeline; in fine-tuning phase, use radius `0.01 * min(span_x, span_y)` and disable uphill acceptance. Most importantly: **gate the node-edge energy term to fine-tuning ONLY** (igraph applies it only after `round >= maxiter`; dagua currently applies always). Expected delta: 0.04-0.10.
2. **Skip final centering** in igraph-fidelity mode: igraph does not recenter; dagua does in `FinalizeDHPositions`. Add a flag (default new behavior on) or split the op. Expected delta: 0.02-0.08.
3. **Replace full energy recomputation with incremental delta**: implement `_move_delta_energy` matching igraph's 5-block diff (node_dist, border, edge_lengths, edge_crossings, node_edge_dist). This naturally fixes phase gating. Expected delta: 0.03-0.08.

If the bundle is too large (>200 lines net), apply just #1 + #2 (the biggest two).

## Process

1. Read `ROUND_19_DIFF_davidson_harel.md` fully.
2. Multi-seed baseline (3 seeds, 5 small graphs):
   ```
   python scripts/algo_fidelity_live_compare.py classic_davidson_harel igraph_davidson_harel \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_20/davidson_harel/baseline
   ```
3. Apply the bundle of fixes. Cite line:line refs from the diff doc as comments only if helpful.
4. Run pytest on davidson_harel + layout suite.
5. Re-measure on the same subset.
6. COMMIT criterion: median improves by >= 0.05, OR aggregate TOST verdict moves up one tier.
7. If criterion met: commit on develop with `feat(fidelity): round 20 davidson_harel -- <short>`.
8. If criterion missed: revert all changes. Write `ROUND_20_RESIDUAL_davidson_harel.md`.
9. Per-round summary: `eval_output/algo_fidelity/round_20/davidson_harel/SUMMARY.md`.
10. Append iteration log row to `algo_fidelity_STATE.md` (be careful -- the parallel Round 20 codexes for other families will also append; use exclusive lock or just claim your row).

## Scope

**Allowed**:
- `dagua/layout/ops/davidson_harel.py`
- `dagua/layout/ops/pipelines/davidson_harel.py`
- `dagua/layout/ops/state.py` (new SolveState fields if needed)
- `eval_output/algo_fidelity/round_20/davidson_harel/**`
- `.project-context/research/sprint_algo_fidelity/ROUND_20_*davidson_harel*.md`
- `tests/test_layout/test_*davidson*.py` for snapshot updates

**Out of scope**:
- All other families' pipelines/ops
- Render, styles, scripts/graphviz_theme_comparison
- Cluster sprint files

## Verification
- `pytest tests/test_layout/ -x --tb=short -q -k "davidson"`
- `git diff --stat HEAD~0` shows only allowed scope

## Action safety
ONE commit on develop only IF measurable improvement. NO branch creation. NO force-push.
</task>

<scope_constraints>
See task body. davidson_harel files only. NO other family code.
</scope_constraints>
