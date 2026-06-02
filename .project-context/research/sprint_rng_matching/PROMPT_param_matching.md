<task>
Make the fidelity comparison use MATCHED PARAMETERS on both sides, across ALL engines. Right
now some reference adapters run the reference at ITS DEFAULTS instead of the dagua variant's
configuration, so the RMSD is meaningless (e.g. gem: dagua runs 100 rounds, the OGDF reference
runs its default 30000 -> spurious 1.15 RMSD even though they bit-match at matched rounds).

Project: /home/jtaylor/projects/dagua.

## What "matched params" means
Each variant in `dagua/eval/variants.py` is `_variant(variant_id, base_engine, display_name,
reimpl_params, original_engine, original_params, ...)`. `reimpl_params` configures dagua;
`original_params` configures the REFERENCE. For a valid bit-exact comparison the reference must
run the SAME algorithmic configuration as dagua: same iteration/round/step count, same
perplexity / n_neighbors / spring constants / cooling, translated into the reference's own
parameter names. The reference ADAPTER must then actually PASS those params to the reference call.

## Steps
1. AUDIT every `classic_*` variant: produce a table (variant_id, reimpl_params, original_params,
   reference_engine) and flag every case where original_params does NOT mirror reimpl_params
   (e.g. dagua steps=100 but reference gets {} -> runs default). Core deliverable.
2. For each reference ADAPTER, verify it passes the params to the actual reference:
   - dagua/eval/competitors/graphviz_competitor.py (neato/sfdp/fdp): does it pass maxiter/steps/
     K/etc as graphviz -G attributes? It already passes -Gseed/-Gstart; extend to each variant's
     algorithmic params.
   - igraph adapters (drl/davidson_harel/lgl/sugiyama/kk/mds): pass maxiter/rounds/etc to the
     igraph layout call.
   - OGDF runner (scripts/ogdf_runner.cpp + its python adapter): gem rounds, fmmm steps, stress
     iters, pivot_mds pivots, maxent params -- the runner currently IGNORES some (gem rounds).
     Fix the runner to accept+apply them (edit C++, rebuild per its build script) OR, if rebuild
     is infeasible here, DOCUMENT exactly which params it can't honor.
   - fa2 / tsne / umap / nx adapters: pass steps/perplexity/n_neighbors/min_dist/etc.
3. FIX: update variants.py original_params to mirror each variant's reimpl_params (correct
   reference param names) and update adapters to pass them through. Where a param has no
   reference equivalent, note it.
4. RE-VERIFY: run `python scripts/rng_match/bitexact_harness.py` (full) after the fix; report
   which engines' RMSD improved (divergence was param-mismatch) vs still-divergent (real
   RNG/arithmetic gap left for the port codexes).
</task>

<constraints>
- 6 per-engine PORT codexes run IN PARALLEL editing pipeline files under dagua/layout/ops/.
  You must edit ONLY: dagua/eval/variants.py, dagua/eval/competitors/*, scripts/ogdf_runner.cpp
  (+ its build). Do NOT touch anything under dagua/layout/ops/ (avoids parallel-edit conflicts).
- This is COMPARISON-CONFIG only. Do not change any layout algorithm/pipeline. Do not relax any
  fidelity bar. Matching params must make the reference run dagua's config -- never the reverse
  (don't change dagua's variant params to match a wrong reference default).
- Do NOT commit (CC commits after review). Report the audit table + before/after harness numbers.
</constraints>

<verification>
- Re-run the harness; gem should now show iters100/iters500 bit-exact (<1e-7) if you made the
  OGDF runner honor rounds. Report the full per-engine before/after.
- Confirm you changed NO files under dagua/layout/ops/.
</verification>

<default_follow_through_policy>
Proceed autonomously. Hard stop only if the OGDF runner rebuild is structurally infeasible in
this environment -- then document precisely which reference params can't be honored and why,
and fix everything else.
</default_follow_through_policy>
