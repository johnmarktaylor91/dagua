<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 9 of the algo_fidelity sprint. Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `.project-context/research/sprint_algo_fidelity/ROUND_8_RE_EVAL.md`
3. `eval_output/algo_fidelity/round_8/SUMMARY.md`

## Round 9 context

Round 8 built the multi-seed comparator and tried to re-evaluate fdp/sfdp
under stochastic-floor lens. The TOST verdict came back:
- fdp: 18/18 testable graphs `not_equivalent`
- sfdp: 21/21 testable graphs `not_equivalent`

**BUT THE TEST WAS BROKEN**: the cached "9 seeds per graphviz engine" in
`eval_output/benchmark_full/positions/<graph>__graphviz_<engine>__seed42.pt`
through `seed50.pt` are ALL IDENTICAL. The within-graphviz "floor" was
measured as <0.000001 because graphviz produced the same layout for all
9 nominal seeds.

Root cause: `dagua/eval/competitors/graphviz_competitor.py` accepts a
`seed: Optional[int]` parameter then literally drops it:
- Line 342 (`GraphvizDot.layout`): `del seed`
- Line 402 (`_GraphvizBase.layout`): `del seed`

So when the benchmark harness runs graphviz with seeds 42..50, all 9
runs use graphviz's default fixed seed (seed=1 for fdp per
`lib/fdpgen/tlayout.c:100  #define DFLT_seed 1`).

Round 9 fixes this and regenerates a true stochastic graphviz cache,
then re-runs Round 8's multi-seed analysis. If fdp/sfdp now show
equivalent_at_<=2x within-graphviz floor, they get reclassified as
faithful. Otherwise the architectural divergence is confirmed.

## Graphviz seed mechanism (CRITICAL -- READ SOURCE)

Graphviz seeds are passed via DOT graph attributes:

- **fdp**: `lib/fdpgen/tlayout.c:97-105`
  ```c
  #define DFLT_seed 1
  T_seed = late_int(g, agattr_text(g, AGRAPH, "seed", NULL), DFLT_seed, MIN_SEED);
  ```
  -> Add `seed=N` graph attribute in DOT input. Default 1.

- **sfdp**: `lib/sfdpgen/spring_electrical.c:62`
  ```c
  ctrl.random_seed = 123;
  ```
  Plus `lib/sfdpgen/sfdpinit.c` parses the `start=` attribute.
  -> Add `start=<int>` graph attribute. Default 123.

- **neato**: `lib/neatogen/neatoinit.c:944`
  ```c
  if (init == INIT_RANDOM) {
      // setSeed reads "start" attribute
      ...
  }
  ```
  -> Add `start=<int>` graph attribute. Same as sfdp.

The DOT input format for graph-level attributes is:
```
digraph G {
  start=42
  seed=42
  ...
}
```

(Or pass `-Gstart=42 -Gseed=42` on the command line.)

## What to do

### Step 1: Fix graphviz competitor seed plumbing (15-30 min)

In `dagua/eval/competitors/graphviz_competitor.py`:
- For each of `GraphvizDot`, `GraphvizSfdp`, `GraphvizNeato`, `GraphvizFdp`,
  remove the `del seed` and instead pass the seed through to the
  graphviz subprocess via either:
  - Adding `seed=<N>` and `start=<N>` to the DOT input graph
    attributes block, OR
  - Adding `-Gseed=<N> -Gstart=<N>` to the dot/neato/fdp/sfdp command line
- For dot specifically: dot is **deterministic** -- seed has no effect.
  So `GraphvizDot` keeps `del seed` (with a comment explaining why).
- For sfdp/fdp/neato: thread the seed through. Use seed for both
  `seed` (fdp uses this) and `start` (neato/sfdp uses this) attributes
  -- doesn't hurt to pass both.

The seed must propagate to the actual graphviz binary invocation.
Verify by running:
```
dot -Tjson -Kfdp -Gseed=42 <input> > /tmp/out_seed42.json
dot -Tjson -Kfdp -Gseed=43 <input> > /tmp/out_seed43.json
diff /tmp/out_seed42.json /tmp/out_seed43.json
```
The two outputs should differ on a moderately-sized graph (e.g.,
petersen_10 from `eval_output/algo_fidelity/round_2/baseline/panels/`).
If they're still identical, the seed isn't reaching the binary.

### Step 2: Generate fresh multi-seed graphviz cache (15-30 min)

Build a small script `scripts/regen_graphviz_seeds.py` (or extend
existing benchmark infra) that:
- For each graph in the test set (`dagua.eval.graphs.get_test_graphs()`)
- For each stochastic graphviz engine (graphviz_fdp, graphviz_sfdp,
  graphviz_neato)
- For seeds 42..50 (9 seeds each)
- Runs the now-fixed graphviz competitor
- Saves position to a NEW directory:
  `eval_output/algo_fidelity/round_9/graphviz_seeded_cache/<graph>__<engine>__seed<S>.pt`

Do NOT overwrite the existing benchmark_full cache (preserves history).

### Step 3: Re-run Round 8 multi-seed comparator (10 min)

The Round 8 comparator (`scripts/algo_fidelity_live_compare.py --seeds 5`)
loads cached graphviz positions from `--input-dir <path>/positions/`.
Add a `--graphviz-cache-dir <path>` flag (or similar) so it can read
from the new round_9 cache instead of benchmark_full.

Run:
```
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --seeds 5 --graphviz-cache-dir eval_output/algo_fidelity/round_9/graphviz_seeded_cache \
    --output-dir eval_output/algo_fidelity/round_9/fdp_re_eval

python scripts/algo_fidelity_live_compare.py classic_sfdp graphviz_sfdp \
    --seeds 5 --graphviz-cache-dir eval_output/algo_fidelity/round_9/graphviz_seeded_cache \
    --output-dir eval_output/algo_fidelity/round_9/sfdp_re_eval

python scripts/algo_fidelity_live_compare.py classic_stress_maj graphviz_neato \
    --seeds 5 --graphviz-cache-dir eval_output/algo_fidelity/round_9/graphviz_seeded_cache \
    --output-dir eval_output/algo_fidelity/round_9/neato_stress_re_eval

python scripts/algo_fidelity_live_compare.py classic_classical_mds graphviz_neato \
    --seeds 5 --graphviz-cache-dir eval_output/algo_fidelity/round_9/graphviz_seeded_cache \
    --output-dir eval_output/algo_fidelity/round_9/neato_mds_re_eval
```

Confirm within-graphviz floor is no longer ~0 (should be 0.05+ for
genuinely stochastic engines on graphs with cycles/symmetry).

### Step 4: Write re-evaluation report

`.project-context/research/sprint_algo_fidelity/ROUND_9_RE_EVAL.md`:
- Table per pairing: within-graphviz floor (median, p95) vs
  dagua-vs-graphviz (median, p95) vs TOST verdict
- Per parked family, the new classification:
  - If equivalent_at_<=2x within-graphviz floor: **CONVERGED -
    stochastic_floor_match**
  - If equivalent_at_<=0.5x: **CONVERGED - well within stochastic_floor**
  - If still not_equivalent: **CONFIRMED architectural divergence**

### Step 5: Update STATE.md classifications

For any family that flips to CONVERGED, update the iteration log and
mark accordingly. Update flail counts only if the classification
actually changes (don't increment for re-evaluation alone).

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_9/SUMMARY.md`.

### Step 7: Tests + commit

```
pytest tests/test_layout/ -x --tb=short -q 2>&1 | tail -20
```

Commit with `feat(fidelity): round 9 -- graphviz seed plumbing fix +
fresh multi-seed re-evaluation`.

The competitor change is potentially impactful -- it changes the
behavior of ALL future graphviz benchmark runs. That's correct (the
old behavior was buggy), but it's a breaking-by-fixing change that
should be clearly noted in the commit message.

## Strategic note

This round answers the question: are dagua's stochastic-family fidelity
gaps real, or measurement artifacts? The Round 8 evidence pointed to
"real" but used a known-broken cache. Round 9 settles it definitively.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All `dagua/layout/ops/**` files (Round 9 is INFRA, not algorithm changes)

**Allowed in Round 9:**
- `dagua/eval/competitors/graphviz_competitor.py` (PRIMARY -- fix the
  `del seed` bug; this counts as a "real bug" per the standing scope rule)
- `scripts/regen_graphviz_seeds.py` (NEW)
- `scripts/algo_fidelity_live_compare.py` (extend to support
  `--graphviz-cache-dir`)
- `eval_output/algo_fidelity/round_9/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- `tests/test_eval/test_graphviz_competitor.py` (NEW or update -- add
  unit test verifying seed propagates to graphviz subprocess)

**Out of scope:**
- Pipeline algorithm changes (those are for Round 10+ if the verdict
  remains not_equivalent)
- Touching `eval_output/benchmark_full/positions/` (preserve history)
</scope_constraints>

<default_follow_through_policy>
This is the round that resolves the measurement-correctness question.
The output is a clean classification per family, not a code-shape
optimization. If the test verdict still says not_equivalent after a
proper multi-seed comparison, you've definitively proven that algorithm
changes are needed -- which is itself a valuable result.

Verify the seed actually reaches the binary (Step 1 verification step
matters). If outputs are still identical across seeds after the fix,
something else is broken and you should write ROUND_9_BLOCKED.md
rather than proceed.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED**: competitor fixed, fresh cache generated, multi-seed
   re-evaluation produces meaningful within-graphviz floor (> 0.001 on
   most graphs), TOST verdicts updated, RE_EVAL written, SUMMARY
   written, STATE updated, commit on develop.
2. **BLOCKED**: ROUND_9_BLOCKED.md if seed propagation can't be made
   to work despite the source-documented mechanism.
</completeness_contract>

<verification_loop>
- After fixing competitor: `pytest tests/test_eval/ -x --tb=short -q -k "graphviz"`
- Smoke test: run graphviz_fdp twice with seed=42 and seed=43, confirm
  outputs differ
- multi_seed_summary.json from re-eval must show within-graphviz floor
  > 0.001 (mean across testable graphs)
- pytest tests/test_layout/ -x --tb=short -q (regression check)
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- After threading seed through, graphviz outputs are still identical
  across different seeds (suggests deeper issue with how the JSON
  output captures positions, or graphviz binary version doesn't
  honor the seed)
- The benchmark cache rewrite would break other tests that depend on
  specific cached values
- statsmodels TOST raises an error on the new data

Write ROUND_9_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF infra works + re-evaluation done.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files. The new round_9/ cache is additive.
- Don't modify benchmark_full/ -- treat it as historical record.
</action_safety>
