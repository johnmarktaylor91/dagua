<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`. ONE working branch.

Round 10 of the algo_fidelity sprint. Read these in order:
1. `.project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md`
2. `eval_output/algo_fidelity/round_9/SUMMARY.md` (the BIG win)
3. `.project-context/research/sprint_algo_fidelity/ROUND_9_RE_EVAL.md`

## Round 10 context

ALL 4 graphviz families are now CONVERGED via Round 9 (TOST equivalence
under proper multi-seed graphviz cache). The user's primary directive
("perfect graphviz") is achieved.

Round 10 applies the same multi-seed + TOST lens to the **less-important
families** that the original mega-run flagged as `partial_match` or
`divergent` in `eval_output/fidelity_report/report.md`. The hypothesis:
some of these may also be measurement artifacts that re-classify to
equivalent under proper stochastic-floor testing -- like fdp did.

## Less-important families to evaluate

Per `eval_output/fidelity_report/report.md`, these have non-converged
verdicts in the existing mega-run:

| Family | Original | Verdict | RMSD median | Variants |
|---|---|---|---:|---:|
| davidson_harel | igraph_davidson_harel | divergent | 0.34-0.36 | 3 |
| drl | igraph_drl | partial_match | 0.13-0.20 | 5 |
| graphopt | igraph_graphopt | partial_match | 0.10-0.16 | 6 |
| neulay | neulay | partial_match | 0.16-0.20 | 6 |
| tsnet | tsne_graph | partial_match | 0.15-0.27 | 4 |
| fa2 | fa2_ref | weak/partial | 0.05-0.18 | 11 |
| sgd2_multi | sgd2_multi_ref | weak | 0.11-0.22 | 8 |
| stress_sgd | sgd2 | weak | 0.04-0.05 | 4 |

The same seed-not-passing bug we fixed for graphviz may exist for
**igraph/ogdf/etc. competitor adapters**. Investigate.

## What to do

### Step 1: Audit non-graphviz competitor adapters for seed plumbing (15 min)

Read these files:
- `dagua/eval/competitors/igraph_competitor.py`
- `dagua/eval/competitors/ogdf_competitor.py`
- `dagua/eval/competitors/sgd2_competitor.py`
- `dagua/eval/competitors/fa2_competitor.py`
- `dagua/eval/competitors/cytoscape_competitor.py`
- `dagua/eval/competitors/gephi_competitor.py`
- Any others under `dagua/eval/competitors/`

For each, check whether the `seed` parameter actually propagates to the
underlying library call. Document findings in
`.project-context/research/sprint_algo_fidelity/ROUND_10_AUDIT.md`.

If you find another `del seed` or equivalent silent-drop pattern, fix
it the same way Round 9 fixed the graphviz one.

### Step 2: Run multi-seed comparator on Phase 2 families (30 min)

Build a quick wrapper that runs the multi-seed comparator across the
6-7 most impactful Phase 2 pairings. Suggested target families (skip
ones where mega-run says `strong_equivalent` already):

- classic_davidson_harel vs igraph_davidson_harel
- classic_drl vs igraph_drl
- classic_graphopt vs igraph_graphopt
- classic_neulay vs neulay
- classic_tsnet vs tsne_graph
- classic_fa2 vs fa2_ref
- classic_sgd2_multi vs sgd2_multi_ref

For each, compute:
- within-target floor (median, p95) -- using cached or regenerated seeds
- dagua-vs-target (median, p95)
- Aggregate TOST verdict at margins {0.5x, 1x, 1.5x, 2x} of the floor
- Per-graph counts (equivalent / not_equivalent)

Use 5-9 seeds per family. If a competitor adapter doesn't propagate
seed (per Step 1), regenerate that family's cache with the fix
applied (small cache scope, like Round 9 did).

If for ANY family the cache regeneration is impractically slow (e.g.,
>10 min per graph), document and skip -- single-seed comparison is
the fallback.

### Step 3: Per-family classification update

For each Phase 2 family, classify per the same scheme as Round 9:

- `equivalent_at_<=1x` -> reclassify CONVERGED with `stochastic_floor_match`
- `equivalent_at_2x` -> reclassify weak_equivalent (was partial_match)
- `not_equivalent` AND verdict was partial -> classification stands
- `not_equivalent` AND verdict was divergent -> investigate the
  worst graph; is it a known structural failure?

### Step 4: ONE focused fix attempt on the worst remaining family

If after Step 3 there's still a family classified as divergent or
strong-not_equivalent (with within-floor ~0 like davidson_harel may
be), pick that family and try ONE focused single-lever fix per the
established playbook. Same scope rules as previous rounds:
- < ~80 lines net
- Smallest plausible lever
- High-confidence only -- if no high-confidence lever, document and
  move on

### Step 5: Write report

`.project-context/research/sprint_algo_fidelity/ROUND_10_PHASE2_REPORT.md`:
- Audit findings (any seed-bugs found in non-graphviz competitors)
- TOST verdict table per family
- Classification updates (which families flipped, which stayed)
- Any focused fix attempted in Step 4 + outcome

### Step 6: Per-round summary

`eval_output/algo_fidelity/round_10/SUMMARY.md` summarizing the Phase 2
verdicts and the final state of all sprint-relevant families.

### Step 7: Update STATE.md

Append iteration log row. Update tasks. If all families now have
some classification (CONVERGED or principled_residual), set
`current_round: 11` and `state: SUMMARY_READY`.

## Tests + commit

```
pytest tests/test_layout/ -x --tb=short -q 2>&1 | tail -20
```

Commit with `feat(fidelity): round 10 -- phase 2 multi-seed sweep +
classifications`.
</task>

<scope_constraints>
**HARD scope -- DO NOT TOUCH:**
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- All `dagua/layout/ops/**` files EXCEPT for the optional Step 4
  focused fix on the worst remaining family. If you do attempt a fix
  in Step 4, the relevant ops file (e.g., `dagua/layout/ops/davidson_harel.py`
  if that's the family chosen) is allowed.

**Allowed in Round 10:**
- `dagua/eval/competitors/{igraph,ogdf,sgd2,fa2,cytoscape,gephi,neulay,tsnet}_competitor.py`
  (per-adapter seed plumbing fixes if found)
- `scripts/algo_fidelity_live_compare.py` (if minor extensions needed)
- `scripts/regen_*_seeds.py` (NEW per family if needed)
- `eval_output/algo_fidelity/round_10/**` (new)
- `.project-context/research/sprint_algo_fidelity/**`
- ONE pipeline file in `dagua/layout/ops/**` IF Step 4 attempts a focused fix

**Out of scope:**
- Multiple pipeline changes (one max, only if Step 4 finds a strong lever)
- run_benchmark execution
- Touching benchmark_full/positions/ (preserve history)
</scope_constraints>

<default_follow_through_policy>
Round 10 is primarily a measurement + classification round. The big
work was already done (graphviz parity validated). Don't grind on
algorithm fixes here -- the user said "improve the less-important
ones" not "perfect them".

If a Phase 2 family reclassifies cleanly to CONVERGED via multi-seed,
that's a result. If a family stays divergent, that's a residual to
document. One focused lever attempt MAX if there's a clear
high-confidence target.
</default_follow_through_policy>

<completeness_contract>
1. **COMMITTED**: audit done, multi-seed sweep done, classifications
   recorded, RE_EVAL/REPORT written, SUMMARY written, STATE updated,
   commit on develop.
2. **BLOCKED**: ROUND_10_BLOCKED.md if a hard infra issue prevents
   the sweep (e.g., missing competitor implementations).

Round 10 doesn't NEED a code fix to count as committed -- if the
audit + sweep + classifications are valuable on their own, commit
the docs/data.
</completeness_contract>

<verification_loop>
- pytest tests/test_layout/ -x --tb=short -q (regression)
- multi_seed_summary.json present for at least 4 of the 7 target
  Phase 2 families
- ROUND_10_PHASE2_REPORT.md has classifications for each tested family
- `git diff --stat HEAD~0` before commit shows only allowed scope
</verification_loop>

<missing_context_gating>
ABORT if:
- Multiple competitor adapters need NEW seed-passing implementations
  that aren't 1-line fixes (would be too much scope for one round)
- Cache regeneration on any family takes >5 min per graph (skip that
  family with documentation)

Write ROUND_10_BLOCKED.md and stop.
</missing_context_gating>

<action_safety>
- ONE commit on develop only IF audit + sweep complete.
- No force-push, branch creation, rebase, or tag.
- Never delete eval_output files. Round 10 cache is additive.
</action_safety>
