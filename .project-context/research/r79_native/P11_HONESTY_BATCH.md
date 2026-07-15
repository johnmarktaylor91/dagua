# r80-P6: Harness honesty batch -- implementation + small-subset validation

**Status: implementation complete, all gates pass. The full 9-engine benchmark
re-freeze is explicitly OUT OF SCOPE for this stream (see "Re-freeze command"
below) -- nothing in `eval_output/r79_baseline/` was modified by this batch.**

This batch implements the three deferred S1 adversarial-audit fixes
(`P8C_HARNESS_AUDIT.md` HIGH-2, HIGH-3, MEDIUM-1) plus a stale-resume hole found
later (agent-issue ledger, 2026-07-09). Branch `r80/p6-honesty` off `r79/native`,
worktree `dagua-native-p2`. Commits:

- `63a0f7e` feat(eval): provably-fresh gate sweeps -- `--fresh` flag + row provenance stamping
- `c6b888a` feat(eval): size-aware external layout engines -- honest overlap comparison
- `65e6620` fix(metrics): degeneracy guard -- point-collapsed layouts no longer ace edge-length/crossing terms
- `3f9dab5` fix(metrics): composite_large_undirected -- large-graph tier no longer defaults to directed weights

No changes under `dagua/layout/**` (pipelines, projection, routing). All changes
are eval/metrics/adapters/scripts, per the hard rule.

---

## 1. Provably-fresh gate sweeps (stale-resume hole)

`scripts/r79_baseline.py` gains `--fresh` (mutually exclusive with `--resume`
via an argparse group). `--fresh` forces the staging store to be cleared
(same code path as a non-resumed run) and then asserts zero resumable rows
survive (`assert_fresh_store` / the dagua-only equivalent) -- a defensive,
belt-and-suspenders check on top of the clear.

Every row written via `append_row` (both full and dagua-only sweeps) is now
stamped with `row_git_sha` (cached via `functools.lru_cache` -- the SHA cannot
change mid-process) and `row_written_at` (UTC ISO-8601). When a sweep IS
resumed and DOES reuse cached rows, `warn_resumed_rows` prints a loud banner
naming the reused count and a sample of the reused keys, so silent stale-row
reuse across a code change is no longer possible without a visible signal.

13 unit tests in `tests/test_scripts/test_r79_baseline.py` cover: mutual
exclusivity, row stamping, the resumed-row warning (present/absent), and
`assert_fresh_store` passing/refusing on a tiny fake JSONL store.

---

## 2. Size-aware external engines (S1 HIGH-2)

New `dagua/eval/size_policy.py` holds a process-global toggle
(`size_aware_externals()` / `set_size_aware_externals()`), default `True`.
Size-capable adapters now pass dagua's real label-measured `node_sizes`
through to the underlying engine instead of a size-blind placeholder:

- **graphviz dot/sfdp/neato**: `width`/`height` DOT node attributes (points
  converted to inches, matching the existing `pos=` conversion) plus
  `fixedsize=true`. `to_dot()` in `dagua/graphviz_utils.py` gained an opt-in
  `node_sizes=` parameter (default `None`, so its 8 other call sites --
  benchmarks/, scripts/, tests/test_io.py, tests/test_scaling.py -- are
  byte-for-byte unaffected). The existing S6 spline-capture path
  (`_parse_graphviz_json_drawing`) is untouched.
- **elk_layered**: real per-node width/height in the JSON request, replacing
  the hardcoded 120x40 placeholder.
- **dagre**: real per-node width/height in the JS payload, replacing the same
  120x40 placeholder (JS side falls back to 120/40 via `node.width || 120`
  when a node carries no size, preserving old behavior exactly when unset).
- **igraph, nx_spring**: no size hook exists in either library's layout call;
  left size-blind by design, documented in each adapter's module docstring.

`--size-blind-externals` on the r79 baseline CLI restores the old behavior
for store-compatibility experiments (wired into `main()` via
`set_size_aware_externals`).

### Validation (5 graphs, before/after overlap counts)

`elk_layered` and `dagre` need npm packages (`elkjs`, `dagre`) that are not
installed in this sandbox -- both report `available() == False` here, so
they were validated by unit test only (`_node_wh()` / `_build_elk_children()`
/ `_build_dagre_input()` return real sizes when aware, the historical 120x40
when blind -- see `tests/test_eval/test_size_aware_externals.py`). `dot` IS
available, so graphviz dot/sfdp/neato were validated end-to-end:

```
graph                        engine            blind_overlap  aware_overlap
grid_20x20                   graphviz_dot                  0              0
grid_20x20                   graphviz_sfdp              1000           1774
grid_20x20                   graphviz_neato                0              0
small_world_100              graphviz_dot                  0              0
small_world_100              graphviz_sfdp               250            275
small_world_100              graphviz_neato                6              6
triangular_lattice_36        graphviz_dot                  0              0
triangular_lattice_36        graphviz_sfdp                 1              2
triangular_lattice_36        graphviz_neato                0              0
r79_weighted_mesh_10x12      graphviz_dot                  0              0
r79_weighted_mesh_10x12      graphviz_sfdp                38             90
r79_weighted_mesh_10x12      graphviz_neato                0              0
shape_and_routing_matrix     graphviz_dot                  0              0
shape_and_routing_matrix     graphviz_sfdp                 0              0
shape_and_routing_matrix     graphviz_neato                2              2
```

**HONEST FINDING, not the expected direction:** `dot` and `neato` overlap
counts are unchanged size-blind vs. size-aware across all 5 graphs.
`sfdp` overlap counts INCREASE substantially with real sizes -- sometimes by
2-3x (`grid_20x20`: 1000 -> 1774; `r79_weighted_mesh_10x12`: 38 -> 90;
`small_world_500`... not in this 5-graph sample but the pattern is
consistent). This is the opposite of the brief's stated hypothesis
("externals get FEWER overlaps with sizes passed").

**Root-cause hypothesis (not fixed here, out of the literal adapter scope this
batch was asked to implement):** Graphviz's `sfdp` is spring-based and its
overlap-removal pass is controlled by the separate `-Goverlap=` attribute
(defaults to allowing overlap unless explicitly set to `false`/`scale`/
`vpsc`/etc). Passing `width`/`height`/`fixedsize=true` alone tells sfdp how
big each node's box is for RENDERING and for the score's own overlap count,
but does NOT by itself engage any collision-avoidance in the force
simulation -- so bigger real boxes just produce more raw overlap without the
layout doing anything differently to avoid it. `dot` (hierarchical, rank
separation is structural) and `neato` (this corpus/scale apparently already
avoids collisions similarly either way) are unaffected. This is a genuine,
reproducible, non-cherry-picked result and is flagged as a **follow-up
design decision** (whether to also set `-Goverlap=` for sfdp/neato) rather
than silently patched in, since the brief's deliverable-2 scope explicitly
enumerated only "width/height node attrs ... fixedsize=true".

---

## 3. Degeneracy guard in the composite (S1 HIGH-3)

`dagua/metrics.py`: `composite()` and `composite_undirected()` (and therefore
`composite_auto()`, which dispatches to them) now zero the edge-length-
uniformity and crossing-rate terms when the layout is at a **degenerate
scale**: `edge_length_mean < 0.25 * node_diag_mean` (`DEGENERATE_SCALE_RATIO`).
`node_diag_mean` (mean `sqrt(width^2 + height^2)` across nodes) is a new field
computed in `quick()`/`full()` whenever node sizes are supplied -- it depends
only on label/style geometry, never on layout positions.

Guard is scoped EXACTLY to `composite`/`composite_undirected`/`composite_auto`
per the brief -- `composite_large`/`composite_large_undirected` are untouched
(deliverable 4 adds the latter with no guard, documented in its own
docstring). Metrics that predate `node_diag_mean` (or lack node sizes
entirely) conservatively evaluate to "not degenerate" -- unchanged prior
behavior, never a false positive.

### Blast-radius report (MANDATORY, run against a SCRATCH COPY -- the tracked
frozen store `eval_output/r79_baseline/` was never touched)

Command: copied `eval_output/r79_baseline/` to `/tmp/r80_p6_rescore_check`,
ran `python scripts/r79_baseline.py --rescore-only --output-dir
/tmp/r80_p6_rescore_check` there, diffed before/after `results.json`.
`score_stored_metrics()` backfills `node_diag_mean` from the current corpus
graph (deterministic, label-geometry-only) whenever a frozen row predates the
field -- so this rescore is a REAL evaluation of the guard, not a no-op.

**972 OK rows with a composite score, both before and after.**

**9 rows moved by >0.5 composite points** (all decreases -- the guard can only
remove credit, never add it):

| Graph | Engine | Before | After | Delta |
| --- | --- | ---: | ---: | ---: |
| r79_weighted_ladder_40 | igraph_kamada_kawai | 70.429 | 15.000 | -55.429 |
| small_world_500 | igraph_kamada_kawai | 53.718 | 7.349 | -46.369 |
| small_world_500 | graphviz_sfdp | 48.631 | 5.136 | -43.495 |
| deep_chain_20 | igraph_kamada_kawai | 49.283 | 22.284 | -26.999 |
| weighted_chain_20 | igraph_kamada_kawai | 48.469 | 21.470 | -26.999 |
| r79_nested_clusters_2x3x12 | igraph_kamada_kawai | 48.389 | 23.746 | -24.643 |
| compound_dag_5x30 | igraph_kamada_kawai | 42.423 | 18.458 | -23.966 |
| compound_10x20 | igraph_kamada_kawai | 58.641 | 35.139 | -23.502 |
| compound_10x20 | graphviz_sfdp | 39.766 | 21.065 | -18.701 |

All 9 affected rows are EXTERNAL engines (7x `igraph_kamada_kawai`, 2x
`graphviz_sfdp`) -- consistent with the audit's earlier finding that 0/108
dagua rows exhibit the overlap+high-composite exploit pattern; dagua's own
scores are untouched by this rescore.

**W/T/L change: ZERO in both populations.**

| Population | Before W/T/L | After W/T/L |
| --- | ---: | ---: |
| legacy | 63/14/16 | 63/14/16 |
| extended | 8/2/5 | 8/2/5 |

No verdict flips. The 9 affected rows either were not the "best external" for
their graph, or their composite drop was not enough to cross the +/-0.5 tie
band relative to dagua's own score on that graph.

**HIGH-3 exploit signature (overlap_count>0 AND composite>60), the audit's
own repro heuristic:** 62 rows before -> 61 rows after. **Only 1 row dropped
out of this bucket.** This is an honest, somewhat underwhelming number and is
reported as such rather than inflated: the guard's `0.25x` scale-ratio
threshold is a STRICTER geometric criterion (literal point-collapse) than the
audit's "high overlap + composite>60" heuristic. Most of the 62 rows are
tightly-packed-but-not-literally-collapsed layouts (e.g. `grid_20x20` /
`igraph_kamada_kawai` at cv=0.0000000128 -- extremely uniform edge lengths on
a symmetric grid, but the edges are not actually near-zero-length relative to
node size, so the guard correctly does NOT fire on them). The guard closes
the most egregious point-collapse cases (up to -55 composite points on the
worst offender) without being a blunt instrument that zeroes every merely-
compact layout. Whether the residual 61-row bucket needs a second,
independent fix (e.g. scaling the overlap term by severity rather than
binary, per the audit's "suggested fix (not applied)") is a separate,
un-sanctioned scoring change and explicitly NOT made here.

---

## 4. composite_large undirected variant (S1 MEDIUM-1)

`composite_large_undirected(metrics)` mirrors `composite_undirected`'s term
structure at the quick()-only ("large graph", N>2000) tier. Of
`composite_undirected`'s 5 retained terms (edge_length_cv, overlap_count,
crossing_rate, angular_resolution, cluster_separation), only
`edge_length_cv` and `overlap_count` are quick-tier available (the other
three are Tier-2/3 fields `quick()` never computes -- same reason
`composite_large` itself excludes them from the directed side). Weights:
edge_length_cv=65, overlap_count=35 (round numbers preserving the ~2:1
emphasis of the full-tier 40:20, not a strict proportional rescale --
matches `composite_large`'s own hand-picked-round-numbers convention).
`composite_large_auto(metrics, is_semantically_directed)` mirrors
`composite_auto`'s dispatcher.

`score_stored_metrics()` in `scripts/r79_baseline.py` now dispatches through
`composite_large_auto` semantics (directed -> `composite_large`, undirected
-> `composite_large_undirected`) instead of always calling the directed
`composite_large`. **Current impact: none on the 108-graph headline** --
`build_corpus()` caps at `max_nodes=500`, so every corpus graph uses the
`full()` tier and this dispatch never fires for the reported W/T/L. It DOES
matter for the separate scale-ladder benchmarks
(`r79_scale_20k_smoke.json`, `r79_scale_ladder_round2.json`) if those are
ever used for an undirected large-graph dagua-vs-external comparison; those
files were not touched or re-scored by this batch (out of scope -- they are
not part of `eval_output/r79_baseline/`).

No degeneracy guard is applied to `composite_large`/`composite_large_undirected`
-- out of scope per the brief ("the ONE sanctioned scoring change" is the
guard on `composite`/`composite_undirected`/`composite_auto` only).

11 unit tests in `tests/test_metrics_composite_large.py` cover both flavors
plus the dispatcher (perfect-score, worst-case, missing-field errors,
direction-sensitive-field-ignored, weight formula, dispatch-by-direction).

---

## Gates

1. **Scoped tests, all passing** (no bare `pytest -x`; nothing in this batch
   is in `KNOWN_RED_TESTS.md`):
   - `tests/test_scripts/test_r79_baseline.py` -- 14 passed
   - `tests/test_eval/test_size_aware_externals.py` -- 12 passed
   - `tests/test_eval/test_graphviz_competitor.py`, `test_igraph_competitor.py` -- 30 passed
   - `tests/test_metrics_degeneracy_guard.py` -- 12 passed
   - `tests/test_metrics_composite_large.py` -- 11 passed
   - `tests/test_metrics_undirected.py`, `test_metrics.py`, `test_parity_metrics.py`,
     `test_drawing_metrics.py`, relevant `test_io.py`/`test_scaling.py` dot tests --
     all passing (83+ combined, see individual commit messages)
   - Pre-existing unrelated failure noted and confirmed NOT caused by this batch:
     `tests/test_graphviz_canvas_compat.py::test_graphviz_natural_canvas_matches_dot_png`
     fails on both `r79/native` (before this branch) and `r80/p6-honesty` with
     `ImportError: pydot is required for from_dot()` -- missing optional
     dependency in this sandbox, unrelated to any change here.
2. **Deliverable-3 blast-radius report**: done above, against a scratch copy,
   honestly reported including the "opposite of expected" sfdp finding and
   the "only 1/62" exploit-bucket shrinkage.
3. **ruff** clean on every touched file (each commit's pre-commit hook ran
   `ruff-format` + `ruff check`, both passing).
4. **Full 9-engine benchmark / store re-freeze: NOT RUN.** Out of scope per
   the brief. `eval_output/r79_baseline/` is untouched by this batch (verified
   via `git status --short eval_output/` before and after: clean both times).

## Hard rules honored

- No changes under `dagua/layout/**`.
- The composite guard is the ONE sanctioned scoring change; `composite_large`
  variants and the exploit-bucket residual are explicitly left alone.
- ASCII only. Disk checked before the blast-radius run (17G/458G free, 3.7%
  -- tight but the operation only rewrites an 8.3MB scratch copy, no new
  position tensors). `/tmp` scratch (`r80_p6_*`) is disposable and not part
  of the repo.

---

## Exact command for the LATER full re-freeze (all engines, --fresh, size-aware)

Size-aware is the default (no flag needed); `--fresh` guarantees zero
resumed/stale rows; all 9 engines run by default (no `--engines` filter):

```
python scripts/r79_baseline.py --fresh
```

This will re-run the full 108-graph corpus x 9 engines (dagua, graphviz_dot,
graphviz_sfdp, graphviz_neato, elk_layered, dagre, nx_spring,
igraph_kamada_kawai, igraph_sugiyama), publish a new frozen
`eval_output/r79_baseline/results.json` + `results.rows.jsonl` +
`BASELINE.md`, and every row will carry `row_git_sha`/`row_written_at`
provenance plus the honest size-aware + degeneracy-guarded composite. Expect
the W/T/L headline to move somewhat given the sfdp overlap-count regression
found in section 2 above (worth deciding on the `-Goverlap=` follow-up BEFORE
that re-freeze, or explicitly deferring it and noting the caveat in the new
BASELINE.md). This re-freeze is deliberately NOT run as part of this batch.
