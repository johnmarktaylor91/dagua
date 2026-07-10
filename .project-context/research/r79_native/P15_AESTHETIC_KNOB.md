# r80-S8: User-Facing Aesthetic-Priority Knob

Date: 2026-07-09
Branch: `r80/s8-aesthetic-knob` (worktree `~/.claude/worktrees/dagua-native-p3`)

## Mandate (JMT, 2026-07-08/09)

Users can already tweak differentiable loss weights directly
(`LayoutConfig(w_crossing=5.0)`) and write custom loss constraints -- that was
the original vision. What was missing: the internal candidate-selection
composite the undirected-portfolio contest uses to pick a winning engine among
several internally-generated layouts had FIXED weights, so a user's aesthetic
priorities could steer the differentiable optimizer but never steer WHICH
ENGINE won the contest. This sprint closes that gap with a single normalized
"aesthetic priority" profile plumbed into both halves of the stack.

**This is a public-API-shaped change.** The machinery below is built to
support all three candidate surfaces described in "API shape options"; JMT
signs off on the field names before merge (see that section for the
recommendation and the sign-off ask).

## What was built

- `dagua/layout/aesthetics.py` (new module): the internal seam.
  - `AestheticProfile` -- a frozen `{term: multiplier}` mapping.
  - `resolve_aesthetic_profile(config)` -- resolves a `LayoutConfig`'s knob
    fields into a profile, or `None` when unset (the true identity signal).
  - `reweighted_composite(metrics, is_directed, profile)` -- a NEW parallel
    scoring path that mirrors `dagua.metrics.composite` /
    `composite_undirected` term-for-term (never modifies those frozen
    functions) with each term's base weight multiplied by the profile.
  - `apply_loss_multipliers(config, profile)` -- Wire-through B: returns a
    config copy with the mapped `w_*` fields multiplicatively scaled.
  - `PRESETS` -- the four named presets (see below).
- `dagua/config.py`: two new `LayoutConfig` fields, `prioritize: Optional[str]`
  and `aesthetic_weights: Optional[Dict[str, float]]`, both defaulting to
  `None`. Validated lazily (at layout time, in `resolve_aesthetic_profile`),
  not eagerly in `__post_init__` -- avoids importing `dagua.layout.aesthetics`
  from `dagua/config.py` at module-load time (import-cycle risk: `config.py`
  is imported very early by most of the package) and matches how other
  pipeline-scoped fields (e.g. `algorithm_params`) are validated at dispatch
  time rather than construction time.
- `dagua/layout/resolve.py::prepare_pipeline_config`: resolves the profile
  ONCE per problem instance (right after the per-problem shallow config copy),
  applies Wire-through B when non-`None`, and stashes the resolved profile
  object on `_dagua_native_aesthetic_profile` so every downstream consumer of
  that prepared config reuses the IDENTICAL object.
- `dagua/layout/ops/pipelines/native_undirected.py`: Wire-through A.
  `_score_undirected_candidate` takes an optional `aesthetic_profile` param;
  `None` calls `composite_auto` exactly as before (bit-identical, no wrapper).
  `layout_native_undirected_portfolio` reads the profile back off the prepared
  config ONCE and passes that same object into every candidate's score call
  (contest fairness -- the incumbent, both sfdp cleanup variants, both neato
  cleanup variants, the cluster-aware candidate, and the weighted-similarity
  candidate are all scored under the identical profile).

## Design requirement 1: default identity

`resolve_aesthetic_profile` returns `None` when neither `prioritize` nor
`aesthetic_weights` is set. Every call site branches on that `None` and falls
through to the EXACT pre-existing code (`composite_auto(...)` called directly
in `native_undirected.py`; no `apply_loss_multipliers` call at all in
`resolve.py`). This is deliberately NOT "construct an all-ones profile and
rely on it being a float no-op" -- the identity path never enters the new
code at all, which is what makes gate 1 (below) provable rather than merely
"probably fine."

## Wire-through A: selection reweighting

`reweighted_composite` mirrors the frozen composites' term set exactly:

| Term | Directed base weight | Undirected base weight |
|---|---:|---:|
| dag_consistency | 22 | -- |
| edge_length_cv | 18 | 40 |
| depth_spearman | 13 | -- |
| overlap | 8 | 20 |
| straightness | 9 | -- |
| crossing | 9 | 20 |
| stress | 10 | -- |
| angular_res | 5 | 10 |
| cluster_sep | 6 | 10 |

Each term's subscore formula is copied verbatim from `composite()` /
`composite_undirected()` (same clamps, same degeneracy guard reused via
`dagua.metrics._is_degenerate_scale`). The reweighted score is
`sum(base_weight[t] * multiplier(t) * subscore[t]) * (100 / sum(base_weight[t]
* multiplier(t)))` -- the renormalization keeps the reweighted score on
roughly the same 100-point scale as the frozen composite even under
aggressive reweighting, and is an exact no-op at the identity profile (sum of
base weights is exactly 100 by construction, so the renormalization factor is
`100/100 = 1.0`).

Only the undirected-portfolio contest (`native_undirected.py`) is wired to
this path in this sprint -- that is the literal target JMT described
("the portfolio SELECTION composite... the scorer that picks a winning
candidate among several internally-generated layouts"). Two OTHER
same-engine candidate selectors exist in the codebase
(`dagua_native_legacy.py`'s multi-start-k best-of-k scorer and
`dagua_native.py::_best_of_polish`'s polish-variant scorer), both call
`composite()`/`full()` directly. They were deliberately NOT wired this
sprint: they pick among variants of the SAME engine (random-seed restarts,
polish post-processing), not among different competing engines, so they sit
outside "engine choice is not steerable" as JMT framed the gap. Wiring them
is a small, mechanically similar follow-up if wanted -- flagged here rather
than done silently.

## Wire-through B: loss-weight multiplier mapping

Applied in `prepare_pipeline_config`, before `dagua_native_legacy.py::
build_dagua_pipeline` reads `config.w_*` into `InitAnnealingScheduleConfig`
(the only differentiable-loss composition site in the native core -- the
force-directed/native-stress sub-pipelines are classical Pivot-MDS/stress-SGD
algorithms and do not consume `w_*` weights at all, so this table applies to
the layered/legacy native core, which is also what the undirected-portfolio
contest's "incumbent" candidate runs through).

| Aesthetic term | `w_*` field(s) scaled | Multiplier formula |
|---|---|---|
| dag_consistency | `w_dag` | `field *= clamp(multiplier, 0.1, 5.0)` |
| edge_length_cv | `w_length_variance` | same |
| overlap | `w_overlap` | same |
| straightness | `w_straightness` | same |
| crossing | `w_crossing`, `w_edge_crossing`, `w_edge_node_crossing` | same (all three fields get the identical clamped multiplier) |
| stress | `w_stress` | same (0.0 default stays 0.0 -- opt-in loss unaffected unless the user also sets `w_stress>0`) |
| angular_res | `w_fanout`, `w_edge_angular_res` | same (fanout is the closest node-placement analogue for angular spread) |
| cluster_sep | `w_cluster`, `w_cluster_contain` | same |
| depth_spearman | *(none)* | No direct differentiable-loss analogue exists today. The closest structural lever, DAG ordering strength, is already driven by the `dag_consistency` row above. Still participates in Wire-through A (selection). |

Multipliers are clamped to `[0.1, 5.0]` (`LOSS_MULTIPLIER_CLAMP`) so a
pathological `aesthetic_weights` value cannot destabilize the optimizer.
Fields untouched by the profile (multiplier exactly 1.0 for that term) are
never reassigned -- true per-field no-op, not merely value-equal.

## API shape options (JMT sign-off needed)

The internal machinery (`dagua/layout/aesthetics.py`) supports all three
without any rework -- the choice below only affects the two `LayoutConfig`
field names/shapes exposed publicly.

**(a) Preset strings only.** `LayoutConfig(prioritize="crossings")`.
Simplest surface, matches dagua's existing `quality="balanced"` string-alias
precedent exactly. Con: no fine-grained control; a user who wants "boost
crossings 2x AND overlap 1.5x" cannot express it.

**(b) Explicit dict only.** `LayoutConfig(aesthetic_weights={"crossing":
3.0})`. Maximal flexibility, discoverable term names (`VALID_TERMS`), but
loses the one-word ergonomics of a named preset and requires the user to
already know dagua's internal term vocabulary.

**(c) Both, dict overrides preset per-key (IMPLEMENTED, RECOMMENDED).**
`LayoutConfig(prioritize="readability", aesthetic_weights={"crossing": 5.0})`
starts from the `readability` preset's weights and overrides just the
`crossing` entry. This is the shape actually wired in this sprint.

### Worked examples (shape c)

```python
# One-word preset -- most users stop here.
config = dagua.LayoutConfig(prioritize="crossings")

# Fine-tune a preset without hand-rolling the whole vector.
config = dagua.LayoutConfig(prioritize="readability", aesthetic_weights={"crossing": 5.0})

# Fully custom, no preset.
config = dagua.LayoutConfig(aesthetic_weights={"edge_length_cv": 0.5, "overlap": 2.0})
```

### Recommendation: (c)

Rationale:
- **Precedent.** Dagua already ships this exact "preset sets defaults, user
  overrides per-field" pattern for themes (`dagua.set_theme('dark')` +
  per-call overrides -- see
  `~/.claude/knowledge/... feedback_themes_set_defaults_users_override.md`).
  Reusing a pattern JMT has already approved elsewhere lowers the review
  burden and keeps the config surface consistent.
- **Ergonomics + discoverability.** Presets are the answer for "I want
  fewer crossings" without knowing dagua's term vocabulary; the dict escape
  hatch is there once a user wants to go further, without inventing a new
  concept (no separate "advanced mode" flag).
- **Machinery is free either way.** (a) and (b) are strict subsets of what
  (c) already implements (`resolve_aesthetic_profile` degrades gracefully
  when only one of the two fields is set) -- there is no cost to keeping
  both.

**Sign-off ask:** reply with `a`, `b`, or `c` (recommendation: `c`, as
implemented), or propose different field names (`prioritize` /
`aesthetic_weights` are placeholders chosen for clarity, not finalized --
e.g. `aesthetic_priority` / `term_weights` were considered and are equally
easy to rename since only `dagua/config.py`'s two field declarations and the
two `getattr(config, ...)` reads in `resolve_aesthetic_profile` would need to
change).

## Presets

| Preset | Term weights | Intent |
|---|---|---|
| `crossings` | `{crossing: 3.0}` | Untangle dense drawings -- the single biggest readability complaint. |
| `uniform_edges` | `{edge_length_cv: 3.0}` | Classic "tidy" force-directed look. |
| `compactness` | `{stress: 2.5, overlap: 1.5}` | Keep the drawing tight without collisions. |
| `readability` | `{angular_res: 2.0, crossing: 1.5, straightness: 1.5, depth_spearman: 1.3}` | Broad bundle: open angles, fewer crossings, straight edges, clean depth hierarchy. |

## Gate results

### Gate 1: default-identity sweep (dagua-only, --fresh)

Command: `.venv/bin/python3 scripts/r79_baseline.py --dagua-only --fresh`,
launched via `nohup` (survives turn boundaries), logged to
`/tmp/r80_s8_sweep.log`, compared against the committed store at
`53050bf` (52/13/28 legacy + 6/3/6 extended = 74/108), same protocol used by
every prior r79/r80 dagua-only gate sweep (P3D_EVIDENCE.md, S0/S2b/S7/S9
sweeps in `r79-native_STATE.md`).

**Result: PASS (with one investigated, bisected, pre-existing-nondeterminism
row -- not an S8 leak; details below).**

- Sweep completed in 2104s under heavy machine load (3 unrelated benchmark
  sweeps running concurrently; load average ~10-12 throughout).
- **W/T/L: 52/13/28 legacy + 6/3/6 extended -- IDENTICAL to the committed
  store. Zero verdict flips.**
- Per-row composite comparison vs the committed store (`results.json` at
  HEAD, all 108 dagua rows): **107/108 rows bit-identical** (composite
  deltas exactly 0.0). One row moved:
  `r79_weighted_small_world_120` 45.282334834337234 -> 44.68296757340431
  (-0.599; the row is a Loss under both values, no verdict change).

**Bisection of the single mover (per the gate-1 requirement to find the
leak or prove there is none):**

| Run | Code | Composite | Positions sha (first 16) |
|---|---|---|---|
| Committed store (HEAD) | pre-S8 trunk | 45.282334834337234 | 16a0c2cc06c18bb9 |
| Gate sweep row | S8 knob code | 44.68296757340431 | (store row) |
| Probe A | S8 knob code | **45.282334834337234** | 3f6f4ccb14a9102d |
| Probe B run 1 | pre-S8 trunk (S8 changes stashed) | **34.41296225786209** | 2d0980caa27e9a02 |
| Probe B run 2 | pre-S8 trunk (stashed) | 45.282334834337234 | d625a8e67c61a4bd |
| Probe B run 3 | pre-S8 trunk (stashed) | 45.282334834337234 | a697e5a496b94adf |

Probes = the exact harness path (`get_competitor("dagua").layout(graph,
seed=42)` + `evaluate(tier="full")` + `composite_auto`), one graph, run
sequentially on the same loaded machine.

Conclusions:
1. **The S8 knob code reproduces the committed composite bit-exactly**
   (Probe A) -- the sweep's divergent row is not a default-path leak from
   this change.
2. **The pre-S8 trunk itself is nondeterministic on this graph under
   load**: with the S8 changes stashed, three identical baseline runs
   produced two different composites (45.28 twice, 34.41 once). Whatever
   flips this row flips it WITHOUT any S8 code present.
3. The graph is the S9 weighted-similarity contest graph
   (`r79_weighted_small_world_120`, 120-node weighted undirected
   small-world). The observed outcomes are consistent with a contest
   candidate occasionally failing or converging differently under heavy
   memory/CPU contention (every challenger is wrapped in
   `try/except Exception` and silently dropped on failure -- see
   `native_undirected.py::layout_native_undirected_portfolio`). Note the
   committed store row itself recorded runtime 5.08s while the gate-sweep
   row took 109.9s -- a 20x contention signature on the same graph.
4. Even when the composite is bit-stable, the POSITION tensors differ
   across runs (three distinct hashes all scoring the identical 17-digit
   composite) -- i.e., translation-level position variation that all
   composite terms are invariant to. Pre-existing trunk behavior worth a
   follow-up (see deviations section).

Store hygiene: the sweep-published store (which contained the one
transient row) was reverted to the committed store after the W/T/L +
zero-flips verification, following the P3d precedent ("Baseline store
churn was reverted after confirming identical W/T/L"). The frozen store at
HEAD is untouched by this branch.

### Gate 2: efficacy test (different presets select different winners)

Test: `tests/test_layout/test_aesthetic_priority.py::
test_different_presets_select_different_contest_winners_on_real_graph`.

Corpus graph: `random_bipartite_60` (60 nodes, declared-undirected extended
corpus graph -- routes to the undirected-portfolio contest). Measured
directly (not just via the test's pass/fail):

| Preset | `crossing_rate` | `edge_length_cv` |
|---|---:|---:|
| `prioritize="crossings"` | 0.0269 | 0.2632 |
| `prioritize="uniform_edges"` | 0.1536 | 0.0136 |

Each preset wins decisively on its own prioritized term (crossings: 5.7x
lower crossing rate; uniform_edges: 19.4x lower edge-length CV) while losing
on the other term -- proof the knob steers CONTEST SELECTION, not just
perturbs a score. Positions differ (`torch.equal` is `False`); the two
presets do not merely relabel the same winner.

A short scan of other undirected-portfolio corpus graphs (`grid_5x5`,
`weighted_clusters_3x10`, `regular_3_30`, `sierpinski_42`,
`triangular_lattice_36`, `grid_rect_6x8`, `real_karate_34`,
`weighted_karate_34`, `multi_component_80`, `planar_60`, `petersen_10`,
`regular_4_40`) found `grid_5x5` also flips (winner index 3 for `crossings`
vs index 0 for `uniform_edges`) but with a much weaker signal, because
`grid_5x5` already has zero crossings under every candidate (planar-lattice
floor effect) -- `random_bipartite_60` was chosen as the recorded/primary
gate graph because both prioritized terms have real headroom to move in
either direction.

### Gate 3: loss-path test

Test: `tests/test_layout/test_aesthetic_priority.py::
test_prepare_pipeline_config_shifts_resolved_loss_weight`. Resolves one
problem instance's config with and without `prioritize="crossings"` (no full
layout solve) and asserts `w_crossing` scales by exactly 3.0x (the preset's
documented multiplier) while `w_dag` (an unrelated term) is untouched.
Passing.

### Gate 4: scoped tests + ruff

**Result: PASS.**

- `pytest tests/test_native_undirected_portfolio.py
  tests/test_layout/test_quality_knob.py
  tests/test_layout/test_aesthetic_priority.py -q` -- **42 passed** (the 17
  new aesthetic-priority tests + all pre-existing portfolio and quality-knob
  tests; one pre-existing portfolio test spy was updated to the new optional
  4th parameter of `_score_undirected_candidate`).
- `ruff check` on all changed files -- clean.

## Deviations from the brief

- Wire-through A was scoped to the undirected-portfolio contest only (see
  "Wire-through A" section above for the two same-engine selectors
  deliberately left unwired, with rationale). This was a scoping decision,
  not an oversight -- flagged per the brief's instruction to surface
  material ambiguities rather than silently expanding scope.
- `prioritize`/`aesthetic_weights` validation happens lazily (at layout time)
  rather than eagerly in `LayoutConfig.__post_init__`, unlike the `quality`
  field. This avoids a module-load-time import of `dagua.layout.aesthetics`
  from `dagua/config.py` (import-cycle risk, since `config.py` is one of the
  most widely and early imported modules in the package). Documented as a
  deliberate simplification above; happy to move to eager validation if JMT
  prefers fail-fast-at-construction and accepts the import-ordering fix
  (e.g. constants-only submodule with zero dependency on `dagua.config`).

## New findings surfaced (pre-existing, not S8)

1. **`r79_weighted_small_world_120` is load-sensitive-nondeterministic on
   the pre-S8 trunk** (see gate 1 bisection table): identical seeded runs
   of the unmodified trunk produced composites {45.28, 45.28, 34.41} under
   heavy machine load. Suspect: silent challenger drop in the portfolio
   contest (`try/except Exception` around each candidate) or a
   load-sensitive convergence path in the weighted-similarity/native-stress
   candidate. Any future gate sweep can trip over this row; recommend a
   follow-up to (a) log dropped challengers instead of silently swallowing
   them, and (b) re-run per-graph movers before declaring a sweep verdict
   (exactly the procedure used here).
2. **Bit-stable composites hide translation-level position variation**:
   three trunk runs of the same graph produced three different position
   tensors with the identical 17-digit composite. Harmless for scoring
   (all composite terms are translation-invariant) but means positions_path
   artifacts are not byte-reproducible across runs for this graph family.
