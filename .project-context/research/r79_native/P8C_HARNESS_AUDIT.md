# r80-S1: Adversarial audit of the r79 native-algo eval harness

**Verdict: SOUND-WITH-CAVEATS**

The headline W/T/L numbers (56/8/29 legacy, 8/2/5 extended) are honestly computed,
internally consistent, and reproduce exactly under independent recount. Frozen-store
integrity, tie-band symmetry, and the actual metric-determinism path used by the
harness are all clean. However, three real structural issues were found and proven
with concrete repro evidence: (1) the sprint's own "no re-layout, pure rescore"
claim for the P3a oracle fix is factually false -- dagua's positions were silently
regenerated with newer algorithm code in the same commit, conflating two effects
in the reported 87/12/9 -> 56/8/29 swing; (2) every external engine is laid out
completely blind to the node-size convention it is then scored against, which
structurally advantages dagua on the "no overlaps" composite term; (3) the
composite formula has a genuine degenerate-layout exploit (collapsing a graph to
a point scores higher than a normal random layout) that is already manifesting in
62/972 real production rows. None of these three currently appear to flip a
recorded dagua-vs-external verdict (spot-checked), but they are real, proven,
non-hypothetical defects that erode confidence in the composite as an absolute
quality signal and in the "old number was pure fiction" narrative.

---

## Findings (ordered by severity)

### HIGH-1: "P3a rescored, no re-layout" is false -- dagua was silently re-run with newer code in the same commit

**Files:** `.project-context/research/r79_native/r79-native_SUMMARY.md:36` (claim);
`eval_output/r79_baseline/positions/*__dagua.pt` (evidence); commit `2c19c45`.

**Claim under test:** SUMMARY.md's merge table says of P3a: *"ORACLE FIX:
is_semantically_directed() -- 39 graphs retagged undirected, frozen store
rescored (no re-layout)"* -- i.e. the 87/12/9 -> 56/8/29 W/T/L swing should be
attributable ENTIRELY to the oracle/tag fix, with dagua's actual layout positions
held constant.

**Repro (ran, not hypothesized):**
```
git cat-file -p 3b706c5:eval_output/r79_baseline/positions/er_500__dagua.pt > /tmp/before.pt
git cat-file -p 2c19c45:eval_output/r79_baseline/positions/er_500__dagua.pt > /tmp/after.pt
python -c "
import torch
a = torch.load('/tmp/before.pt'); b = torch.load('/tmp/after.pt')
print(torch.equal(a,b), (a-b).abs().max().item())
"
# -> False, 25477.10546875   (position range for this graph is roughly [-85000, 68000])
```
`git log --follow` on `positions/er_500__dagua.pt` shows exactly two commits ever
touched it: `c80e970` (initial freeze) and `2c19c45` (the "semantics fix" commit).
Every intervening commit on the layout side -- `11beeec` (P1 registry fix, claimed
"bit-identical"), `e1a8344` (P2c layered-DAG polish, a genuine algorithm change) --
did NOT touch the position file, meaning `2c19c45` was the FIRST re-run of dagua
since the freeze, and it silently picked up all the accumulated algorithm code
changes. This is not floating-point noise: I confirmed today's dagua layout is
bit-exact deterministic (two live runs of `er_500` gave `torch.equal() == True`,
and a live run today matches the *currently stored* position exactly, max abs
diff 0.0 -- see "checked and clean" below). So the diff is a real, deterministic
consequence of code that changed between the freeze and the rescore commit, not
noise.

Also: the same commit's diff to `dagua/eval/graphs.py` shows `is_semantically_directed`
was moved into that file and ~15 graph constructors gained an `"undirected"` tag
(pure metadata) -- no `graph=` edge construction changed. So the position delta
cannot be attributed to the tag change itself; it is attributable to accumulated
layout-code changes that happened to get exercised for the first time in the same
commit as the tag fix.

**Expected vs actual:** Expected: dagua positions frozen at `c80e970`-era values
through the P3a rescore, isolating the oracle fix's contribution to the W/T/L
swing. Actual: dagua positions changed substantially and simultaneously with the
oracle fix, so the reported 87/12/9 -> 56/8/29 delta is a MIX of (a) the honest
oracle correction and (b) an unquantified contribution from whatever native-algo
changes had landed since the freeze (P1 registry fix, P2c layered polish). The
SUMMARY's framing ("the old number was partly fiction") is directionally correct
about the oracle bug but overstates isolation -- some of the 31-point legacy swing
is not oracle-attributable at all.

**Corroborating evidence -- metadata has already drifted from itself:** The stored
`results.json["metadata"]["semantics_fix"]` field records the W/T/L snapshot taken
*at the moment of the 2c19c45 commit*: `before_wtl: {legacy:[76,9,8], extended:
[10,2,3]}`, `after_wtl: {legacy:[55,8,30], extended:[8,1,6]}`. My independent
recount of the CURRENT store (below) gives legacy `56/8/29`, extended `8/2/5` --
one graph flipped LOSS->WIN in legacy and one flipped LOSS->TIE in extended
*since* the semantics-fix snapshot was taken, from subsequent `--dagua-only`
reruns (P2c-fix, P3d, etc.) that were never reflected back into the `semantics_fix`
metadata field. This is a second, independently-detectable symptom of the same
root cause: the harness has no mechanism that keeps the "before/after oracle fix"
narrative in sync with ongoing `--dagua-only` reruns, so the causal story printed
in docs and embedded in results.json metadata silently goes stale within days.

**Severity rationale:** HIGH. This does not corrupt the CURRENT headline numbers
(those are internally consistent, see clean-checks below) but it means the
sprint's own explanation of *why* the numbers are what they are is not accurate,
and the pattern (bundling an oracle/scoring fix with an unrelated algorithm rerun
in one commit, with no automated separation) is a process risk that will recur.

**Suggested fix (not applied -- report only):** When rerunning `--dagua-only`
specifically to pick up an oracle/tag fix, do it as a *pure* rescore
(`--rescore-only`, which only recomputes composites from already-stored metrics
and never touches position tensors) as a separate, isolated step from any
algorithm-code-driven `--dagua-only` rerun, and record both deltas separately in
metadata.

---

### HIGH-2: External engines are laid out size-blind but scored size-aware; the mismatch is quantified and goes in both directions

**Files:** `dagua/eval/competitors/graphviz_competitor.py:114-151` (`_node_statement`,
no width/height emitted at all), `dagua/eval/competitors/dagre_competitor.py:32`
(hardcoded `{width:120, height:40}` for every node), `dagua/eval/competitors/elk_competitor.py:122,137`
(same hardcoded 120x40), `dagua/eval/competitors/igraph_competitor.py` and
`networkx_competitor.py` (no size handling found by grep at all), vs.
`dagua/metrics.py:1970-1975` (`evaluate()` always scores using `graph.node_sizes`,
dagua's own per-node label-measured box size, for EVERY engine's positions).

**Repro (ran):**
```python
# /tmp/r80_size_fairness_check.py
graph = graphs["shape_and_routing_matrix"].graph
graph.compute_node_sizes()
dot_src = _graph_to_dot(graph)          # what graphviz actually receives
data = json.loads(subprocess.run(["dot","-Tjson", ...]).stdout)  # what graphviz assumed
# compare graph.node_sizes (dagua's convention, used for scoring ALL engines)
# against the width/height graphviz reports it actually used
```
Output (6-node graph, all values in points):
```
idx  dagua_w  dagua_h  gv_w(pt)  gv_h(pt)  w_ratio  h_ratio
  0    44.00    34.00     54.00    36.00    0.815    0.944
  1    69.52    34.00    105.88    36.00    0.657    0.944
  2   129.19    99.38    161.93    36.00    0.798    2.761
  3    77.45    34.00    119.43    36.00    0.649    0.944
  4    48.26    34.00     61.56    36.00    0.784    0.944
  5    62.01    62.01     91.11    91.11    0.681    0.681
```
Graphviz's own auto-fit boxes are consistently 20-50% WIDER than dagua's
node_sizes (width ratio 0.65-0.82) -- meaning graphviz spaces its layout for
bigger boxes than it is scored against, understating its true overlap risk under
dagua's convention (unfair advantage to graphviz). For node 2, graphviz's box is
36pt tall vs dagua's 99.38pt estimate (h_ratio 2.76) -- graphviz packed that
multi-line-label node into a box less than half the height dagua thinks it needs,
which would inflate that node's overlap risk under dagua's scoring (unfair
disadvantage). The bias is real, quantified, and goes in different directions
depending on dimension and label shape -- exactly the "biases either way" the
brief predicted.

Confirmed the same size-blindness for the whole external roster:
```
dagre_competitor.py:32:   g.setNode(node.id, { width: 120, height: 40 });   # hardcoded, every node
elk_competitor.py:122/137: {"id": ..., "width": 120, "height": 40}          # hardcoded, every node
igraph_competitor.py / networkx_competitor.py: no width/height reference at all
```
Only `dagua` itself is laid out AND scored under the same node_sizes convention
by construction (its own overlap-avoidance loss targets `graph.node_sizes`
directly). So the "no overlaps" composite term (worth 8/100 directed, 20/100
undirected) structurally favors dagua: it is the only engine that ever sees the
box sizes it is graded against.

**Measured real-world footprint of the overlap term specifically:**
```
zero_overlap_rate: dagua=100.0%, dagre=99.1%, elk_layered=99.1%, graphviz_dot=99.1%,
                    graphviz_neato=23.1%, graphviz_sfdp=39.8%, igraph_kamada_kawai=3.7%,
                    igraph_sugiyama=57.4%, nx_spring=29.6%
```
Note this is NOT purely a size-fairness artifact -- the low rates for
neato/sfdp/kamada_kawai/nx_spring are mostly architectural (force-directed
methods don't explicitly avoid overlap as a design goal, independent of box-size
knowledge), so I am not claiming this single-handedly explains those numbers.
But the size mismatch adds unquantified bias/noise on top of that architectural
gap, in both directions, on every graph, for every external engine.

**Severity rationale:** HIGH. Structural, sprint-wide (affects all 8 external
adapters, not one), directly touches a term that's part of every single
comparison, and is provably non-hypothetical (measured on real graphviz output).
I did not find evidence it is the dominant driver of any specific W/T/L verdict
(the architectural force-directed overlap gap likely dominates), but it is an
unquantified confound sitting inside the "no overlaps" score for literally every
non-layered external engine.

**Suggested fix (not applied):** Emit `width=`/`height=` (or engine-equivalent)
attributes derived from `graph.node_sizes` to every adapter that accepts them
(graphviz, dagre, elk all support this), so externals are laid out under the same
convention they're scored against.

---

### HIGH-3: Composite formula rewards degenerate/collapsed layouts; already manifesting in 62/972 real rows

**Files:** `dagua/metrics.py:1460-1513` (`composite_undirected`), `dagua/metrics.py:1394-1457`
(`composite`).

**Repro (ran, synthetic):**
```python
# /tmp/r80_composite_exploit.py, graph=grid_5x5 (25 nodes, undirected)
random baseline (torch.randn*50)        composite=29.300  overlap=7    crossing=0.372  cv=0.500
all-collapsed-to-origin (every node at (0,0)) composite=65.000  overlap=300  crossing=0.000  cv=0.000
tiny-bbox-cluster (randn*1e-6)          composite=53.581  overlap=300  crossing=0.000  cv=0.477
collinear-line                          composite=21.425  overlap=285  crossing=0.171  cv=0.679
huge-spread (randn*1e6)                 composite=47.870  overlap=0    crossing=0.258  cv=0.572
```
Collapsing every node onto the same point scores 65/100 -- HIGHER than an actual
random, non-degenerate layout at 29.3/100 -- because zero-length edges trivially
ace `edge_length_cv` (CV of a degenerate all-equal distribution is 0 -> full 40/40
undirected-weight credit) and `crossing_rate` (zero-length segments never register
as crossing under the segment-intersection test -> full 20/20 credit). Only the
binary 20-point overlap term correctly zeroes out; the other 80 points are
unguarded against this degenerate case.

**Repro (ran, production data -- not hypothetical):**
```python
# /tmp/r80_overlap_vs_composite.py over eval_output/r79_baseline/results.json
# rows with overlap_count>0 AND composite>60: 62 / 972
('grid_20x20', 'igraph_kamada_kawai', overlap=1242.0, composite=74.49, cv=0.0000000128, crossing=0.0)
('r79_weighted_mesh_10x12', 'igraph_kamada_kawai', overlap=385.0, composite=72.52, cv=0.062, crossing=0.0)
('grid_5x5', 'igraph_kamada_kawai', overlap=20.0, composite=74.53, cv=0.012, crossing=0.0)
('triangular_lattice_36', 'igraph_kamada_kawai', overlap=85.0, composite=74.48, cv=0.013, crossing=0.0)
# ... 58 more
```
`grid_20x20` under `igraph_kamada_kawai` has 1242 pairwise node overlaps (out of
400 nodes) and still scores 74.49/100 -- a "B" grade -- because Kamada-Kawai on a
symmetric grid produces geometrically regular (near-zero edge-length variance)
and non-crossing output even though nodes are stacked on top of each other. This
is the exact degenerate-layout pattern from the synthetic repro, occurring
organically.

**Did it flip any dagua-vs-external verdict?** Checked all 20 distinct graphs that
appear in the overlap>0/composite>60 flagged set (`/tmp/r80_check_exploit_flips_wtl.py`):
for 19/20, the row crowned "best external" for that graph has `overlap=0` (i.e. a
*different*, cleaner engine outscored the exploit-affected row, so the exploit
row was never actually selected as dagua's opponent). For the one exception,
`small_world_100`, `igraph_kamada_kawai` (overlap=328) IS the crowned best
external at composite 68.62 -- but dagua still wins outright at 91.75 (delta
+23.13), so the exploit does not flip that verdict either. Also confirmed dagua's
OWN rows never trigger this pattern: 0/108 dagua rows have `overlap_count>0 AND
composite>60`.

**Severity rationale:** HIGH despite zero verdicts flipped today. The exploit is
proven in production data (not a contrived edge case), affects 6.4% of all rows,
and its current non-impact on W/T/L looks like luck (no engine happened to be
BOTH the best-scoring AND exploit-affected on the same graph as dagua's closest
competitor) rather than a structural guarantee. As dagua's own algorithm evolves
(e.g. any future local-minimum collapse under an aggressive stress/annealing
schedule), it could earn undeserved composite credit the same way.

**Suggested fix (not applied):** Scale the overlap term by overlap *severity*
(fraction of node pairs overlapping, or total overlap area) rather than a binary
0/1, and/or gate `edge_length_cv`/`crossing_rate` credit on a minimum bounding-box
span so degenerate near-zero-extent layouts cannot ace those terms.

---

### MEDIUM-1: `composite_large` has no undirected variant -- currently dead for the 108-graph headline, but a latent landmine

**Files:** `dagua/metrics.py:1566-1601` (`composite_large`), `scripts/r79_baseline.py:903-925`
(`score_stored_metrics` -- selects `composite_large` whenever the full-tier fields
are absent).

`composite_large` (used for N>2000 graphs where only `quick()` metrics are
available) hardcodes the DIRECTED weight scheme (`dag_consistency: 30` of 100
points) with no undirected counterpart, unlike the full-tier path which correctly
branches `composite` vs `composite_undirected` via `composite_auto(metrics,
is_semantically_directed(...))`. Any undirected graph scored through this path
would have 30/100 points determined by a DAG-consistency metric that is
meaningless for it.

**Current impact: none.** `build_corpus()` filters to `max_nodes=500`
(`scripts/r79_baseline.py:456`), so every one of the 108 corpus graphs uses the
`full()` tier, and `score_stored_metrics` never falls through to
`composite_large` for the headline W/T/L (verified: `full_fields.issubset(...)`
is true for all 972 rows I inspected). This is a real defect but it is
NOT currently affecting the reported numbers. It would matter for the separate
scale-ladder benchmarks (`r79_scale_20k_smoke.json`,
`r79_scale_ladder_round2.json`) if those are ever used for dagua-vs-external
comparison on undirected large graphs -- I did not audit those files in depth
(out of the 108-graph corpus this brief scoped me to), so flag as unverified
scope for now.

### LOW-1: A parallel, unseeded stochastic crossing-count path exists but is not on the r79 scoring path

**Files:** `dagua/metrics.py:2074-2120` (`count_crossings`, default `seed=None` ->
falls through to `sampled_crossing_rate(..., seed=None)` which uses the global
torch RNG state for graphs with >500 edges), `dagua/metrics.py:2014-2049`
(`compute_all_metrics`, calls `count_crossings` with no seed at all).

This legacy path is genuinely non-deterministic (repeat calls on the same
positions could give different `crossing_rate` -> different composite), but
`scripts/r79_baseline.py` imports and calls only `evaluate()` (which routes
through `full()` at `dagua/metrics.py:1809`, which fixes `seed=0` explicitly).
`compute_all_metrics`/unseeded `count_crossings` is used elsewhere
(`dagua/eval/equivalence_metrics.py:541-542`, `scripts/definitive_fidelity_analysis.py:2216`,
a research script) -- outside r79 scope, flagged only as a landmine for whoever
reuses that path expecting reproducibility.

---

## Checked and clean

1. **Hand-recount of W/T/L matches the reported 56/8/29 legacy + 8/2/5 extended
   exactly.** Independent from-scratch script (`/tmp/r80_wtl_recount.py`), not
   reusing any of `scripts/r79_baseline.py`'s summarization code, re-derives
   best-external-per-graph and the win/tie/loss delta from raw `results.json`
   rows:
   ```
   population=extended: graphs=15 wins=8 ties=2 losses=5
   population=legacy:   graphs=93 wins=56 ties=8 losses=29
   ```
   Exact match to the documented headline. Total 972 rows = 108 graphs x 9
   engines, confirmed.

2. **Frozen store integrity.** All 972 rows have `status == "OK"` (zero
   ERROR/SKIP/timeout rows in the current store -- verified by direct scan, not
   just trusting `validate_store()`). 972 `positions/*.pt` files on disk exactly
   match the 972 `positions_path` references in `results.json` -- zero missing,
   zero orphaned. `results.rows.jsonl` line count (972) matches `results.json`
   row count (972) exactly. Current `get_test_graphs(max_nodes=500)` corpus
   (108 names) is set-identical to the graph names actually scored in the store
   -- zero stale, zero missing.
   Command: `/tmp/r80_wtl_recount.py` (status breakdown) + inline python
   comparing `os.listdir(positions/)` against `results.json` rows + a
   `current_corpus - stored` / `stored - current_corpus` set diff.

3. **Rescore-path metric-version consistency (the core of hypothesis 1, minus
   the P3a re-layout issue above).** `git log c80e970..HEAD -- dagua/metrics.py`
   returns ZERO commits -- the composite/composite_auto/composite_undirected/
   composite_large formulas have not changed since the original freeze, so
   `--dagua-only`'s practice of reusing frozen external composites verbatim
   (`run_dagua_only` in `scripts/r79_baseline.py:1429-1466`, which copies
   `external_rows` unchanged and only recomputes dagua's row via a fresh
   `run_engine` -> `composite_auto` call) has not actually produced a
   metric-version skew in the current store, because nothing on the external
   side has gone stale relative to nothing having changed. This conclusion is
   scoped to "as of today"; the mechanism has no automated staleness guard (see
   HIGH-1), so this could silently break the next time someone edits
   `metrics.py` without a matching `--rescore-only` pass.
   Confirmed externals are genuinely frozen at the byte level:
   `graphviz_neato` position for `er_500` is byte-identical from `c80e970`
   through current `HEAD`; `dagre`/`igraph_kamada_kawai` (added one commit later,
   at `672a941`) are unchanged since their single addition commit (`git log
   --follow` shows exactly one commit in each file's history).

4. **Directedness oracle (`dagua/eval/graphs.py:65-79`, `is_semantically_directed`).**
   It is a simple tag lookup (`"undirected" not in test_graph.tags`), not a
   structural analysis -- confirmed by reading the source. The brief's suggested
   structural check (reciprocal edge fraction, acyclicity via networkx) turned
   out to be UNINFORMATIVE for this corpus: sampled 14 graphs across
   undirected/dag/scc/weighted/r79_ext/social/random/community tags and found
   `recip_frac == 0.000` for every single one (`/tmp/r80_directedness_audit.py`),
   because BOTH directed and undirected graphs in this corpus are stored as a
   single-orientation edge list (undirected edges get one arbitrary direction,
   not a reciprocal pair) -- the docstring says as much
   ("their stored edge orientation is an implementation artifact"). `is_dag=True`
   for all 14 sampled graphs too, for the same reason (single-orientation edge
   lists constructed via canonical/creation-order tend to be trivially acyclic
   regardless of true semantics). So I fell back to reading each flagged graph's
   construction code/docstring directly and cross-checking against its tag:
   `make_erdos_renyi` (correctly `undirected`), `make_real_karate_graph`
   (correctly `undirected`), `citation_dag_300`/`outerplanar_dag_20` (correctly
   directed, real DAGs), `r79_weighted_skew_dag_6x10` (correctly directed). Two
   graphs looked suspicious on first pass (`powerlaw_500` tagged directed
   despite superficially resembling the undirected `ba_500`/`scale_free_ba_120`
   scale-free graphs, and `sparse_pair_50`/`dense_pair_50` tagged directed with
   no dag/dependency signal in their tags) but both resolved cleanly on reading
   their construction functions: `make_powerlaw_dag` (`dagua/eval/graphs.py:3557`)
   explicitly builds a DAG via "earlier -> later" preferential attachment
   (docstring: "DAG with power-law out-degree distribution"), and
   `sparse_pair_50`/`dense_pair_50`'s own descriptions say "half of a matched
   sparse-versus-dense DAG pair" (`dagua/eval/graphs.py:4865-4877`). Zero
   confirmed mismatches found in 16 manually-verified graphs.
   NOTE: A SEPARATE, unrelated directedness oracle exists at LAYOUT time
   (`_infer_semantically_directed`, used by dagua's own engine routing, not by
   `dagua/eval/graphs.py`) which a parallel r80 stream (S4) has already found
   real bugs in (mislabels karate/sbm_4x30/ba_120/small_world_100/grid_5x5/
   weighted_community/weighted_small_world as directed, transformer_layer as
   undirected) per `.project-context/research/r79_native/r79-native_STATE.md`'s
   2026-07-08 kickoff note. That is a different function in a different file
   (layout routing, not eval scoring) and is out of this audit's scope --
   flagged here only so the two "directedness" findings aren't conflated.

5. **Metric determinism on the actual scoring path.** `sampled_stress`
   (`dagua/metrics.py:655`) samples source/target indices via
   `_deterministic_sample_indices` -- evenly-spaced index arithmetic, no RNG at
   all. `sampled_crossing_rate` is called from `full()` (the function
   `evaluate(..., tier="full")` actually uses, `dagua/metrics.py:1809`) with
   `seed=0` fixed. Live-verified layout+metric determinism, not just read the
   code: ran dagua's actual `layout()` on `er_500` (500 nodes, the largest
   legacy graph) twice today with `seed=42` -- `torch.equal(run1, run2) == True`,
   max abs diff `0.0`. Also ran it against the CURRENTLY STORED position for the
   same graph: exact match, max abs diff `0.0` -- confirming the current frozen
   store is reproducible from today's code, not just self-consistent by
   assumption. (This result also underpins HIGH-1: it rules out nondeterminism
   as the explanation for the large `2c19c45`-era position delta, leaving
   "genuine accumulated algorithm-code change" as the only remaining
   explanation.)
   Command: `/tmp/r80_determinism_check.py er_500` and
   `/tmp/r80_determinism_check.py parallel_multiedge_bundle` (small-graph
   sanity check, also exact match).

6. **Tie-band symmetry and best-external exclusion.** `TIE_BAND = 0.5`
   (`scripts/r79_baseline.py:28`); `summarize_wtl` computes
   `delta = dagua.composite - best_external.composite` and applies
   `delta > TIE_BAND -> WIN`, `delta >= -TIE_BAND -> TIE`, else `LOSS`
   (`scripts/r79_baseline.py:893-899`) -- symmetric band of total width 1.0
   centered on zero, no directional bias. `graph_best_external`
   (`scripts/r79_baseline.py:840-863`) filters `status == "OK"` and
   `composite is not None` before taking the max, so errored/skipped engines
   are correctly excluded rather than silently scored as a loss or a phantom
   win. (Could not construct a case where this filtering mattered in the
   current store since zero rows are non-OK, but the code path is correct by
   inspection and my independent recount -- which reimplements this filter from
   scratch -- reproduces the exact same numbers, which would not happen if the
   exclusion logic were wrong in either script.)
