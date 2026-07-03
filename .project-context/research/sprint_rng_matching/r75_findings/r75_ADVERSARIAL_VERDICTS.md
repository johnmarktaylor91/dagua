# r75 adversarial verdicts

Date: 2026-07-01
Repo: `/home/jtaylor/projects/dagua`, `develop` at `89ed3c3`

This report is intentionally conservative. A source-level difference is not a fix approval unless it is
verified against the version that actually produces the benchmark reference and has a blast-radius gate.

## Version audit

- Graphviz: benchmark binary is `dot - graphviz version 7.0.5 (20221231.0122)`. The checkout at
  `/home/jtaylor/projects/_references/graphviz` is `233597cd4`, not the benchmark source. I used
  `git show 7.0.5:<path>` for Graphviz rulings. The installed binary is not linked to GTS/Triangle
  (`ldd $(which dot) | rg -i 'gts|triangle'` returned nothing), so `AM_PRISM` overlap removal is
  compiled out in the installed environment. Source: `7.0.5:lib/sfdpgen/sfdpinit.c:271-275`,
  `7.0.5:lib/neatogen/adjust.c:1089-1093`.
- igraph: installed `python-igraph` is `1.0.0`. The local source checkout is `d03122b` on `main`,
  grafted, with no local `1.0.0` tag. Igraph source citations from `_references/igraph` are therefore
  not version-pinned to the installed wheel. Treat igraph source-only claims as unverified until a
  matching 1.0.0 source tree or installed-binary trace confirms them.
- OGDF: benchmark runner is `scripts/ogdf_runner`, built by
  `scripts/rng_match/build_ogdf_runner.sh` from `/home/jtaylor/tools/ogdf-src`, tag
  `foxglove-202510`, SHA `5b6795655399b9d8e2921afec9d97bab9107d5ee`; not
  `/home/jtaylor/projects/_references/ogdf` (`1a40505`). OGDF rulings cite `/home/jtaylor/tools/ogdf-src`.
  Runner sets `randSeed(seed)`, `StopCriterion::FixedIterations`, and `fixedIterations(...)` for FMMM at
  `scripts/ogdf_runner.cpp:320-328`.
- `eval_output/fidelity_definitive/r75_truebaseline.jsonl` exists but has only 28 rows. I used it as
  a direct correction where present, not as a full replacement for the five target JSONs.

## Verdicts

### SFDP

1. **No-canonical-reference tier for `classic_sfdp_theta04`, `classic_sfdp_theta08`, `classic_sfdp_steps200`: APPROVE-WITH-CHANGES.**
   Graphviz 7.0.5 `sfdp` reads `K`, `repulsiveforce`, `levels`, `smoothing`, `quadtree`, `beautify`,
   `overlap_shrink`, `rotation`, and `label_scheme`, but not `theta` or `maxiter`:
   `7.0.5:lib/sfdpgen/sfdpinit.c:238-247`. Defaults hard-code `bh=0.6` and `maxiter=500`:
   `7.0.5:lib/sfdpgen/spring_electrical.c:42-47`. Do not change dagua behavior. Mark these variants
   non-counting/non-canonical for the north-star score, subject to JMT sign-off.

2. **Change `GraphvizRandom.permutation()` from raw modulo to rejection sampling: REJECT.**
   Sonnet is correct for the pinned source. Graphviz 7.0.5 uses raw `rand() % n` in `irand()` and
   `random_permutation()` calls `irand(len)`: `7.0.5:lib/sparse/general.c:20-41`. SFDP coarsening calls
   that permutation at `7.0.5:lib/sfdpgen/Multilevel.c:99-111`. Dagua's raw-modulo permutation is the
   7.0.5-faithful consumer. The Codex report read the newer HEAD refactor.

3. **Set `overlap=false` or otherwise undo default SFDP overlap handling: REJECT.**
   In this installed build, Graphviz's default prism overlap path is not active because GTS/Triangle is
   absent; `sfdpinit.c` falls back to `graphAdjustMode(..., 0)`: `7.0.5:lib/sfdpgen/sfdpinit.c:271-275`.
   A blanket adapter change would compare against a non-default reference.

4. **Keep the r74 `p_neg2` clamp: APPROVE as already-landed, no new code.**
   Graphviz clamps `repulsiveforce=-2.0` through `late_double(..., minimum=0.0)` and later resets
   nonnegative effective `p` to `-1`; the source path starts at `7.0.5:lib/sfdpgen/sfdpinit.c:238-239`.
   The corrected true-baseline rows also show `p_neg2` now behaving like the default where present.

5. **Shared-control disconnected SFDP component loop / packSubgraphs port: NEEDS-EXPERIMENT.**
   The source difference is plausible: Graphviz uses one `spring_electrical_control` over components
   after `tuneControl()` (`7.0.5:lib/sfdpgen/sfdpinit.c:268-295`), while dagua recurses per component.
   But no decisive before/after exists. Minimal experiment: in a scratch branch, thread one shared
   control-like state through `_layout_graphviz_sfdp_components`, preserve `K`, `random_start`,
   `adaptive_cooling`, and `step`, then run 5 seeds on `parallel_cycles_4x5`,
   `disconnected_label_cycle_collage`, `multi_component_80` against saved Graphviz references.

6. **`parallel_cycles_4x5` degenerate SFDP packing fix: NEEDS-EXPERIMENT.**
   Symptom is real in the reports, but mechanism is not isolated. First inspect saved positions and
   component bboxes before touching shared neato pack code.

7. **SFDP FP-chaos floor for the remaining connected rows: NEEDS-EXPERIMENT.**
   Source parity on major parameters is good, but a floor claim requires perturbation evidence. Run the
   proposed 1-ULP initial-position or force-order perturbation on one connected target such as
   `grid_5x5` and compare final stress/crossing deltas to observed dagua-vs-Graphviz gaps.

### FMMM

8. **Port OGDF NMM as explanation/fix for all 29 OGDF-FMMM rows: REJECT.**
   Codex's broad claim is false for the runner's OGDF version. Although default FMMM selects NMM
   (`/home/jtaylor/tools/ogdf-src/src/ogdf/energybased/FMMMLayout.cpp:283-288`), the NMM class falls
   back to exact repulsion below 175 nodes:
   `/home/jtaylor/tools/ogdf-src/src/ogdf/energybased/fmmm/NewMultipoleMethod.cpp:121-190`.
   This cannot explain the small-graph rows. It is relevant only to levels/components with
   `numberOfNodes() >= 175`, mainly `random_dag_200`.

9. **Coincident-node RNG jitter in FMMM repulsion: NEEDS-EXPERIMENT.**
   The code difference is real. OGDF jitters coincident positions and consumes `randomNumber()`:
   `numexcept.cpp:48-70`, `numexcept.cpp:169-181`; dagua zeros distance-0 pair forces:
   `dagua/layout/ops/pipelines/fmmm.py:562-585`. I attempted a monkeypatch counter probe, but the
   run exceeded the cheap-probe bound before producing output. Minimal decisive experiment: instrument
   `_ogdf_fmmm_tensor_repulsive_forces` to count off-diagonal zero-distance pairs for
   `deep_chain_20`, `grid_5x5`, `weighted_chain_20`, `asymmetric_hourglass_hub` under
   `classic_fmmm_steps10`; approve a fix only if triggers are observed in failing rows.

10. **FMMM oscillation angle formula swap: NEEDS-EXPERIMENT.**
    OGDF uses `atan2(dy2,dx2) - atan2(dy1,dx1)`:
    `/home/jtaylor/tools/ogdf-src/include/ogdf/basic/geometry.h:134-149`; dagua uses `atan2(cross,dot)`:
    `dagua/layout/ops/pipelines/fmmm.py:168-196` and tensor equivalent at `fmmm.py:748-752`. Since the
    result feeds `ceil(angle / 0.52359878)` (`FMMMLayout.cpp:1285-1299`), this is plausible but
    unproven. First run an in-memory formula swap on 3 failing small rows and require a measurable
    position or metric shift.

11. **Advanced multilevel placement port: APPROVE-WITH-CHANGES.**
    Source confirms OGDF's Advanced placement uses placement sectors, same-solar-system candidates,
    random sector placement, and waggle: `Multilevel.cpp:405-465`, `580-656`. Gate strictly to
    `_layout_ogdf_fmmm_multilevel_fidelity` and local components with `local_nodes > 50`
    (`dagua/layout/ops/pipelines/fmmm.py:1787-1825`). Do not touch the single-level path.

12. **Multilevel coarsening/prolongation RNG stream port: APPROVE-WITH-CHANGES.**
    OGDF seeds global RNG once and `Node_Set.set_seed(rand_seed)` per level:
    `Multilevel.cpp:55-81`, `126-136`; dagua's older hierarchy helper uses one continuous
    `random.Random(seed)`: `dagua/layout/ops/fmmm.py:704-725`. This is a real multilevel mismatch,
    but implement only with a golden trace of sun selections/prolongation random draws for one
    >50-node component. Blast radius: multilevel FMMM only.

13. **Rebucket `classic_fmmm_graphviz_fdp_fidelity` rows out of OGDF FMMM: APPROVE.**
    Those variants target Graphviz FDP, not OGDF FMMM. Do not validate OGDF FMMM fixes on them.

### Sugiyama

14. **Graphviz dot x-coordinate network-simplex port: APPROVE-WITH-CHANGES.**
    This is the strongest Sugiyama finding. Graphviz dot x assignment builds aux constraints, runs
    `rank(g, 2, nsiter2(g))`, then copies `ND_rank` to x:
    `7.0.5:lib/dotgen/position.c:120-135`, `218-343`, `570-584`. Dagua always calls BK x assignment
    regardless of graphviz fidelity: `dagua/layout/ops/pipelines/sugiyama.py:106-124`,
    `dagua/layout/ops/sugiyama.py:1727-1738`. Stage it behind graphviz fidelity only. Initial stage
    may skip labels/clusters, but report residuals for label/cluster graphs explicitly.

15. **Graphviz omega/virtual edge weight table: APPROVE-WITH-CHANGES.**
    Graphviz 7.0.5 uses `C_EE=1`, `C_VS=2`, `C_SS=2`, `C_VV=4` and applies `virtual_weight()`:
    `7.0.5:lib/dotgen/mincross.c:1858-1888`; `class2.c:84-95` calls it during virtual-chain creation.
    Dagua has no equivalent in the graphviz mincross path. Gate to graphviz fidelity and add controls
    for currently-passing graphviz rows.

16. **Graphviz `build_ranks` init order and mincross pass schedule: APPROVE-WITH-CHANGES.**
    Graphviz runs pass 0/1 with `build_ranks()` and a different schedule before pass 2:
    `7.0.5:lib/dotgen/mincross.c:815-855`, `1352-1415`, defaults at `1944-1952`. Dagua starts
    `ordered_layers = [sorted(layer)]` before `graphviz_mincross()`:
    `dagua/layout/ops/sugiyama.py:1255-1270`. Implement in graphviz mode only, after or alongside
    the x port.

17. **Full class2/left2right/ports/clusters mincross port: APPROVE-WITH-CHANGES.**
    Correct direction, high blast radius. Phase it after the x port and omega/init fixes. Do not mix
    this into igraph/default ordering.

18. **Igraph LP objective change to IN/IN: NEEDS-EXPERIMENT.**
    The unpinned local source indeed fills both `indegs` and `outdegs` using `IGRAPH_IN`:
    `_references/igraph/src/layout/sugiyama.c:588-615`. Dagua uses out-strength minus in-strength:
    `dagua/layout/ops/sugiyama.py:495-543`. But the source tree is not installed 1.0.0. Do not land
    an objective revert until verified against installed igraph 1.0.0 source or a runtime trace that
    distinguishes IN/IN from out-in on a small DAG.

19. **Replace HiGHS with GLPK-like simplex for igraph ranks: NEEDS-EXPERIMENT.**
    The reference source uses GLPK with presolve off when compiled with GLPK:
    `_references/igraph/src/layout/sugiyama.c:563-656`. Installed wheel build flags are not verified.
    First prove rank divergence is the first failing stage on target rows.

20. **Igraph BK conflict ordinal-edge behavior: NEEDS-EXPERIMENT.**
    The unpinned source loops gathered neighbor count but indexes `IGRAPH_FROM(graph, j)` /
    `IGRAPH_TO(graph, j)`: `_references/igraph/src/layout/sugiyama.c:898-944`. Runtime impact and
    version match are unproven. Add a pure Python emulation dump for `multiscale_skip_cascade` and
    `regular_4_40` before code.

21. **Igraph BK median/anchor fix: REJECT as stated; keep only targeted follow-up.**
    The suggested anchor mismatch is not present in the local source: igraph chooses the minimum-width
    alignment and shifts left/right alignments exactly as dagua's documented approach implies:
    `_references/igraph/src/layout/sugiyama.c:990-1029`; dagua median-of-four is
    `dagua/layout/ops/sugiyama.py:2291-2313`. Do not change this unless a version-pinned installed
    source says otherwise. Isolated-node/tie-order hypotheses remain NEEDS-EXPERIMENT.

### Classical MDS and small tails

22. **Classical MDS disconnected component split plus DLA merge: APPROVE-WITH-CHANGES.**
    This is confirmed and scoped. Dagua currently computes one global finite-filled distance matrix:
    `dagua/layout/ops/pipelines/classical_mds.py:241-265`,
    `dagua/layout/ops/graph_utils.py:347-351`. Igraph splits weak components, lays each out, DLA
    merges, then reorders rows: `_references/igraph/src/layout/mds.c:223-280`. DLA specifics are at
    `merge_dla.c:100-178`, `266-298`, and grid placement at `merge_grid.c:70-121`. Gate only
    `len(components) > 1`; connected path must be byte-identical before/after. Target rung-3 first if
    exact igraph RNG parity is too costly.

23. **Connected classical MDS as confirmed degenerate-eigenspace floor: REJECT as confirmed floor; NEEDS-EXPERIMENT before closure.**
    Dagua's docstring records prior evidence (`classical_mds.py:53-65`) and igraph source uses top
    LAPACK eigenvectors (`_references/igraph/src/layout/mds.c:113-131`). But project rules require
    current FP-chaos/eigenspace evidence. Run the cheap driver experiment (`evr`, `evx`, `evd`) plus
    eigenspace-invariant comparison on the 7 connected graphs. No code change until that result.

24. **UMAP negative-sampling RNG port: NEEDS-EXPERIMENT.**
    Params are matched, but first divergence is not traced. Instrument a 3-5 node graph for fuzzy
    graph, init, and first SGD epoch before changing RNG.

25. **GEM residual as floor/no fix: NEEDS-EXPERIMENT.**
    Inherited r74 characterization is not enough for these five rows. Run a 1-ULP or update-order
    perturbation on one residual target before closing.

26. **Maxent disconnected component split: REJECT.**
    The r74 revert was correct. Do not reintroduce blanket component splitting. The three remaining
    `random_dag_50` rows need a first-divergence trace of OGDF initial layout/distance fill, not the
    reverted fix.

27. **DrL floor/no fix: NEEDS-EXPERIMENT.**
    Pipeline comments about float32/density-grid sensitivity are not sufficient for a final floor
    label. Trace `real_lesmis_77::classic_drl_coarsen`, which is an outlier.

28. **Neato residual as no-op/floor: NEEDS-EXPERIMENT.**
    Both rows are disconnected and near-margin, but Graphviz pack/RNG/CG details remain untraced.
    Do not change shared packers without a `pack=false` and single-RNG-stream probe.

### Metrics and criteria

29. **Broad exact-crossing margin widening / 1-crossing blanket floor: REJECT.**
    North star is statistically identical. A deterministic `D=3` vs `R=2` crossing count is not
    identical. Sonnet's distinction is the honest policy: most zero-spread small-count failures are
    algorithm/layout differences, not margin bugs. The count discrepancy is definitional: Codex counted
    any nonzero crossing delta (235), while Sonnet counted TOST-gated crossing failures (163).

30. **Sampled crossing SE propagation and denominator fix: APPROVE-WITH-CHANGES.**
    `sampled_crossing_rate()` computes `crossing_se` but only `crossing_estimated_total` survives:
    `dagua/metrics.py:823-832`, `scripts/definitive_fidelity_analysis.py:1726-1745`. It also scales
    a valid-pair conditional rate by all `E choose 2`, including ineligible adjacent pairs:
    `dagua/metrics.py:803-831`. Gate to `cross_sampled=True`; exact rows should not move.

31. **Exact/sampled crossing predicate consistency: APPROVE-WITH-CHANGES.**
    Vector path counts collinear overlaps (`dagua/metrics.py:146-201`); exact path calls a scalar
    predicate at `dagua/metrics.py:2068-2085` with different degeneracy behavior. Fix with replay and
    controls; do not combine with margin widening in the same change.

32. **Fractional near-zero crossing floor: NEEDS-EXPERIMENT.**
    Only consider after per-seed histograms prove a noisy 0/1 seed-average population. Expected impact
    is small, about 10-15 rows. Deterministic zero-spread rows stay divergent unless layout fixes close
    them.

33. **Huge-graph bounded-time approximate scoring: APPROVE-WITH-CHANGES.**
    Needed, but stale worklist counts are unresolved. `r74_analysis.jsonl` does not cover the full
    big-graph tier; use a fresh capped current-develop rescore, not the stale 337 baseline. Any
    approximate leg with CI wider than margin must return `APPROX_UNRESOLVED`, not pass.

34. **Engine-level population equivalence: APPROVE-WITH-CHANGES as non-north-star metadata only.**
    Useful aggregate claim, but cannot certify individual combos as identical.

35. **`QUALITY_SUPERIOR_DISTINCT` / dagua-better tier: APPROVE-WITH-CHANGES.**
    Add as triage metadata only. Do not feed it into `quality_identical_raw` or any north-star
    identical count. This matches the current one-sided NP policy
    (`scripts/definitive_fidelity_analysis.py:1642-1669`) without turning stress/cross into
    one-sided tests.

## Recommended implementation order

1. **Metadata/criteria infrastructure first.** Add no-canonical-reference handling for SFDP
   theta/steps variants, sampled-crossing SE/denominator plumbing, exact/sampled predicate replay,
   and the quality-superior-distinct tier. These are scoring/reporting changes and should not touch
   layout code.
2. **Classical MDS disconnected DLA.** Highest-confidence layout fix with a clean graph-class gate.
   Regression gate: all connected `classic_classical_mds_{default,igraph_fidelity}` combos must be
   byte-identical before/after; rerun the 16 disconnected targets and the r74 reverted regression set.
3. **Graphviz Sugiyama staged port.** Land x-network-simplex first on simple graphviz rows
   (`binary_tree`, `bipartite_4_3_4`), then omega weights, then `build_ranks`/mincross schedule.
   Regression gate: graphviz-fidelity only; default/igraph Sugiyama must be unchanged.
4. **FMMM experiments before FMMM code.** Run angle-swap and zero-distance-trigger probes. If
   positive, implement the smallest gated fix. Multilevel RNG/Advanced placement may proceed only for
   `local_nodes > 50` with a golden trace.
5. **Tail experiments.** UMAP first-divergence, connected-MDS eigenspace driver test, GEM/DrL
   perturbation, maxent initial-layout trace, neato pack/RNG probe. Do not land floor labels or code
   changes without those results.

## Blast-radius notes

- Any fix not gated to `fidelity_mode` and graph class is suspect. The r74 failure mode was broad
  component/packing changes breaking already-good rows.
- Shared packers are high risk (`neato`, SFDP disconnected, FMMM/FDP paths). Require before/after
  checks for every algorithm sharing the helper.
- Graphviz-version citations must stay pinned to `git show 7.0.5:` until the benchmark binary changes.
- Igraph source work must not cite `_references/igraph` as installed truth until a 1.0.0 source tree is
  available or an installed-wheel trace proves the behavior.

## Tests/probes run for this critique

- `dot -V`: confirmed Graphviz 7.0.5.
- `ldd $(which dot) | rg -i 'gts|triangle'`: no GTS/Triangle linkage found.
- `git show 7.0.5:` source reads for SFDP random/permutation, attrs, overlap, dot position, mincross.
- `/home/jtaylor/tools/ogdf-src` source reads for FMMM/NMM/numexcept/multilevel at
  `foxglove-202510`.
- `r75_truebaseline.jsonl` parsed; present but only 28 rows, so not used as a complete rerank.
- Attempted FMMM zero-distance monkeypatch probe; stopped after exceeding the cheap-probe bound with
  no output. Verdict downgraded to NEEDS-EXPERIMENT.
