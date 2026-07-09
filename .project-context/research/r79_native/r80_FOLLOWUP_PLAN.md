# r80 Follow-up Sprint Plan (native placement, continued)

Prereq reading: r79-native_SUMMARY.md, then P3B2_STRESS_FORENSICS.md (the single most
important doc), then r79_DESIGN.md. Start by verifying r79/native still builds and re-running
scripts/r79_baseline.py --dagua-only to confirm the honest baseline (expect ~56/8/29 + 8/2/5).
The frozen store is committed; the harness is in scripts/r79_baseline.py.

## The one problem that matters: the undirected class (~26 losses)
Most of the 34 losses are semantically-undirected graphs (social/community/SBM/small-world/
mesh/lattice/regular/scale-free) where force engines (graphviz_sfdp/neato, nx_spring) win and
dagua's layered default cannot. The r79 bet -- route these to a stress core -- failed 3x.

### What we KNOW from forensics (do not re-derive)
- Stress-SGD itself is HEALTHY: it improves the honest score dramatically before the final
  stages on every target graph. The core optimizer is not the problem.
- The overlap projector (projection.py) does NOT converge on dense overlap cliques (real
  torch advanced-index accumulation bug). P3b3 rewrote it toward index_add_ accumulation but
  the full fixes-1-5 bundle still failed the sweep gate (54/8/31). The projector fix alone
  gave 5/26 flips but net regressions elsewhere.
- HEADROOM PROOF: sfdp's own positions + a convergent overlap cleanup score +12..20 over best
  external. So the winning layout EXISTS in the space; the issue is our composition arriving
  at it without wrecking length-CV / crossings.

### Genuinely different angles for r80 (pick by evidence, do not just re-run P3b3)
1. **ROUTE TO OUR OWN REIMPLEMENTED FORCE ENGINES instead of a bespoke stress core.** We have
   bit-faithful sfdp/neato/fr/fa2 pipelines. For declared-undirected graphs, the honest,
   low-risk win may be: route to classic_sfdp/neato internally, THEN apply our convergent
   size-aware overlap projection (the headroom proof says this beats external). This sidesteps
   the "our stress core loses on length-CV" problem entirely -- we adopt the engine that wins,
   then add the one thing external engines lack (overlap removal). STRONGEST CANDIDATE.
2. **Interrogate the composite itself.** GD-2025 research (see r79 SOTA survey in
   the P2b-era notes) shows metric-identical drawings differ to humans, and our composite may
   over-weight terms where sfdp wins by construction. A principled re-weighting toward the
   empirically human-validated 5 proxies (Gabriel ratio, edge-length uniformity, node
   uniformity, angular resolution, crossings) could change which layouts "win" -- but this is
   DANGEROUS (moving the goalposts); only do it with the standard-corpora holdout as arbiter,
   never to chase our own benchmark.
3. **Finish the projector fix in isolation.** The projector bug is real and shared with the
   default path. Land ONLY the convergent projector (index_add_ + damping + iterate-to-zero)
   as a correctness fix, gated on a strict no-regression sweep, INDEPENDENT of routing. It
   gave 5/26 flips solo; some may be jitter-stable keepers even if the bundle failed.
4. **Resistance-distance stress targets** (Omega, arXiv:2512.21901) -- never tried; the design
   flagged it. Isometrically-embeddable targets may fix the length-CV loss that sinks stress
   layouts on community graphs. Medium risk, novel.

Recommended r80 order: (3) land the projector correctness fix solo -> (1) route-to-own-force
+ projection for undirected -> measure -> only then consider (2)/(4).

## Other residuals (lower priority)
- **Hybrid v2 (SCC)**: built, routed off (regressed forced SCC targets 58->49). Needs better
  internal-SCC layout + a stronger predicate than SCC-coverage>25%. native_hybrid_v2.py exists.
- **1M-node runtime**: multilevel path fits memory (1.2GB) but misses the <10min target;
  bottleneck is coarsest-solve stress-SGD time even after contraction fix. Needs a
  sampled/pivot coarsest solver or a faster refine schedule. native_stress_ml.py.
- **P5 layered clusters**: recursive layered cluster placement regressed 49/9/35; the
  "dot-algorithm core difficulty" (cluster-contiguous rank assignment). Deep. native_stress
  recursive clusters DO work (9eb3d2c) -- resolve the warning decision and merge that part.
- **Quality knob**: merged; could add per-quality-level benchmark validation and doc/gallery
  regen (scripts/build_glossary.py etc. per CLAUDE.md maintenance checklist).

## Standard-corpora HELDOUT eval (explicit JMT todo; also in .project-context/todos.md)
Once happy on our 108-graph suite, run on Rome / North(AT&T) / SuiteSparse samples with the
established metric suite. HOLDOUT DISCIPLINE: do NOT tune against them before the first full
eval. This is the honest generalization test AND the publication evidence base (the router/
portfolio result is genuine white space -- see SUMMARY). Building the harness (fetch + wire
adapters, no tuning) is safe delegatable prep -- may already be dispatched (check
scripts/ for r79_stdcorpora* / r80_*).

## P6 (finish r79) -- remaining to declare the sprint mergeable to develop
1. Resolve the P5 warning decision + merge P5's native_stress recursion.
2. Fresh FULL benchmark (all 8 external engines, not just --dagua-only) on the merged head to
   confirm the consolidated scoreboard; regenerate BASELINE.md.
3. Opus visual audit of the full merged head on the top improved + representative graphs.
4. Adversarial review (Claude + Codex) of the merged diff -- especially the projector, the
   router predicates, and the quality-knob override precedence.
5. Rebuild generated docs (glossary/gallery/explainer/visual-audit per CLAUDE.md).
6. Merge r79/native -> develop; delete merged branches; sync notes.

## Branch inventory (as of 2026-07-07)
- r79/native (worktree dagua-native) -- MERGED head: P1..P4 + P3d quality knob. This is the trunk.
- r79/p5-clusters (9eb3d2c, worktree dagua-native-p1) -- HELD, warning decision pending.
- r79/p3b-wip (cd04c7e, in dagua-native history) -- failed stress route-flip attempts (residual).
- r79/p3-hybrid (6f4b246) -- MERGED (inert). r79/p4-scale (1cf774c) -- MERGED (opt-in).
- r79/p2c-fix (e2a9f8b) -- MERGED. r79/p3d-quality (208342a) -- MERGED.
Sweep helpers: many stale branches can be swept after P6 merge per the branch-sweep discipline.

## How to resume cold (checklist)
1. Read this + SUMMARY + P3B2_STRESS_FORENSICS.
2. cd ~/.claude/worktrees/dagua-native; .venv/bin/python -c "import dagua" (or recreate venv:
   uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]" python-igraph; ln -s the
   repo node_modules for elk/dagre).
3. scripts/r79_baseline.py --dagua-only to confirm baseline.
4. Pick the r80 angle (recommend projector-solo then route-to-own-force). Dispatch via codex
   with tight touched-file test gates + KNOWN_RED_TESTS.md deselects (NOT bare pytest -x).

## P6a harness status (2026-07-07)
scripts/r79_stdcorpora_eval.py + loaders (.graph/.gml/.mtx) + tests BUILT and green on
r79/p6a-stdcorpora (6bbf9a2). Fetch of graphdrawing.org/data/rome/rome.tar.gz FAILED (SSL
hostname mismatch on their cert) -> no live corpus yet. FOLLOWUP: fetch Rome/North from a
working mirror (GitHub mirrors of Rome-Lib exist; SuiteSparse via ssgetpy or sparse.tamu.edu),
drop into eval_output/stdcorpora/, run scripts/r79_stdcorpora_eval.py. Harness works on any
dropped-in corpus; no tuning against it (holdout).

## Adversarial review findings (2026-07-07, Opus over a33afa8..r79/native; verdict SAFE TO MERGE)
Default path empirically BIT-IDENTICAL to pre-sprint; all heavy subsystems off-by-default+gated.
Deferred hardening items (none merge-blocking; address in r80):
1. [PARTIALLY DONE] scc.py recursive Tarjan: recursion-limit LEAK fixed now (save/restore in finally,
   commit on r79/native). STILL TODO before hybrid_v2 auto-routing: rewrite to ITERATIVE Tarjan
   (explicit stack) to remove the deep-graph SEGFAULT risk (Python doesn't raise catchable
   RecursionError past real C-stack depth).
2. Quality knob cannot detect an explicitly-set multi_start_k=1 (== default sentinel) -> gets
   overwritten by quality>=high. Documented limitation (LayoutConfig doesn't track constructor-set
   fields). Fix: track explicitly-set fields, or use a sentinel default (None).
3. native_stress_ml _assert_memory_budget uses a LINEAR (N+E)*3 estimate; under-counts if any
   guarded stage is super-linear. Add an N^2 term or assert stages are O(N+E).
4. native_stress_ml violates decomposable-ops: _hub_stride_sample/_induced_sample_problem/
   _interpolate_from_sample/_apply_local_repulsion are private funcs, should be registered ops.
5. CLARIFICATION: projection.py is NOT modified in r79/native -- the P3b3 projector fix lives on
   r79/p3b-wip (unmerged). The merged head's shared _project_exact still has last-write-wins
   advanced-index accumulation (converges because damped+iterated). r80 angle 3 lands the real fix.
