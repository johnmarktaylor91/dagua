<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 28 OGDF INFRASTRUCTURE: rebuild OGDF runner with seed plumbing, then
generate multi-seed reference cache for ogdf_* targets.

## Why this matters

Currently the bounded fidelity comparators report `not_tested` for OGDF-based
references (`ogdf_fmmm`, `ogdf_gem`, `ogdf_pivot_mds`, `ogdf_stress`) because
the cache has only 1 seed per graph per engine. With multi-seed cache:
- TOST equivalence test can actually run (needs target variance)
- fmmm at median 0.016 will likely flip from `DIVERGENT_FROM_DETERMINISTIC_REF`
  (classification artifact) to `CONVERGED_at_<margin>`.
- gem residual gets sharper diagnostic fingerprint.

## Your job

### Phase A: Build OGDF dev libs (if not installed)

The OGDF source is cloned at `/home/jtaylor/projects/_references/ogdf/`. Build
it locally (NO sudo apt-install — operate in user-space):
```bash
cd /home/jtaylor/projects/_references/ogdf
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make -j$(nproc)
make install   # installs to ~/.local/{include,lib}
```

If make fails on dependencies, document and fall back to "skip OGDF rebuild"
plus "skip cache regen" in your SUMMARY.

### Phase B: Rebuild scripts/ogdf_runner with seed plumbing

`scripts/ogdf_runner.cpp` is the C++ binary that wraps OGDF. Currently:
- The seed parameter for stochastic OGDF algorithms is dropped or hardcoded
  (R23 codex flagged "del seed" or equivalent in the runner).

Add seed plumbing: every algorithm that has stochastic init should:
1. Accept a `--seed N` flag (or whatever the runner uses for arguments)
2. Wire it into the OGDF algorithm's random source (typically
   `OGDF_RANDOM_SEED(seed)` or via `setSeed` on the layout)

Build:
```bash
cd /home/jtaylor/projects/dagua
g++ -std=c++17 -O2 scripts/ogdf_runner.cpp \
    -I$HOME/.local/include \
    -L$HOME/.local/lib \
    -lOGDF -lCOIN \
    -o scripts/ogdf_runner
```
(Adjust as needed for your install prefix.)

Verify:
```bash
./scripts/ogdf_runner --help
./scripts/ogdf_runner --algorithm fmmm --seed 42 --input <graph> --output <pos>
./scripts/ogdf_runner --algorithm fmmm --seed 43 --input <graph> --output <pos>
diff <both outputs>  # should differ since seed differs
```

### Phase C: Run multi-seed cache regen

For each of `ogdf_fmmm`, `ogdf_gem`, `ogdf_pivot_mds`, `ogdf_stress` and
the bounded 5-graph subset:
- linear_3layer_mlp, parallel_multiedge_bundle, nested_shallow_enc_dec,
  tl_mlp_3layer, mixed_width_labels

Run each (engine, graph) pair across seeds 42-71 (30 seeds) and write the
positions to the cache. The cache is at
`eval_output/competitor_cache/` (or wherever the existing cache lives;
inspect `dagua/eval/pipeline_io.py` and `scripts/algo_fidelity_cross.py`
for the format).

You may write a helper script `scripts/regen_ogdf_multiseed_cache.py` that:
- For each (engine, graph) pair, calls the OGDF runner with seeds 42-71
- Saves positions in the cache format the live_compare expects
- Idempotent (skip if entry exists)

If unsure of cache format, look at how `graphviz_seeded_cache` is laid out
(eval_output/algo_fidelity/round_19/graphviz_seeded_cache_60/) — same shape
should work.

### Phase D: Re-run live_compare for OGDF families

After cache populated:
```bash
for fam in fmmm gem pivot_mds maxent_stress stress_maj; do
  case $fam in
    fmmm) dagua=classic_fmmm; target=ogdf_fmmm ;;
    gem)  dagua=classic_gem;  target=ogdf_gem ;;
    pivot_mds) dagua=classic_pivot_mds; target=ogdf_pivot_mds ;;
    maxent_stress) dagua=classic_maxent_stress; target=ogdf_stress ;;
    stress_maj) dagua=classic_stress_maj; target=ogdf_stress ;;
  esac
  python scripts/algo_fidelity_live_compare.py "$dagua" "$target" \
      --seeds 30 \
      --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
      --output-dir "eval_output/algo_fidelity/round_28/ogdf_$fam"
done
```

Compare to Round 26 results — TOST verdicts should now actually exist
instead of `not_tested`.

## Scope

- DO NOT TOUCH render/styles, cluster sprint files
- Stage commits with explicit `git add`. NO `git add -A`.
- Commit format: `feat(fidelity): round 28 ogdf -- <terse>`
- Multiple commits OK (build infra, runner change, cache regen script, etc.)

## Output

Per-round SUMMARY at `eval_output/algo_fidelity/round_28/ogdf/SUMMARY.md`
covering:
- Did OGDF build succeed? Where are libs installed?
- Did runner rebuild succeed? Which algos now honor seed?
- Is the multi-seed cache populated? How many (engine, graph, seed) entries?
- Per-OGDF-family TOST verdicts before vs after multi-seed cache
- Any blockers / residuals

</task>

<completeness_contract>
Phase A-D in order. If Phase A fails, mark blocked and write SUMMARY explaining
why. Otherwise drive through to Phase D and surface multi-seed TOST verdicts.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. If any
phase fails, document the blocker and continue with what's possible.
</default_follow_through_policy>

<action_safety>
Build artifacts go to ~/.local/ (user-space, no sudo). Do NOT run sudo or
modify system paths. Do NOT install to /usr/local/.
</action_safety>
