<task>
Build a PERMANENT, reproducible, logging-only-instrumented Graphviz 7.0.5 and PROVE it is
veridical (its layout output is bit-for-bit identical to stock graphviz 7.0.5 -- the
instrumentation only ADDS logging, never changes behavior).

## Context
dagua (/home/jtaylor/projects/dagua) is reverse-engineering graphviz's force layouts
(fdp/neato/sfdp) to bit-exactly reproduce them in pure Python/torch. A prior sprint built an
instrumented graphviz in /tmp (now wiped) that printed graphviz's internal per-iteration state
so the dagua port could be matched step-by-step. We need that capability back, PERMANENTLY and
reproducibly, and we need to PROVE the instrumentation doesn't change graphviz's output.

The system graphviz is conda's at ~/anaconda3/envs/py311/bin/dot, version 7.0.5 (20221231.0122).

## What the prior instrumentation logged (from eval_output/algo_fidelity/round_48 + round_53 notes)
- lib/fdpgen/xlayout.c (~:125, :507): per-iteration "XLAYOUT" rows -- overlap count, bbox,
  iteration cnt, spring constant K, temperature, node/edge counts.
- lib/fdpgen/layout.c (~:67, :82): "FINALCC"/"FINALCC_COMPONENT" rows around finalCC() (component packing).
- lib/common/utils.c (~:774, :834): "COMPUTE_BB" per-component bbox rows.
- a per-iteration node-POSITION trace ("STEP" rows) at %.17g full float64 precision.
(Line numbers approximate -- find the right spots in graphviz 7.0.5 source.)

## Steps
1. Get graphviz 7.0.5 source (download the official 7.0.5 release tarball from gitlab.com/
   graphviz/graphviz or the graphviz.org archive; verify version). Put source under
   ~/tools/graphviz-7.0.5-src/.
2. Add LOGGING-ONLY instrumentation (printf/fprintf to stderr or a GV_TRACE_FILE env-var path)
   at the locations above, emitting fdp/neato/sfdp internal state per iteration at %.17g.
   CRITICAL: logging only -- do NOT alter any layout math, control flow, RNG, or data. Guard
   the logging behind an env var (e.g. only emit if GV_TRACE=1) so default runs are silent/fast.
3. Build with cmake to a PERMANENT prefix ~/tools/graphviz-7.0.5-instr/ (bin/dot etc.).
4. PROVE VERIDICAL: on ~6 small graphs (path8, star8, grid3x3, cycle6, complete5, tree7) at
   seeds 1,2,3, run BOTH ~/tools/graphviz-7.0.5-instr/bin/dot AND the stock conda dot
   (~/anaconda3/envs/py311/bin/dot) with fdp/neato/sfdp, `-Gseed=N -Gstart=N`, output positions
   (-Tplain or json), GV_TRACE OFF for the instrumented one. Assert the position outputs are
   IDENTICAL (bit-for-bit, or RMSD 0) between instrumented and stock. If they differ, the
   instrumentation changed behavior -> FIX until identical. Report the comparison.
5. Write a reproducible build script `scripts/rng_match/build_instrumented_graphviz.sh`
   (download -> patch -> cmake -> install -> veridical-check) and a short README documenting
   the instrumentation points + the GV_TRACE usage + the veridical-proof result.
</task>

<constraints>
- Instrumentation = ADDITIVE LOGGING ONLY. The instrumented binary's LAYOUT OUTPUT must be
  bit-identical to stock 7.0.5. This is the whole point (proving the instrumented build is a
  veridical reference). If you cannot keep it logging-only, STOP and report.
- Permanent location ~/tools/ (NOT /tmp). Build artifacts NOT committed to git (too big);
  ONLY the build script + README + the instrumentation patch (as a .patch file under
  scripts/rng_match/) are committed.
- Do not modify the dagua python codebase in this task. This is purely the graphviz tool.
- If graphviz 7.0.5 won't build cleanly in this environment, document the blocker precisely
  (missing deps, cmake errors) and stop -- do not hack around it silently.
</constraints>

<verification>
- `~/tools/graphviz-7.0.5-instr/bin/dot -V` reports 7.0.5.
- Veridical proof: instrumented-vs-stock position outputs identical on the 6 small graphs x
  3 seeds x {fdp,neato,sfdp}. Report the max difference (must be 0 / exact).
- `GV_TRACE=1 ~/tools/graphviz-7.0.5-instr/bin/dot -Kfdp -Gseed=1 path8.dot` emits trace rows
  with %.17g positions; default (GV_TRACE unset) emits none.
</verification>

<output>
- ~/tools/graphviz-7.0.5-instr/ (built binary)
- scripts/rng_match/build_instrumented_graphviz.sh (reproducible)
- scripts/rng_match/graphviz_7.0.5_instrumentation.patch
- scripts/rng_match/INSTRUMENTED_GRAPHVIZ_README.md (instrumentation points, GV_TRACE usage,
  veridical-proof numbers)
- Write the veridical-proof result (instrumented==stock, max diff) into the README AND print it.
Commit the script + patch + README (NOT the build dir) to branch develop.
</output>

<default_follow_through_policy>
Proceed autonomously. The one hard stop: if you cannot make the instrumented build's output
bit-identical to stock 7.0.5 (i.e. can't keep it logging-only) or graphviz won't build,
STOP and report precisely -- do not fake the veridical proof.
</default_follow_through_policy>
