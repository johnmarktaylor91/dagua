<task>
Install OGDF from source (user-space, no sudo) and rebuild the dagua OGDF runner so it honors
matched parameters, unblocking bit-exact comparison for the OGDF engines (gem, fmmm,
maxent_stress, stress_maj, pivot_mds).

Project: /home/jtaylor/projects/dagua.

## Background
dagua's OGDF reference adapter shells out to a compiled runner `scripts/ogdf_runner` (built
from scripts/ogdf_runner.cpp). The runner currently IGNORES per-variant params (e.g. GEM
rounds), so the benchmark compared dagua@100-rounds vs OGDF@default-30000-rounds -> spurious
divergence. A prior codex already UPDATED scripts/ogdf_runner.cpp to accept+apply those params,
but could not rebuild because OGDF headers are not installed:
  `fatal error: ogdf/basic/Graph.h: No such file or directory`
A diagnostic confirmed: dagua's GEM is BIT-EXACT to OGDF at matched rounds (100 rounds=3.86e-13,
500=3.93e-08); only the runner-rebuild is blocking it.

## Steps
1. Determine which OGDF version the existing runner/build targets (check scripts/ogdf_runner.cpp
   includes, any existing build script, CMakeLists, or notes). Use that version; if unknown, use
   a recent stable OGDF release from the OFFICIAL repo (https://github.com/ogdf/ogdf, a tagged
   release -- NOT random forks).
2. Clone + cmake-build OGDF from source to a PERMANENT prefix ~/tools/ogdf/ (headers + lib).
   User-space only -- no sudo. (OGDF is a self-contained C++ lib; static build is fine.)
3. Rebuild scripts/ogdf_runner against ~/tools/ogdf (update its build script / cmake to point at
   the OGDF headers+lib). Confirm the runner binary builds and runs.
4. VERIFY the rebuilt runner honors matched params:
   `export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:$LD_LIBRARY_PATH`
   `python scripts/rng_match/check_engine.py classic_gem_iters100` -- should now be BIT-EXACT
   (~1e-13, since dagua's GEM matches OGDF at matched rounds). Also check classic_gem_iters500
   and classic_fmmm_steps100. Report before/after.
5. Write a reproducible build script `scripts/rng_match/build_ogdf_runner.sh` (clone -> build OGDF
   -> rebuild runner -> verify) + a short README note (OGDF version, build location, the matched-
   params verification numbers).
</task>

<constraints>
- User-space install only (~/tools/ogdf, no sudo, no system package manager). If a true sudo/
  system dep is unavoidable, STOP and report exactly what's needed (do NOT attempt sudo).
- Official OGDF source only (github.com/ogdf/ogdf tagged release). No unaudited forks, no curl|bash.
- Do NOT edit dagua/layout/ops/ (port codexes' territory) or dagua/eval/variants.py. You MAY edit
  scripts/ogdf_runner.cpp (only if its param-handling needs finishing) + its build script.
- Build artifacts (~/tools/ogdf, the runner binary) are NOT committed; the build SCRIPT + README are.
- Do NOT commit (CC commits after verifying).
</constraints>

<verification>
- ~/tools/ogdf/ has OGDF headers (ogdf/basic/Graph.h) + lib.
- scripts/ogdf_runner rebuilt + runs.
- check_engine.py classic_gem_iters100 now < 1e-7 (ideally ~1e-13). Report the number.
</verification>

<default_follow_through_policy>
Proceed autonomously, user-space. Hard stop only if OGDF genuinely cannot build without root/
system packages -- then report the exact missing system dep so JMT can decide.
</default_follow_through_policy>
