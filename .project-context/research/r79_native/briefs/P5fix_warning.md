<task>
Worktree: /home/jtaylor/.claude/worktrees/dagua-native-p1 (currently on branch
r79/p5-clusters, HEAD 9eb3d2c). Python: .venv/bin/python.

CONTEXT: The r79 sprint's P5 round added recursive cluster placement for native_stress
(good, keep) but ALSO removed the "cluster_aware=True is not yet supported for
algorithm='dagua_native'; falling back to legacy flat placement" warning for ALL native
clustered graphs. Problem: LAYERED/DAG clustered graphs are STILL laid out flat (recursive
layered clustering regressed the sweep 49/9/35 and was correctly left disabled), so removing
the warning for them silences a message that is still TRUE -> a silent quality gap. Verified:
on transformer_layer (a layered clustered graph), cluster_aware=True now emits NO warning
while behavior is unchanged (still flat).

GOAL (small, surgical): make the warning HONEST. Keep the warning (or an equivalent
info/warn) for graphs that STILL fall back to flat native placement (the layered/DAG
clustered path), and suppress it ONLY for graphs that now get genuine recursive cluster
placement (the native_stress / declared-undirected path). Do not change any layout geometry
-- this is purely restoring an accurate warning on the path that still falls back.

APPROACH: find where P5 removed/short-circuited the warning (engine.py ~1079-1092 fallback
path, or wherever the native cluster-aware dispatch decides recursive-vs-flat). Gate the
warning on "did this graph actually get recursive cluster placement?" -- emit it when the
native path fell back to flat, suppress it when the recursive driver handled the clusters.
Message can be updated to be accurate (e.g. "recursive cluster placement not yet available
for layered/DAG clusters with algorithm='dagua_native'; using flat placement with cluster
losses").

VERIFY:
- transformer_layer (layered clustered): warning FIRES again.
- an undirected clustered graph routed through native_stress recursion: warning does NOT fire.
- Add/adjust a warnings-capture test asserting both.
- Containment unchanged (0 violations) and default sweep unaffected: run
  `.venv/bin/python -m pytest tests/test_layout/test_cluster_driver.py -q` (scoped -- do NOT
  run the full suite; see .project-context/research/r79_native/KNOWN_RED_TESTS.md).
- ruff check on touched files.

CONSTRAINTS: geometry/positions must NOT change (prove by spot-checking 2 clustered graphs'
positions identical before/after your change at seed 42). ASCII; conventional commit; no AI
attribution; COMMIT REQUIRED on r79/p5-clusters. Append a short note to
.project-context/research/r79_native/P5_EVIDENCE.md ("## Warning honesty fix").
</task>

<operational_rules>
1. Any assistant message WITHOUT a tool call TERMINATES your session; final no-tool-call
   message = report, only after commit verified. 2. stdin closed. 3. Scoped test gate only,
   NEVER bare pytest -x. 4. Long cmds in one exec call with timeout.
</operational_rules>

<completeness_contract>
Done = warning fires for flat-fallback layered clusters, suppressed for recursive
native_stress clusters, test asserts both, positions unchanged, committed on r79/p5-clusters.
</completeness_contract>

FINAL REPORT: what gated the warning, the two-case verification, position-unchanged proof,
commit sha.
