<task>
Worktree: /home/jtaylor/.claude/worktrees/dagua-native-p4. FIRST:
`git checkout r79/native && git checkout -b r79/p6a-stdcorpora`. Python: .venv/bin/python.
Read first: r79-native_SUMMARY.md and r80_FOLLOWUP_PLAN.md (the "Standard-corpora HELDOUT
eval" section) at .project-context/research/r79_native/ (mirror at
/home/jtaylor/.claude/research/dagua/r79-native/).

GOAL: build the reusable HELDOUT evaluation harness for community-standard graph-drawing
corpora (Rome-Lib, North/AT&T DAGs, SuiteSparse samples). This is measurement infrastructure
for the follow-up sprint's honest generalization test. HOLDOUT DISCIPLINE IS THE POINT: this
task ONLY builds the harness and (optionally) runs a first measurement. It must NOT tune,
route-flip, or change ANY dagua/layout/ code. Zero edits under dagua/layout/.

DELIVERABLE 1 -- scripts/r79_stdcorpora_eval.py:
- Consumes a corpus directory of standard graph files (support: Rome/North .graph or GML,
  and SuiteSparse Matrix Market .mtx -> undirected graph from the sparsity pattern). Provide
  a small loader per format (reuse networkx read_gml / scipy.io.mmread if available;
  scipy/networkx are dev deps -- OK for the EVAL side, they are competitors/utilities, never
  imported by dagua/layout). Default corpus dir: eval_output/stdcorpora/ (gitignored; large).
- For each graph: lay out with dagua's native default (dagua.layout, seed 42) and with the
  frozen competitor engines already used by scripts/r79_baseline.py (graphviz_dot,
  graphviz_sfdp, graphviz_neato, elk_layered, dagre, nx_spring, igraph_kamada_kawai,
  igraph_sugiyama) via the existing dagua/eval/competitors adapters. Skip engines gracefully
  if unavailable (record the skip). Cap graph size at <= 2000 nodes for this first harness
  (Rome/North are small; filter SuiteSparse to small structural matrices).
- Score with the ESTABLISHED metric suite already in dagua/metrics.py: use composite_auto
  (undirected for these -- Rome/North/SuiteSparse are treated as undirected drawings unless a
  North DAG is explicitly directed) PLUS the individual normalized subterms, and REPORT
  sampled_stress + neighborhood_preservation + crossing_rate + edge_length_cv separately so
  the follow-up can compare against the human-validated proxy set.
- Output: eval_output/stdcorpora/results.json (per graph x engine: metrics + composite) and
  eval_output/stdcorpora/STDCORPORA.md (W/T/L of dagua-native vs best-external, per-corpus
  breakdown, tie band +/-0.5). Same integrity discipline as r79_baseline.py (validate rows).
- --dagua-only mode for cheap re-measurement against a frozen competitor snapshot, like
  r79_baseline.py.

DELIVERABLE 2 -- corpus acquisition (BEST EFFORT, MUST NOT BLOCK OR THRASH):
- Provide scripts/fetch_stdcorpora.sh (or a --fetch flag) that downloads a SMALL sample from
  OFFICIAL sources only: Rome-Lib and North DAGs from the graphdrawing.org / AT&T graph
  archive (or a well-known academic mirror), and a handful of small SuiteSparse structural
  matrices from sparse.tamu.edu (or via ssgetpy if installed). HARD LIMITS: total download
  < 200MB; check `df -h /` before and after and ABORT if < 10GB free; at most a few dozen
  graphs per corpus for this first pass. If a source is unreachable after 2 attempts, WRITE A
  CLEAR README in eval_output/stdcorpora/ documenting the expected file layout and the source
  URLs, and STOP -- do NOT synthesize fake "standard" graphs, do NOT thrash retrying.
- The harness (Deliverable 1) MUST work on any dropped-in corpus dir even if the fetch is
  skipped -- fetching is secondary; the reusable harness is the primary deliverable.

DELIVERABLE 3 -- if corpus acquisition succeeded, run one measurement pass and put the honest
numbers in STDCORPORA.md. If it did not, that is fine -- the harness + README is the deliverable.

TESTS: a unit test for each loader (tiny synthetic .graph/.gml/.mtx fixture -> correct
edge_index + node count) and a smoke test that the harness runs end-to-end on a 3-graph
synthetic mini-corpus. TEST GATE = ONLY your new test file + the loaders:
`.venv/bin/python -m pytest tests/test_stdcorpora_eval.py -q`. DO NOT run the full suite
(`pytest tests/ -x` whack-a-moles for hours on pre-existing stale failures -- see
.project-context/research/r79_native/KNOWN_RED_TESTS.md). ruff check on touched files only.

CONSTRAINTS: ZERO edits under dagua/layout/. ASCII only. Conventional commits, no AI
attribution. eval_output/stdcorpora/ data is gitignored (add to .gitignore if not already);
commit the script(s), tests, .gitignore entry, and STDCORPORA.md/README. COMMITS REQUIRED on
completion (orchestrator-git notes in AGENTS.md do NOT apply). Evidence/notes to
.project-context/research/r79_native/P6A_EVIDENCE.md.
</task>

<operational_rules>
1. Any assistant message WITHOUT a tool call TERMINATES your session; final no-tool-call
   message = your report, only after commits verified.
2. stdin closed; never wait for interactive input.
3. Long/network commands in ONE exec call with a bounded timeout; NEVER background-and-idle;
   NEVER retry a failing download more than twice.
4. Disk: `df -h /` before any download; abort if < 10GB free. ENOSPC -> stop, report.
5. NEVER run bare `pytest tests/ -x`. Scope the gate to your new test file.
</operational_rules>

<default_follow_through_policy>
Most reasonable low-risk interpretation; keep going; note choices. The holdout discipline
(no layout-code edits, no tuning) and the "harness works without fetch" contract are
non-negotiable. Partial (harness + README, no live corpus) is an acceptable honest outcome.
</default_follow_through_policy>

<completeness_contract>
Done = reusable scripts/r79_stdcorpora_eval.py + loaders + loader tests + end-to-end smoke on
a synthetic mini-corpus, all green; corpus fetch attempted with README fallback; if corpus
present, one measurement pass with honest W/T/L in STDCORPORA.md; committed on
r79/p6a-stdcorpora; ZERO dagua/layout/ edits.
</completeness_contract>

<verification_loop>
1) Loader unit tests green. 2) End-to-end smoke on synthetic mini-corpus green. 3) git diff
shows ZERO dagua/layout/ changes. 4) ruff clean on touched files. 5) git status clean, data
dir gitignored.
</verification_loop>

FINAL REPORT: what the harness does; corpus fetch outcome (fetched N graphs / documented
README); if measured, the honest std-corpora W/T/L; files added; commit shas.
