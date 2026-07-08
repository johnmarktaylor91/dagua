# r80-S3: Fetch standard holdout corpora (Rome / North / SuiteSparse)

## Context
The r79 native-algo sprint needs a HOLDOUT evaluation on standard graph-drawing corpora to
prove we did not overfit to our 108-graph iteration suite. The harness is already built and
green: scripts/r79_stdcorpora_eval.py + loaders (.graph/.gml/.mtx) on branch
r79/p6a-stdcorpora in worktree /home/jtaylor/.claude/worktrees/dagua-native-p4 (commit
6bbf9a2). The original fetch failed: graphdrawing.org has a broken SSL cert. Read the
README/fallback notes the harness left (grep for stdcorpora under scripts/ and
eval_output/ in that worktree).

## Task (data fetch + verify ONLY -- zero layout code changes, zero tuning)
Work in /home/jtaylor/.claude/worktrees/dagua-native-p4 (already on r79/p6a-stdcorpora).
Python: use /home/jtaylor/.claude/worktrees/dagua-native/.venv/bin/python but verify
`import dagua` resolves to the p4 worktree when run from p4; if not, `uv venv .venv &&
uv pip install -p .venv/bin/python -e ".[dev]"` in p4.

1. **Rome-Lib** (~11.5k small graphs; we need a SAMPLE, not all): find a working mirror --
   GitHub mirrors of Rome-Lib exist (search github for "rome-lib" / "rome graphs" GML/graphml);
   also try http (not https) on graphdrawing.org, and curl --insecure is acceptable ONLY to
   inspect; prefer a proper mirror for the actual data. Download and extract a deterministic
   sample: sort filenames, take every Nth to get ~150 graphs spanning the size range.
2. **North (AT&T) DAGs**: same approach, deterministic ~100-graph sample.
3. **SuiteSparse**: fetch ~15 SMALL symmetric matrices (N between 100 and 5000, nnz < 100k)
   via `ssgetpy` (pip install into the venv) or direct https://sparse.tamu.edu downloads.
   Deterministic selection: e.g. the smallest N>=100 undirected/symmetric matrices from
   well-known groups (HB, Pajek). Keep TOTAL download under 500MB; disk is tight (19GB free).
4. Drop everything under eval_output/stdcorpora/<corpus>/ in the p4 worktree, with a
   MANIFEST.md per corpus: source URL, retrieval date, selection rule, file count, license
   note.
5. **Verify loaders parse**: run the harness's loader tests + a dry parse of every fetched
   file with the project loaders; report parse success rate per corpus. If a loader chokes
   on a format variant, record it as a finding -- do NOT modify loader code unless it is a
   trivial (<10 line) format tolerance fix, in which case commit it separately with a test.
6. DO NOT run the full evaluation. DO NOT look at layout quality. This corpus is a holdout;
   fetching + parsing is the entire scope.

## Safety
- Downloads only from: github.com, sparse.tamu.edu, graphdrawing.org. No curl|bash, no
  executables, data files only.
- Watch disk before/after: `df -h /` -- abort downloads if free space would drop below 15GB.

## Output contract
- Committed MANIFEST.md files + any trivial loader-tolerance fixes on r79/p6a-stdcorpora.
  Data files themselves: check whether eval_output/ is gitignored (likely) -- if so they
  stay uncommitted; that is fine, the MANIFEST goes in
  .project-context/research/r79_native/ instead (which is tracked in that worktree).
- Final message: per-corpus inventory (count, size range, parse rate), where the files
  landed, any loader findings, disk used.
- ASCII only.
