# R32 Hook Rollback Diagnosis

## Root Cause

The R31 rollback was caused by pre-commit's unstaged-change stash colliding with
hook auto-fixes, not by `detect-secrets` or a permanent repository exclusion
problem.

The active hooks in `.pre-commit-config.yaml` are:

- `trailing-whitespace`
- `end-of-file-fixer`
- `check-yaml`
- `check-added-large-files`
- `detect-secrets`
- `ruff-format`
- `ruff --fix`

The saved file
`/home/jtaylor/.cache/pre-commit/patch1779664547-579381` is pre-commit's stash
patch for local unstaged changes. It is not the hook's own auto-fix patch. That
patch contains unstaged DrL edits plus unrelated dirty files:

- `dagua/eval/competitors/classic_competitor.py`
- `dagua/layout/ops/drl.py`
- `dagua/layout/ops/pipelines/drl.py`
- `eval_output/perceptual_divergence_report.md`
- `scripts/fidelity_analysis.py`
- `tests/test_pipeline_drl.py`

This matches the standard pre-commit failure mode where:

1. `git commit` starts with both staged and unstaged edits present.
2. pre-commit stashes the unstaged edits into
   `/home/jtaylor/.cache/pre-commit/patch1779664547-579381`.
3. an auto-fixing hook modifies staged files.
4. pre-commit cannot re-apply the unstaged stash cleanly because it overlaps the
   hook-modified files.
5. pre-commit rolls back the hook edits and restores the stash, so the commit
   does not land.

The concrete hook observed reproducing the same rollback pattern while writing
this report was `end-of-file-fixer`: the first normal commit attempt stashed
unstaged files to `/home/jtaylor/.cache/pre-commit/patch1779678622-728508`,
then `end-of-file-fixer` modified the staged report and stopped the commit. That
identifies the hook family responsible for at least this repository's rollback
mechanism: an auto-fixing hook rewrote staged content while unstaged work was
stashed. For the R31 Python files, `ruff-format` or `ruff --fix` could also have
triggered the same mechanism if any partially staged DrL/tsNET hunk needed
formatting, but the saved patch itself does not record the hook's internal diff.

## Code Changes Needed

No `.pre-commit-config.yaml` exclusion is needed. The underlying fix is to let
the auto-fixing hook update the file, then explicitly re-stage that update and
commit again. For algorithm files, also avoid committing partially staged files
while auto-fixing hooks are enabled.

Before committing DrL or tsNET changes, run:

```bash
ruff format dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/ops/pipelines/tsnet.py tests/test_pipeline_tsnet.py
ruff check --fix dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/ops/pipelines/tsnet.py tests/test_pipeline_tsnet.py
git add dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/ops/pipelines/tsnet.py tests/test_pipeline_tsnet.py
git commit
```

If pre-commit reports that a hook modified files, run `git add` for exactly
those files and retry `git commit`. That avoids pre-commit having to reconcile
overlapping unstaged edits with formatter/linter or EOF rewrites.

## Patch Recovery

The DrL and tsNET implementation/test content from the rollback sequence has
already landed in `HEAD`, but not as standalone DrL/tsNET commits. It is present
inside:

- `a6fd45c49879b45bc3f97fcfcfa44926a42584d8`
  `fix(layout): round 31 umap -- per-axis scale + smooth_knn + multi-comp + arpack`

That commit includes:

- `dagua/layout/ops/drl.py`
- `dagua/layout/ops/pipelines/drl.py`
- `tests/test_pipeline_drl.py`
- `dagua/layout/ops/tsnet.py`
- `dagua/layout/ops/pipelines/tsnet.py`
- `tests/test_pipeline_tsnet.py`

The rollback patch cannot be applied cleanly to the current working tree because
its DrL hunks are already present in `HEAD`, while its remaining hunks are still
local dirty changes in:

- `dagua/eval/competitors/classic_competitor.py`
- `eval_output/perceptual_divergence_report.md`
- `scripts/fidelity_analysis.py`

No duplicate DrL/tsNET recovery commit was created.

## Verification

Commands run:

```text
git apply --stat /home/jtaylor/.cache/pre-commit/patch1779664547-579381
git apply --check /home/jtaylor/.cache/pre-commit/patch1779664547-579381
git show --name-only --oneline --decorate a6fd45c
git show --check --format=short a6fd45c -- dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/ops/pipelines/tsnet.py tests/test_pipeline_tsnet.py
ruff format --check dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/ops/pipelines/tsnet.py tests/test_pipeline_tsnet.py
ruff check dagua/layout/ops/drl.py dagua/layout/ops/pipelines/drl.py tests/test_pipeline_drl.py dagua/layout/ops/tsnet.py dagua/layout/ops/pipelines/tsnet.py tests/test_pipeline_tsnet.py
git commit -m "docs(fidelity): diagnose round 31 hook rollback"
```

Results:

```text
6 files already formatted
All checks passed!

First report commit attempt:
fix end of files.........................................................Failed
- hook id: end-of-file-fixer
- files were modified by this hook
```

`git apply --check` failed because the patch is stale relative to current
`HEAD`; this is expected after the recovered hunks landed in `a6fd45c`.
