<task>
R32 INFRA: diagnose why pre-commit hooks rolled back drl + tsnet R31 commits.

## Evidence

During R31 commit phase:
- `git commit` for drl (commit message `fix(layout): round 31 drl -- preset + init + jump sign + edge cut`) ran pre-commit hooks
- Pre-commit reported `Restored changes from /home/jtaylor/.cache/pre-commit/patch1779664547-579381`
- The patch file exists at `/home/jtaylor/.cache/pre-commit/patch1779664547-579381` (size 33659 bytes)
- The actual commit was rolled back (HEAD did not advance)
- Same pattern for tsnet

The umap/lgl/graphopt commits ALSO triggered pre-commit hooks but went through. Difference unknown.

## Your job

1. Diagnose: read `.pre-commit-config.yaml` and figure out which hook produced the rollback. Likely candidates:
   - ruff-format (long lines, trailing whitespace, etc)
   - ruff check
   - detect-secrets
   - end-of-file-fixer / trailing-whitespace

2. Inspect the rolled-back patch contents at `/home/jtaylor/.cache/pre-commit/patch1779664547-579381` to see what auto-fix the hook tried to apply

3. Fix the underlying issue (e.g., a long line that needs split, or auto-formattable code that conflicts), OR add an exclusion if the files have un-fixable patterns

4. As a verification: re-apply the patch content from `/home/jtaylor/.cache/pre-commit/patch1779664547-579381` to the working tree (use `git apply -R` if needed) and attempt commit again normally. If it lands clean, recover drl/tsnet.

## Output

`eval_output/algo_fidelity/round_32/hook_diagnose/REPORT.md` with:
- Root cause of the R31 rollback
- Code changes needed (lint fix, formatting, exclusion)
- If you successfully recovered drl/tsnet content from the patch and committed cleanly, note the commits.

## Scope
- DO TOUCH `.pre-commit-config.yaml` if needed
- DO TOUCH `dagua/layout/ops/drl.py`, `dagua/layout/ops/tsnet.py` and their pipelines IF the rollback was content-related
- Explicit git add. Commits should land via the normal hook path.
</task>

<completeness_contract>
Diagnose + document. If recoverable, recover drl + tsnet commits.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
