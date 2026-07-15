# P6A Standard-Corpora Heldout Harness Evidence

Status: complete.

## Scope

- Build reusable heldout harness for Rome/North/SuiteSparse dropped-in corpus files.
- Preserve holdout discipline: no `dagua/layout/` edits and no tuning.
- Keep acquisition best-effort and bounded.

## Implementation

- Added `scripts/r79_stdcorpora_eval.py`.
- Added `scripts/fetch_stdcorpora.sh`.
- Added `tests/test_stdcorpora_eval.py`.
- Wrote acquisition fallback documentation to `eval_output/stdcorpora/README.md`.

## Fetch Outcome

Disk before acquisition:

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       458G  412G   23G  95% /
```

The Rome archive URL `https://graphdrawing.org/data/rome/rome.tar.gz` failed twice with
`curl: (60) SSL: no alternative certificate subject name matches target host name
'graphdrawing.org'`. Per the task contract, acquisition stopped and no synthetic corpora were
created.

## Test Evidence

```text
.venv/bin/python -m ruff check scripts/r79_stdcorpora_eval.py tests/test_stdcorpora_eval.py --fix
All checks passed!

.venv/bin/python -m pytest tests/test_stdcorpora_eval.py -q
....                                                                     [100%]
4 passed, 2 warnings in 6.89s
```

## Holdout Discipline

- No edits under `dagua/layout/`.
- No route flip, tuning, or native layout implementation changes.
