# Contributing

This project is still early, so the main goal is disciplined iteration rather than heavy process.

## Working Principles

- Preserve the distinction between:
  - placement quality
  - routing quality
  - rendering quality
- Do not hide weak visual decisions behind nice language.
- Prefer small, validated commits over sprawling unverified change batches.
- Generated docs/artifacts should be rebuildable from scripts in the repo.

## Recommended Workflow

Read first:
- `docs/DEVELOPER_OVERVIEW.md`
- `docs/ITERATION_WORKFLOW.md`
- `docs/MAINTENANCE_CHECKLIST.md`
- `docs/STATUS.md`

For visual iteration:
- `docs/MONEY_GRAPHS.md`
- `eval_output/visual_review_session/`
- `docs/VISUAL_RESET_BRIEF.md`

For placement iteration:
- `docs/BENCHMARK_ARTIFACT_GUIDE.md`
- `eval_output/report/placement_summary.md`
- `eval_output/report/layout_similarity.md`

For scale work:
- `docs/OPEN_ISSUES.md`
- `scripts/bench_large.py`
- `/tmp/dagua-bench-1b.log`

## Handy Commands

```bash
make benchmark-status
make placement-sprint
make visual-session
make visual-audit
make glossary
make gallery
make explainer
```

## Typing And Quality

- Prefer explicit types on orchestration code, CLI code, benchmarks, and docs tooling.
- Widen strict mypy coverage gradually and honestly.
- Do not claim strictness that the codebase does not actually satisfy.

## Artifacts

Human-authored specs:
- YAML by default

Machine/generated artifacts:
- JSON by default

Large generated outputs should usually live under:
- `eval_output/`
- `docs/<generated-surface>/`

## Benchmarks

- Standard suite is persistent and resumable.
- Rare suite is manual and should not be rerun casually.
- Reuse cached non-Dagua competitors whenever possible during iteration.
- Treat the latest completed standard run as the active placement baseline unless
  a newer frozen baseline is explicitly designated.
