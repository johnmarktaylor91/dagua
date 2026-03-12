# Dagua Docs

Public documentation surfaces, roughly in the order a new user or reviewer is likely to need them.

## If You Need One Thing Fast

- Understand the codebase end to end:
  - `docs/DEVELOPER_OVERVIEW.md`
- Understand current project reality:
  - `docs/STATUS.md`
- Iterate on node placement:
  - `docs/ITERATION_WORKFLOW.md`
  - `docs/BENCHMARK_ARTIFACT_GUIDE.md`
- Iterate on visuals:
  - `docs/VISUAL_RESET_BRIEF.md`
  - `docs/MONEY_GRAPHS.md`
  - `docs/COMMAND_CHEATSHEET.md`
- Understand what competitors are doing downstream of placement:
  - `docs/COMPETITOR_GEOMETRY_MEMO.md`

## Start Here

- Tutorial walkthrough: `docs/tutorial_walkthrough.ipynb`
- Human-facing reference manual: `docs/glossary/dagua_glossary.pdf`
- Public agent guide: `docs/LLM_TUTORIAL.md`
- Developer-facing codebase map: `docs/DEVELOPER_OVERVIEW.md`
- Algorithm explainer: `docs/how_dagua_works.md`
- Competitor geometry memo: `docs/COMPETITOR_GEOMETRY_MEMO.md`

## Visuals

- Showcase gallery: `docs/gallery/README.md`
- Video resources and shot planning: `docs/video/README.md`
- Generated algorithm visuals: `docs/how_dagua_works/`

## Iteration And Maintenance

- Current project status: `docs/STATUS.md`
- Maintenance checklist: `docs/MAINTENANCE_CHECKLIST.md`
- Iteration workflow: `docs/ITERATION_WORKFLOW.md`
- Money graphs shortlist: `docs/MONEY_GRAPHS.md`
- Benchmark artifact guide: `docs/BENCHMARK_ARTIFACT_GUIDE.md`
- Benchmark failure taxonomy: `docs/BENCHMARK_FAILURE_TAXONOMY.md`
- Layout vs render reference: `docs/LAYOUT_VS_RENDER_REFERENCE.md`
- Command cheat sheet: `docs/COMMAND_CHEATSHEET.md`
- Baseline playbook: `docs/BASELINE_PLAYBOOK.md`
- Visual reset brief: `docs/VISUAL_RESET_BRIEF.md`
- Open issues ledger: `docs/OPEN_ISSUES.md`
- Contributing guide: `CONTRIBUTING.md`
- Visual-audit workflow output root: `eval_output/visual_audit/`
- Numbered collaborative visual review root: `eval_output/visual_review_session/`
- Benchmark report output root: `eval_output/report/`

## Working Sets

- Placement-first loop:
  - `docs/STATUS.md`
  - `docs/OPEN_ISSUES.md`
  - `docs/BENCHMARK_ARTIFACT_GUIDE.md`
  - `eval_output/report/placement_summary.md`
  - `eval_output/report/placement_tuning.md`
  - `eval_output/report/layout_similarity.md`
- Visual-reset loop:
  - `docs/VISUAL_RESET_BRIEF.md`
  - `docs/MONEY_GRAPHS.md`
  - `eval_output/visual_audit/`
  - `eval_output/visual_review_session/`
- Scale/billion loop:
  - `scripts/bench_large.py`
  - `/tmp/dagua-bench-1b.log`
  - `docs/OPEN_ISSUES.md`

## Regeneration Commands

```bash
make benchmark-status
make placement-sprint
dagua placement-tune --output-dir /home/jtaylor/projects/dagua/eval_output/report
make visual-session
make visual-audit
python scripts/build_glossary.py --output-dir docs/glossary
python scripts/build_gallery.py --output-dir docs/gallery
python scripts/build_how_dagua_works.py
python scripts/build_visual_audit.py --output-dir eval_output/visual_audit
```
