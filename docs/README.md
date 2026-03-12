# Dagua Docs

Public documentation surfaces, roughly in the order a new user or reviewer is likely to need them.

## Start Here

- Tutorial walkthrough: `docs/tutorial_walkthrough.ipynb`
- Human-facing reference manual: `docs/glossary/dagua_glossary.pdf`
- Public agent guide: `docs/LLM_TUTORIAL.md`
- Developer-facing codebase map: `docs/DEVELOPER_OVERVIEW.md`
- Algorithm explainer: `docs/how_dagua_works.md`

## Visuals

- Showcase gallery: `docs/gallery/README.md`
- Video resources and shot planning: `docs/video/README.md`
- Generated algorithm visuals: `docs/how_dagua_works/`

## Iteration And Maintenance

- Maintenance checklist: `docs/MAINTENANCE_CHECKLIST.md`
- Command cheat sheet: `docs/COMMAND_CHEATSHEET.md`
- Baseline playbook: `docs/BASELINE_PLAYBOOK.md`
- Visual reset brief: `docs/VISUAL_RESET_BRIEF.md`
- Open issues ledger: `docs/OPEN_ISSUES.md`
- Visual-audit workflow output root: `eval_output/visual_audit/`
- Benchmark report output root: `eval_output/report/`

## Regeneration Commands

```bash
python scripts/build_glossary.py --output-dir docs/glossary
python scripts/build_gallery.py --output-dir docs/gallery
python scripts/build_how_dagua_works.py
python scripts/build_visual_audit.py --output-dir eval_output/visual_audit
```
