.PHONY: benchmark-status placement-sprint placement-tune visual-session visual-audit glossary gallery explainer artifact-index

benchmark-status:
	python -m dagua.eval.benchmark --suite standard --output-dir eval_output --status-only

placement-sprint:
	python -m dagua.cli placement-sprint --output-dir eval_output

placement-tune:
	python -m dagua.cli placement-tune --output-dir eval_output/report

artifact-index:
	python -m dagua.cli placement-sprint --output-dir eval_output

visual-session:
	python -m dagua.cli visual-session-build --output-dir eval_output/visual_review_session

visual-audit:
	python -m dagua.cli visual-audit-build --output-dir eval_output/visual_audit

glossary:
	python scripts/build_glossary.py --output-dir docs/glossary

gallery:
	python scripts/build_gallery.py --output-dir docs/gallery

explainer:
	python scripts/build_how_dagua_works.py
