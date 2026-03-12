.PHONY: benchmark-status placement-sprint visual-session visual-audit glossary gallery explainer

benchmark-status:
	python -m dagua.eval.benchmark --suite standard --output-dir eval_output --status-only

placement-sprint:
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
