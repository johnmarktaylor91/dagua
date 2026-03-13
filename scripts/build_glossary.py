#!/usr/bin/env python
"""Build the Dagua glossary/reference PDF and supporting visuals."""

from __future__ import annotations

import argparse

from dagua.reference_glossary import build_glossary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/glossary")
    parser.add_argument("--no-pdf", action="store_true", help="Skip pdflatex compilation.")
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=30,
        help="Layout steps used when generating explanatory visuals.",
    )
    args = parser.parse_args()

    result = build_glossary(
        output_dir=args.output_dir,
        compile_pdf=not args.no_pdf,
        sample_steps=args.sample_steps,
    )
    print(result.tex_path)
    if result.pdf_path:
        print(result.pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
