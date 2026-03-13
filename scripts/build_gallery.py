#!/usr/bin/env python
"""Build the Dagua user-facing showcase gallery."""

from __future__ import annotations

import argparse

from dagua.showcase_gallery import build_showcase_gallery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/gallery")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-animations", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--animation-steps", type=int, default=None)
    args = parser.parse_args()

    result = build_showcase_gallery(
        output_dir=args.output_dir,
        include_animations=not args.no_animations,
        limit=args.limit,
        sample_steps=args.sample_steps,
        animation_steps=args.animation_steps,
    )
    print(result.readme_path)
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
