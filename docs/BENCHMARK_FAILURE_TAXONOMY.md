# Benchmark Failure Taxonomy

Use these categories when reading benchmark outputs, debugging wrappers, or summarizing results.

## Status Classes

- `OK`
  - layout succeeded and metrics are meaningful
- `SKIPPED`
  - intentionally not run
  - usually exceeds known tool scale ceiling
- `FAILED`
  - attempted, but did not produce a usable result

## Failure Subtypes

These should be used in logs, notes, and future report improvements.

- `FAILED_TIMEOUT`
  - process exceeded the allowed time budget
- `FAILED_TOOL`
  - competitor executable/runtime crashed or exited nonzero
- `FAILED_PARSE`
  - tool ran but output could not be parsed into positions
- `FAILED_WRAPPER`
  - our adapter code called the tool incorrectly or mishandled its result
- `FAILED_ALGO`
  - tool completed but produced unusable layout geometry
- `FAILED_RESOURCE`
  - OOM / VRAM exhaustion / hard system kill
- `FAILED_ENV`
  - missing dependency, binary, runtime, or incompatible local install

## Why This Matters

These are not equivalent.

Examples:
- A wrapper mismatch is not evidence the layout engine is weak.
- A timeout at 50K is meaningful algorithmic evidence.
- A parse failure is mainly a tooling problem.

The benchmark story gets much more honest once these are separated.
