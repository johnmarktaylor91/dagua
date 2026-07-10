# MANIFEST: Rome-Lib holdout sample

## Source
- URL: https://graphdrawing.unipg.it/data/rome-graphml.tgz
- Note: `graphdrawing.org` now 301-redirects (over plain HTTP) to
  `graphdrawing.unipg.it`, which serves a valid SSL certificate. The prior
  fetch failure logged in `eval_output/stdcorpora/README.md` (2026-07-07,
  "SSL hostname mismatch for graphdrawing.org") was against the old domain;
  the archive itself was never broken.
- Archive contains 11,534 `.graphml` files (Y-Files XML export), 10-100
  nodes per graph per the source page's stated range. This is the
  "Undirected graphs (from the Rome graphs)" GraphML release, generated
  from the original GDToolkit/Rome test suite.

## Retrieval date
2026-07-08

## Selection rule
Deterministic stride sample: sort all `.graphml` filenames lexicographically,
take every 76th file (`files[::76]`), starting at index 0. This yields 152
files (~150 target) spanning the full size range present in the archive.

## File count
152 files, all `.graphml`.

## Size range
10-100 nodes (min 10, max 100, mean 52.3). Matches the archive's documented
10-100 node coverage.

## Directedness
All 152 sample graphs load as **undirected** (verified against the actual
loader, not just the manifest metadata) -- consistent with `edgedefault="undirected"`
present on every sampled file's `<graph>` element.

## Parse verification
100% parse success (152/152) against `scripts/r79_stdcorpora_eval.py`'s
`load_graphml_file` loader (added this run -- see "Loader findings" below).

## License note
No standalone license file ships with the archive. The Rome Graphs /
GDToolkit test suite has been distributed as a free academic
graph-drawing benchmark by the Graph Drawing symposium community since
the 1990s (see https://graphdrawing.unipg.it/data.html); treated here as
research/benchmark use consistent with its long-standing public
distribution. No redistribution restriction is stated on the source page.

## Loader findings
See `.project-context/research/r79_native/stdcorpora/LOADER_FINDINGS.md`.
