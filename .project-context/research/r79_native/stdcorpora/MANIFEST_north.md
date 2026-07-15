# MANIFEST: North (AT&T) DAG holdout sample

## Source
- URL: https://graphdrawing.unipg.it/data/north-graphml.tgz
- Same domain-migration note as Rome (see `MANIFEST_rome.md`): the official
  `graphdrawing.org` host now redirects to `graphdrawing.unipg.it`, which has
  a valid cert. This is the "Directed graphs (from the AT&T graphs)" GraphML
  release generated from the original AT&T Graph Catalog, 10-100 nodes per
  graph per the source page.
- Archive contains 1,277 `.graphml` files.

## Retrieval date
2026-07-08

## Selection rule
Deterministic stride sample: sort all `.graphml` filenames lexicographically,
take every 12th file (`files[::12]`), starting at index 0. Yields 107 files
(~100 target) spanning the size range present in the archive.

## File count
107 files, all `.graphml`.

## Size range
10-96 nodes (min 10, max 96, mean 31.9).

## Directedness
All 107 sample graphs load as **directed** (verified against the actual
loader, not just filenames) -- see "Loader findings" below for why this
required a fix, not just a path-based assumption.

## Parse verification
100% parse success (107/107) against `scripts/r79_stdcorpora_eval.py`'s
`load_graphml_file` loader (added this run).

## License note
Same as Rome -- no standalone license file ships with the archive; treated
as research/benchmark use consistent with its long-standing public
distribution via the Graph Drawing symposium's data page.

## Loader findings
See `.project-context/research/r79_native/stdcorpora/LOADER_FINDINGS.md`.
This corpus specifically exercises the North-directedness fallback: the raw
GraphML `<graph>` elements omit `edgedefault` entirely, so NetworkX's
`is_directed()` reports `False` for every file. Without the fallback, all
107 North DAGs would have silently scored as undirected.
