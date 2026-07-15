# MANIFEST: SuiteSparse holdout sample

## Source
- Tool: `ssgetpy` (installed into the shared venv:
  `/home/jtaylor/.claude/worktrees/dagua-native/.venv`), which downloads
  from https://sparse.tamu.edu/ (SuiteSparse Matrix Collection).
- Format: Matrix Market (`.mtx`), coordinate/pattern.

## Retrieval date
2026-07-08

## Selection rule
1. `ssgetpy.search(group=<HB|Pajek>, rowbounds=(100,5000), colbounds=(100,5000),
   nzbounds=(1,100000), limit=200)`.
2. Filter to square matrices (`rows == cols`) with full pattern symmetry
   (`psym == 1.0`), i.e. genuinely undirected/symmetric adjacency structure.
3. Filter to `nnz > 2 * rows` -- excludes near-diagonal "mass matrix"
   variants (e.g. `bcsstm03`/`bcsstm04`/`bcsstm22`, HB's stiffness-matrix
   companions) that are technically pattern-symmetric but parse to ~0
   off-diagonal edges and are useless as layout inputs. This filter was
   added after an initial pass produced 3 zero-edge graphs -- see "Loader
   findings" below.
4. Sort by `(rows, group, name)` and take the first 15 -- the smallest
   N>=100 symmetric matrices with real graph structure from the HB and
   Pajek groups.

## File count
15 files, all `.mtx`.

## Selected matrices

| Group | Name | N (rows) | nnz (raw) | Parsed edges | Kind |
|---|---|---:|---:|---:|---|
| HB | nos4 | 100 | 594 | 247 | structural problem |
| Pajek | GD06_theory | 101 | 380 | 190 | undirected graph |
| HB | bcsstk03 | 112 | 640 | 264 | structural problem |
| Pajek | GD98_c | 112 | 336 | 168 | undirected graph |
| HB | bcspwr03 | 118 | 476 | 179 | power network problem |
| Pajek | Journals | 124 | 12068 | 5972 | undirected weighted graph |
| HB | bcsstk04 | 132 | 3648 | 1758 | structural problem |
| HB | bcsstk22 | 138 | 696 | 279 | structural problem |
| HB | can_144 | 144 | 1296 | 576 | structural problem |
| HB | lund_a | 147 | 2449 | 1151 | structural problem |
| HB | lund_b | 147 | 2441 | 1147 | structural problem |
| HB | bcsstk05 | 153 | 2423 | 1135 | structural problem |
| HB | can_161 | 161 | 1377 | 608 | structural problem |
| HB | dwt_162 | 162 | 1182 | 510 | structural problem |
| HB | can_187 | 187 | 1491 | 652 | structural problem |

("nnz" is the raw SuiteSparse-reported nonzero count including the
diagonal/both triangles; "Parsed edges" is the deduplicated undirected
edge count the harness's `load_mtx_file` actually produces after dropping
self-loops and symmetric duplicates.)

## Size range
N: 100-187 nodes. Parsed edges: 168-5972.

## Parse verification
100% parse success (15/15) against `scripts/r79_stdcorpora_eval.py`'s
`load_mtx_file` loader (pre-existing, unmodified). Zero parse errors, zero
zero-edge graphs in the final selection.

## Total download size
~200 KB (15 small `.mtx` files, well under the 500 MB budget in the brief).

## License note
The SuiteSparse Matrix Collection (Texas A&M / University of Florida) is
distributed for research use; the collection's terms request citation of
the collection when matrices are used in publications (see
https://sparse.tamu.edu/statistics for citation details). No individual
matrix in this sample carries a more restrictive license per its listing
page.

## Loader findings
See `.project-context/research/r79_native/stdcorpora/LOADER_FINDINGS.md`.
