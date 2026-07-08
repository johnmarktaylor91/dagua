# r80-S3 loader findings (Rome / North / SuiteSparse fetch)

Scope note: this was a data-fetch + parse-verify task only. Zero layout
code touched, zero tuning, zero evaluation runs. The one code change made
(`load_graphml_file`) was a trivial (<10 line) format-tolerance addition,
required because without it 0 graphs would have loaded from either the
Rome or North corpus -- not a judgment call, a hard blocker.

## 1. `graphdrawing.org` SSL failure was a stale domain, not a dead source

The prior fetch attempt (2026-07-07, logged in
`eval_output/stdcorpora/README.md`) failed with an SSL hostname mismatch
against `https://graphdrawing.org`. Root cause: the domain now 301-redirects
(over plain HTTP only) to `https://graphdrawing.unipg.it`, which has a valid
certificate and serves the same data page and archives. The original URLs
(`https://graphdrawing.org/data/rome/rome.tar.gz`,
`.../data/north/north.tar.gz`) referenced in `scripts/fetch_stdcorpora.sh`
no longer exist in that path shape either -- the current archives are
`rome-graphml.tgz` / `north-graphml.tgz` (GraphML, not the old `.tar.gz`
adjacency-list bundles) plus separate `GDT-testsuite-*.tgz` archives in the
original Rome/AT&T text formats. `scripts/fetch_stdcorpora.sh` was not
modified (per brief scope) but its hardcoded URLs are now stale; flagging
for whoever next touches that script.

## 2. GraphML (`.graphml`) is a distinct format from GML (`.gml`) -- new loader required

The current official Rome/North mirrors are GraphML XML
(`<?xml ...><graphml><graph edgedefault="..."><node .../><edge .../>`),
not classic bracket-syntax GML
(`graph [ node [ id 0 ] edge [ source 0 target 1 ] ]`). The existing
`load_gml_file` calls `networkx.read_gml`, which raises
`NetworkXError: cannot tokenize <?xml version=...` on these files --
confirmed directly:

```
>>> nx.read_gml("grafo798.26.graphml", label=None)
NetworkXError: cannot tokenize <?xml version="1.0" encoding="UTF-8"?> at (1, 1)
```

And the harness's `load_corpus` only dispatches on a fixed
`{".graph", ".gml", ".mtx"}` suffix map, so `.graphml` files were silently
*skipped* (not errored) before this fix -- `load_corpus` would have
reported "0 graphs loaded" for both Rome and North with no error at all.

Fix applied (`scripts/r79_stdcorpora_eval.py`): added `load_graphml_file`
(uses `networkx.read_graphml`) and registered `.graphml` in the `loaders`
dict inside `load_corpus`. Two new unit tests added to
`tests/test_stdcorpora_eval.py` covering an undirected Rome-style fixture
and the directed-inference edge case below. Full test suite still green
(6/6 passed).

## 3. North GraphML files omit `edgedefault` -- directedness inference bug, fixed inline

North/AT&T GraphML exports (Y-Files XML) omit the `edgedefault` attribute
on `<graph>` entirely (the DTD marks it `#REQUIRED` but the files don't
comply). NetworkX's `read_graphml` defaults `is_directed()` to `False` in
that case. The existing `load_gml_file` pattern passes
`nx_graph.is_directed()` straight into `infer_directed(path, format_directed=...)`,
which short-circuits the path-based `"north" in lowered` fallback whenever
`format_directed` is not `None`. Applied verbatim to GraphML, this would
have silently scored every North DAG as undirected -- a real fidelity bug,
not a cosmetic one, since North's whole point is to hold out *directed*
layouts.

Fix: `load_graphml_file` only trusts the file's own `is_directed()` for
non-`north` corpora; for `north`, it passes `format_directed=None` so
`infer_directed` falls back to the existing (already-correct) path-based
heuristic. Verified against the real 107-file North sample: 107/107 load
as directed. Rome sample (152/152) still loads as undirected via the
file's explicit `edgedefault="undirected"`.

This is scoped narrowly to the new loader function; `load_gml_file` (the
pre-existing classic-GML path) was not touched, since real `.gml` files
reliably encode `directed 1` inline (see existing test fixture) and don't
share this omission behavior.

## 4. SuiteSparse "symmetric" HB mass matrices parse to zero-edge graphs

Initial SuiteSparse selection (smallest N>=100, `psym == 1.0`, HB/Pajek
groups) picked up `bcsstm03`, `bcsstm04`, `bcsstm22` -- HB's "mass matrix"
companions to `bcsstk03/04/22`. These are pattern-symmetric but almost
purely diagonal (`nnz` roughly equals `rows`: e.g. `bcsstm03` has 112 rows
and nnz=72). `load_mtx_file` correctly parses them (no error) but drops all
self-loops per its `source != target` filter, producing exactly 0 edges --
a technically-valid but graph-theoretically useless holdout input.

Not a loader bug (the loader does the right thing, dropping self-loops);
this is a selection-rule gap. Fixed by tightening the selection filter to
require `nnz > 2 * rows` (ensures real off-diagonal structure) before
taking the smallest 15. Re-verified: 15/15 final selection has nonzero
edges, range 168-5972 parsed edges. See `MANIFEST_suitesparse.md` for the
full before/after matrix list.

## Summary: final parse rates

| Corpus | Files | Parse OK | Parse rate |
|---|---:|---:|---:|
| rome | 152 | 152 | 100% |
| north | 107 | 107 | 100% |
| suitesparse | 15 | 15 | 100% |

No holdout evaluation was run against these corpora. This is data-fetch and
loader-verification only, per the brief's scope.
