# Round 27 Diff: classic_linlog

## Verdict

`principled_residual: source_unavailable`

I found source-level copies of Andreas Noack's LinLogLayout code, but I did not
find an installable reference comparator suitable for `classic_linlog` in the
dagua eval environment. I therefore did not add a competitor adapter and did not
change the five `classic_linlog_*` variants in `dagua/eval/variants.py`.

The current registry state remains intentional for this round: all five LinLog
variants still have `original_engine=None` because wiring NetworkX ForceAtlas2
LinLog mode or igraph DrL would create a false comparator for a different
algorithm family.

## Search Trail

Commands run:

```bash
gh search repos "linlog layout" --limit 20
gh search code "linlog" --filename layout.py --limit 50
gh search code "LinLogLayout" --limit 50
gh search code "Noack" "linlog" --limit 50
gh search code "class LinLogLayout" --language Java --limit 50
gh search code "class MinimizerClassic" "LinLog" --language Java --limit 50
gh search repos linloglayout --limit 50
```

Web searches run:

```text
Andreas Noack LinLogLayout Java source
Noack LinLogLayout.java
LinLogLayout Java Andreas Noack Cottbus
"Energy-Based Clustering" "LinLogLayout"
Linloglayout-lib Maven ifunsoftware
code.google.com p linloglayout source archive
linloglayout jar download
Andreas Noack LinLogLayout code.google.com archive
```

Environment checks run:

```bash
python - <<'PY'
import networkx as nx
import inspect
print(nx.__version__)
print(hasattr(nx, "forceatlas2_layout"))
print(inspect.signature(nx.forceatlas2_layout))
PY

python - <<'PY'
import importlib.util
for name in ["networkx", "igraph", "networkit", "tulip"]:
    print(name, importlib.util.find_spec(name) is not None)
PY

java -version
curl -L --silent \
  'https://search.maven.org/solrsearch/select?q=g:%22com.ifunsoftware.thirdparty%22%20AND%20a:%22linloglayout-lib%22&rows=20&wt=json'
```

## References Found

- Original-style Java mirror:
  `https://github.com/ifunsoftware/Linloglayout-lib`
  - The repository contains `LinLogLayout.java`, `MinimizerClassic.java`,
    `MinimizerBarnesHut.java`, `Node.java`, `Edge.java`, and
    `OptimizerModularity.java`.
  - `MinimizerClassic.java` carries `Copyright (C) 2008 Andreas Noack`, names
    Noack as author, and describes itself as a minimizer for the LinLog energy
    model and generalizations.
  - `LinLogLayout.java` reads an edge-list file, symmetrizes the graph, assigns
    node weights from weighted degree, initializes positions with
    `Math.random()`, runs `new MinimizerBarnesHut(nodes, edges, 0.0, 1.0, 0.05)`
    for 100 iterations, then writes positions and clusters.
  - It has a Maven `pom.xml` with group
    `com.ifunsoftware.thirdparty`, artifact `linloglayout-lib`, and version
    `1.0.3-SNAPSHOT`, but Maven Central search returned `numFound: 0`.
  - The POM points to `repository.ifunsoftware.com`; both HTTP and HTTPS checks
    failed DNS resolution from this environment.

- Google Code archive mirrors:
  `DuranLiao/linloglayout`, `Nadiakacem13/linloglayout`,
  `VijayKrishna/linloglayout`, and `EngAAlex/linloglayout` are labeled as
  automatic exports from `code.google.com/p/linloglayout`, but the checked
  `DuranLiao/linloglayout` repository is empty through the GitHub contents API.

- `jnthnclt/nicity`:
  `https://github.com/jnthnclt/nicity/blob/master/nicity-view/src/main/java/linloglayout/LinLogLayout.java`
  - Contains an embedded `linloglayout` package with Noack attribution.
  - This is part of an old Java UI project, not a packaged standalone layout
    executable for dagua eval.

- Tulip:
  `https://github.com/Tulip-Dev/tulip/tree/master/plugins/layout/LinLog`
  - Contains a C++ LinLog plugin named `LinLog Layout (Noack)`.
  - The Python `tulip` module is not installed in this eval environment.
  - Adding a Tulip dependency would be larger than a competitor adapter and was
    not appropriate for this research round.

- NetworkX:
  `https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.forceatlas2_layout.html`
  - Installed version: `networkx 3.6.1`.
  - `forceatlas2_layout(..., linlog=False)` exists, and `linlog=True` switches
    ForceAtlas2 to logarithmic attraction.
  - It references ForceAtlas2, not Noack's classic LinLog optimizer. It exposes
    no `a`/`r` exponent controls matching dagua's five `classic_linlog`
    variants.

- igraph:
  `https://igraph.org/python/versions/latest/api/igraph.GraphBase.html#layout_drl`
  - Installed version: `igraph 1.0.0`.
  - `layout_drl` is present, but it is the Distributed Recursive Layout
    algorithm with its own phase/annealing parameters. It is LinLog-derived at
    most, not a direct Noack LinLog comparator and has no matching `a`/`r`
    surface.

- Networkit:
  `https://pypi.org/project/networkit/`
  - A PyPI package exists, but `networkit` is not installed here.
  - I found no installed dagua competitor path or local dependency entry for a
    Networkit LinLog layout.

## Why No Adapter Was Added

The only true reference candidate found is source code, not an installable
runtime reference in the current eval environment. It would require one of:

- vendoring LGPL Java sources into `dagua/eval/competitors/`,
- dynamically downloading source or a jar during eval,
- adding a new Java build step plus a wrapper CLI,
- adding Tulip or Networkit as a new optional dependency,
- or misusing NetworkX/igraph algorithms that are not classic LinLog.

All five `classic_linlog` variants require a comparator that can accept the
same meaningful knobs:

- default: `a=1.0`, `r=0.0`, `steps=300`
- quadratic: `a=2.0`, `r=0.0`, `steps=300`
- power: `a=1.0`, `r=0.5`, `steps=300`
- steps100: `a=1.0`, `r=0.0`, `steps=100`
- steps500: `a=1.0`, `r=0.0`, `steps=500`

No installed reference exposes that surface. Noack's Java code does expose
attraction and repulsion exponents internally, but the public demo CLI is fixed
to Barnes-Hut, `repuExponent=0.0`, `attrExponent=1.0`, `gravFactor=0.05`, and
100 iterations. A faithful adapter would therefore need a manual Java wrapper
or a port, not just a registry entry.

## Source-Level Notes

These notes are not a line-by-line fidelity diff because no reference was
wired. They identify likely future diff axes if a manual Java adapter is built.

- Initialization:
  Noack's demo uses Java `Math.random() - 0.5` per coordinate. Dagua seeds
  PyTorch random initialization through `LinLogInitializePositions`, so exact
  seed parity will require replacing or wrapping the Java initializer.

- Graph preprocessing:
  Noack's demo reads directed edge rows and explicitly symmetrizes by summing
  each edge with its reverse. Dagua's pipeline receives `edge_index` directly.
  A reference adapter must decide whether to write both directions, rely on
  Noack's symmetrizer, or transform dagua inputs first.

- Node weights:
  Noack uses weighted degree as node repulsion weight. Dagua's current
  `LinLogRepulsionLoss` performs unweighted all-pairs repulsion over node pairs.
  This is a probable high-impact algorithmic difference for R28, but it should
  not be changed in this round.

- Optimization:
  Noack's classic minimizer uses coordinate-wise node moves with line search
  and a staged exponent schedule for low repulsion exponents. Dagua's pipeline
  uses Adam, `LossGroup`, `OptimizerStep`, and `LRDecay`.

- Gravity:
  Noack's demo uses `gravFactor=0.05` to keep components finite. Dagua's
  objective has no visible gravity term in `layout_linlog_pipeline`.

- Output normalization:
  Noack writes raw coordinates plus clusters. Dagua finalizes positions through
  `LinLogFinalizePositions`, which normalizes and casts final coordinates.

## Baseline Measurement

No baseline was run because no reference adapter was added.

The closest available installed commands are intentionally not baselines:

- `networkx.forceatlas2_layout(linlog=True)` is ForceAtlas2 with logarithmic
  attraction, not classic LinLog.
- `igraph.layout_drl` is DrL, not Noack's LinLog optimizer.

## Suggested Future Work

1. Create a dedicated manual Java wrapper around Noack's `MinimizerClassic` or
   `MinimizerBarnesHut` that accepts:
   `dim`, `input`, `output`, `iterations`, `attrExponent`, `repuExponent`,
   `gravFactor`, and a deterministic seed or explicit initial positions.
2. Keep the LGPL source or compiled jar outside dagua unless legal review
   approves vendoring. Prefer a small build script that can consume a pinned
   upstream checkout.
3. Add a `noack_linlog` competitor only after the wrapper can run without GUI
   display and without cluster output being required.
4. Map variants conservatively:
   - `classic_linlog_default` -> `attrExponent=1.0`, `repuExponent=0.0`
   - `classic_linlog_quadratic` -> `attrExponent=2.0`, `repuExponent=0.0`
   - `classic_linlog_power` -> `attrExponent=1.0`, `repuExponent=0.5`
   - step variants -> wrapper iteration count
5. Decide whether fidelity should target `MinimizerClassic` all-pairs behavior
   or `MinimizerBarnesHut` demo behavior. The former is simpler and closer to a
   line-by-line mathematical comparator; the latter is the public demo default.
