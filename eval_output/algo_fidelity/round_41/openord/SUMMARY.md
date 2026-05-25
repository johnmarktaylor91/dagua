# Round 41 OpenOrd Summary

## Verdict

`openord == drl`; no new OpenOrd engine was added.

## Evidence

- Required local source check:

  ```text
  grep -i openord /home/jtaylor/projects/_references/igraph/src/layout/drl/*.cpp
  # no output; exit code 1
  ```

  The igraph DRL implementation does not use the OpenOrd name internally.

- The local igraph source documents `igraph_layout_drl` as Shawn Martin's DrL
  layout generator and cites:

  ```text
  Martin, S., Brown, W.M., Klavans, R., Boyack, K.W.,
  DrL: Distributed Recursive (Graph) Layout. SAND Reports, 2008.
  ```

  Source: `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp`.

- The same local source exposes the phase schedule used by the OpenOrd family:
  liquid, expansion, cooldown, crunch, and simmer. The default template uses
  200 liquid, 200 expansion, 200 cooldown, 50 crunch, and 100 simmer iterations,
  with `edge_cut = 32.0 / 40.0`.

- Current igraph C docs identify `igraph_layout_drl` as the DrL layout generator
  by Shawn Martin et al., cite the same DrL technical report, and note that
  igraph includes only a subset of the full DrL functionality: no parallel runs
  or recursive multilevel layouting.
  URL: https://igraph.org/c/html/main/igraph-Layout.html

- Gephi's OpenOrd plugin page identifies OpenOrd as the same algorithm lineage:
  it describes the same five phases and states that the algorithm "was formerly
  known as DrL, and before that VxOrd." It also cites Martin, Brown, Klavans,
  and Boyack's 2011 OpenOrd paper.
  URL: https://gephi.org/desktop/plugins/openord-layout/

## Decision

Dagua's existing `classic_drl` family is the correct implementation target for
OpenOrd/DrL fidelity. Adding `openord_default`, `openord_liquid`,
`openord_simmer`, or `openord_crunch` would create duplicate engine names rather
than a distinct engine.

## Implementation

Skipped by design. No pipeline, registry, variant, benchmark, or result JSON
changes are required for R41.

## Verification

No smoke tests were run because no code path changed. This round is a source and
documentation verification task.

## Concerns

Gephi's plugin page notes that the plugin version omits the multilevel version
of OpenOrd. igraph likewise documents that recursive multilevel layouting is not
included. That is an implementation-scope limitation, not evidence that OpenOrd
and DrL should be treated as separate algorithms in Dagua.

## Knowledge

For Dagua fidelity tracking, treat OpenOrd as the public/Gephi name for the
DrL/VxOrd algorithm family unless a future task explicitly targets Gephi plugin
UI behavior or parallel/multilevel behavior as a separate compatibility surface.
