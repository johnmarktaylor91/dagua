# r80-S5: Recon -- full-element layout scope (edge routing, labels, node shape/size)

READ-ONLY inventory + gap analysis. Zero file modifications (your /tmp report is the one
exception). Repo: /home/jtaylor/.claude/worktrees/dagua-native (branch r79/native), venv
.venv/bin/python. The sprint mandate expanded: dagua's native layout should excel at ALL
placement-relevant output -- node positions, edge routing, edge/node/cluster label
placement, node shape/size effects -- not just node coordinates. Cosmetics (color/theme)
out of scope.

## Questions to answer (each with file:line evidence)

### A. Edge routing today
1. Where does edge routing happen -- layout-time ops or render-time? Inventory routing
   code (grep: route, spline, bezier, orthogonal, corridor, waypoint) across
   dagua/layout/ and dagua/render/.
2. What routing styles exist (straight, curved, orthogonal, bundled?) and which pipelines
   use which (layered dummy-node corridors vs force layouts)?
3. Do routed edges avoid node boxes (node-edge overlap avoidance) anywhere? Self-loops and
   multi-edges handling?
4. Does ANY eval metric measure edge-routing quality (node-edge crossings, bend count,
   spline smoothness, edge-edge crossing of ROUTED paths vs straight-line)? Check
   dagua/metrics and dagua/eval thoroughly.

### B. Labels and text
1. How node label text determines node extents (text measurement path) -- confirmed used
   by layout (node sizes in overlap losses/projection)?
2. EDGE labels: supported? placed how? do they participate in any overlap avoidance?
3. Cluster labels / titles: placed how? reserved space in layout?
4. Free-standing text/annotations: any support?
5. Does any metric measure label overlap / label-edge collision / label readability?

### C. Node shape/size
1. What shapes exist and does layout see true shape geometry or just bounding boxes
   (overlap projection, spacing losses)?
2. Any layout-time node RESIZING/aspect adaptation (e.g. wide labels -> wide nodes ->
   layout consequences)? Port/anchor points for edges (edge attaches where on the node)?

### D. External comparison surface
1. What do external engines in the benchmark emit that we DISCARD -- e.g. graphviz dot
   -Tjson splines, ELK edge sections, label positions? Check the competitor adapters
   (dagua/eval/competitors/*.py): what is stored in the frozen store (positions only?).
2. If we wanted to score "full drawing quality" (routing + labels) against externals
   fairly, what would the harness need to capture per engine? Estimate effort per adapter.

### E. Gap ranking
Rank the top gaps by (impact on real-drawing quality) x (feasibility), with a one-line
implementation sketch each. Consider: node-edge crossing avoidance in routing; edge-label
placement with overlap avoidance; routed-path-aware crossing metric; cluster label space
reservation; port assignment on node boundary. Note which gaps are LAYOUT problems (affect
positions) vs pure POST-PLACEMENT problems (given positions, place/route the rest) --
the post-placement ones can be improved without touching the node-placement benchmark.

## Output contract
Report to /tmp/r80_s5_full_element_recon_REPORT.md AND full text in your final message:
sections A-E, each claim with file:line, plus a "measured vs unmeasured quality
dimensions" table and your ranked gap list.
