# Criteria Ledger

This is the explicit stage-0 ledger for Dagua optimization criteria.

Use it before changing objectives or claiming a new optimization stage is
"complete."

The point is not to pretend every criterion can be maximized at once. The point
is to name them, classify them, and record which ones are:

- already measured
- partially measured
- still judged visually
- known to conflict

## Node Placement

### Measured now

- DAG consistency
- edge crossings
- node overlaps
- edge-length coefficient of variation
- runtime

### Partially measured / indirect

- same-rank ordering quality
- backbone readability
- skip-edge span readability
- cluster-aware separation

### Not yet measured cleanly

- cluster sibling overlap
- parent / child containment margin
- cluster interleaving
- cluster-mediated outside / inside separation quality

## Downstream Geometry

### Measured now

- very little directly; this stage is still underdeveloped

### Partially measured / indirect

- edge straightness
- some edge crossing proxies through routing and refinement

### Not yet measured cleanly

- edge-label collision count
- edge-node clearance
- edge-cluster clearance
- cluster-edge intrusion count
- cluster box pathology
- cluster label clearance
- text rhythm / offset smoothness

## Visual / Aesthetic Defaults

### Still mostly visual judgment

- typography hierarchy
- stroke hierarchy
- color restraint
- information density
- fill opacity discipline
- semantic emphasis discipline

## Known Tradeoffs

- crossings vs edge-length regularity
- crossings vs whitespace / compactness
- cluster compactness vs sibling cluster separation
- information density vs readability
- strict containment vs calm composition
- runtime vs quality everywhere

## Current Policy

1. Node placement is stage 1 and should keep improving numerically.
2. Downstream geometry is stage 2 and needs its own explicit metrics.
3. Visual defaults are stage 3 and should not substitute for missing stage-2 geometry work.
4. Cluster quality begins in stage 1, not only stage 2.
5. Same-level elements should interact directly; cross-level interactions should be mediated through container geometry where possible.
