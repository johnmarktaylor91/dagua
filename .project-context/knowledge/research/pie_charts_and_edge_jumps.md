# Pie Chart Node Backgrounds & Edge Crossing Jumps

Research date: 2026-03-22
Sources: Cytoscape.js source code, draw.io/mxGraph source code, computational geometry literature

---

## 1. Cytoscape.js Pie Chart Node Backgrounds

### Property Specification

Cytoscape.js supports pie chart backgrounds on nodes via indexed style properties:

**Global pie properties:**
| Property | Type | Default | Description |
|---|---|---|---|
| `pie-size` | sizeMaybePercent | `'100%'` | Diameter of pie relative to node (% of min(W,H)) or absolute px |
| `pie-hole` | sizeMaybePercent | `0` | Diameter of center hole (donut). 0 = solid pie |
| `pie-start-angle` | angle | `'0deg'` | Rotation offset for entire pie. Units: deg or rad |

**Per-slice properties (i = 1..16):**
| Property | Type | Default | Description |
|---|---|---|---|
| `pie-{i}-background-color` | color | `'black'` | Fill color of slice i |
| `pie-{i}-background-size` | percent | `'0%'` | Angular share as percentage [0, 100] |
| `pie-{i}-background-opacity` | zeroOneNumber | `1` | Opacity [0.0, 1.0] |

**Maximum slices:** 16 (hardcoded as `styfn.pieBackgroundN = 16`)

### Rendering Algorithm (from `drawing-nodes.mjs`)

```
radius = min(nodeW, nodeH) / 2
if pie-size is %: radius *= pie-size
if pie-size is px: radius = pie-size / 2

holeRadius = computed similarly from pie-hole
if holeRadius >= radius: return  // invisible

lastPercent = 0
for i in 1..16:
    percent = pie-i-background-size / 100  // map [0,100] -> [0,1]
    if percent + lastPercent > 1: percent = 1 - lastPercent

    angleStart = 1.5*PI + 2*PI*lastPercent + overallStartAngle  // 12 o'clock, clockwise
    angleEnd = angleStart + 2*PI*percent

    if holeRadius == 0:
        // Solid pie slice: moveTo center, arc, closePath
        ctx.moveTo(cx, cy)
        ctx.arc(cx, cy, radius, angleStart, angleEnd)
        ctx.closePath()
    else:
        // Donut slice: outer arc CW, inner arc CCW
        ctx.arc(cx, cy, radius, angleStart, angleEnd)
        ctx.arc(cx, cy, holeRadius, angleEnd, angleStart, anticlockwise=true)
        ctx.closePath()

    ctx.fill(color, opacity)
    lastPercent += percent
```

### Key Implementation Details

- **Shape independence:** Pie uses `min(nodeW, nodeH) / 2` as max radius -- works on ANY node shape,
  not just circles. The pie is always circular regardless of node shape.
- **Slice ordering:** Slices draw from 12 o'clock (1.5*PI) going clockwise.
- **Overflow clamping:** If cumulative slices exceed 100%, remaining slices are clamped/skipped.
- **No gap between slices:** Slices abut exactly (no spacing property).
- **Size values are percentages [0, 100]** -- NOT fractions. A 33% slice = `pie-1-background-size: 33`.
- **Donut (pie-hole):** Creates annular slices using two concentric arcs in opposite directions.
- **Data mapping:** Supports `mapData(attr, inMin, inMax, outMin, outMax)` for dynamic slice sizes.

### Example Usage

```javascript
{
    selector: 'node',
    style: {
        'pie-size': '80%',
        'pie-hole': '60%',           // donut
        'pie-start-angle': '0deg',
        'pie-1-background-color': '#E8747C',
        'pie-1-background-size': 'mapData(foo, 0, 10, 0, 100)',
        'pie-2-background-color': '#74CBE8',
        'pie-2-background-size': 'mapData(bar, 0, 10, 0, 100)',
        'pie-3-background-color': '#74E883',
        'pie-3-background-size': 'mapData(baz, 0, 10, 0, 100)',
    }
}
```

---

## 2. Draw.io Edge Crossing Jump Styles

### Style Properties

| Property | Values | Default | Description |
|---|---|---|---|
| `jumpStyle` | `'none'`, `'arc'`, `'gap'`, `'sharp'`, `'line'` | `'none'` | Visual style at crossing points |
| `jumpSize` | integer (points) | `6` | Size of the jump in points |
| `noJump` | `'1'` | unset | Excludes edge from being jumped over |

Set per-edge in XML style string:
```xml
style="...;jumpStyle=arc;jumpSize=6;"
```

### Which Edge Goes Over vs Under (Z-Order)

- The edge with `jumpStyle != 'none'` displays the visual jump indicator.
- **Only the "top" edge (higher z-order) shows jumps.** The edge created later is on top by default.
- Use "Arrange -> To Front / To Back" to control z-order.
- The `updateLineJumps()` function iterates over `validEdges` (previously processed edges).
  An edge only detects crossings with edges that were validated BEFORE it -- i.e., edges
  lower in the draw order. So the later/higher edge is the one that "jumps."

### Crossing Detection Algorithm (from Graph.js)

The algorithm is an O(E1 * E2) brute force segment-segment intersection:

```
updateLineJumps(state):
    for each segment (p0, p1) of this edge:
        list = []  // intersections on this segment

        for each previously-validated edge (state2):
            if bounding boxes don't overlap: skip
            if state2 has noJump='1': skip

            for each segment (p2, p3) of state2:
                pt = mxUtils.intersection(p0, p1, p2, p3)  // line-line intersection

                if pt exists AND pt is not too close to segment endpoints (> 0.5*scale):
                    // Filter out overlapping incoming/outgoing segments
                    if segment directions are too similar: skip

                    // Insert into list ordered by distance from p0
                    dist = (pt.x - p0.x)^2 + (pt.y - p0.y)^2
                    insert {distSq, x, y} into list sorted by distSq

        // Emit: type=0 for normal waypoints, type=1 for crossings
        for each intersection in list:
            routedPoints.push({type: 1, x, y})
```

Key details:
- Uses `mxUtils.intersection()` for line-line segment intersection (not bezier).
- **Curved edges are excluded:** `isLineJumpState()` returns false for `curved` edges.
  Only `'connector'`, `'filledEdge'`, and `'wire'` shapes support jumps.
- Collinear/overlapping segments are filtered via `ptSegDistSq` and `ptLineDist` checks.
- Intersections too close to segment endpoints (within 0.5*scale) are discarded.
- Multiple intersections on one segment are de-duplicated by position.

### Jump Rendering Geometry (from `mxConnector.prototype.paintLine` override)

At each crossing point, the edge path is split. The segment direction vector `n` is computed
and scaled to `jumpSize`:

```
n = normalize(pt - last) * size
where size = (jumpSize - 2) / 2 + strokeWidth

p0 = crossingPoint - n  // entry point (before crossing)
p1 = crossingPoint + n  // exit point (after crossing)
f = direction sign (1 or -1 based on segment direction)
```

**Arc style** (`jumpStyle='arc'`):
```
f *= 1.3
curveTo(p0.x - n.y*f, p0.y + n.x*f,   // control point 1
        p1.x - n.y*f, p1.y + n.x*f,   // control point 2
        p1.x, p1.y)                     // end point
```
This is a cubic bezier that arcs perpendicular to the edge direction.
The 1.3 factor makes the arc slightly taller than a semicircle.

**Sharp style** (`jumpStyle='sharp'`):
```
lineTo(p0.x - n.y*f, p0.y + n.x*f)   // up-left
lineTo(p1.x - n.y*f, p1.y + n.x*f)   // across-right
lineTo(p1.x, p1.y)                    // back down
```
Creates an inverted-V / triangular bump perpendicular to the edge.

**Gap style** (`jumpStyle='gap'` or default else):
```
moveTo(p1.x, p1.y)
```
Simply lifts the pen -- creates a visible gap in the edge at the crossing.

**Line style** (`jumpStyle='line'`):
```
moveTo(p0.x + n.y*f, p0.y - n.x*f)
lineTo(p0.x - n.y*f, p0.y + n.x*f)   // cross mark at entry
moveTo(p1.x - n.y*f, p1.y + n.x*f)
lineTo(p1.x + n.y*f, p1.y - n.x*f)   // cross mark at exit
moveTo(p1.x, p1.y)
```
Draws perpendicular tick marks at both sides of the crossing.

### Limitations

- **No curved edge support.** Jumps only work on straight polyline segments.
- **No bezier intersection.** Detection is line-line only.
- **O(E^2) worst case.** Every segment of every edge is tested against every segment
  of every previously-validated edge. Mitigated by bounding box pre-filter.
- **Z-order dependent.** The edge drawn later shows jumps. No automatic "which is on top" heuristic.
- **No per-crossing customization.** All crossings on one edge use the same jump style.

---

## 3. Edge Crossing Detection Algorithms

### Approach 1: Brute Force O(E^2) -- for small graphs

For each pair of edges, test all segment pairs:
- Line-line intersection: `mxUtils.intersection()` or cross-product orientation test
- Filter: reject if intersection is outside both segments' parameter range [0, 1]
- This is what draw.io uses. Fine for <100 edges.

### Approach 2: Bentley-Ottmann Sweep Line -- O((n+k) log n)

For n line segments with k intersections:

**Data structures:**
- Event queue Q: priority queue ordered by x-coordinate (min-heap)
- Sweep status T: balanced BST of segments ordered by y-coordinate at current sweep x

**Event types:**
1. Left endpoint: insert segment into T, check intersections with neighbors above/below
2. Right endpoint: remove segment from T, check if former neighbors now intersect
3. Intersection: swap the two segments in T, check new neighbors for intersections

**Pseudocode:**
```
Initialize Q with all 2n endpoints
T = empty BST

while Q not empty:
    event = Q.extractMin()

    if event is LEFT_ENDPOINT of segment s:
        T.insert(s)
        above = T.predecessor(s)
        below = T.successor(s)
        if above exists: remove old (above, below) intersection from Q
        if above intersects s to the right: Q.insert(intersection(above, s))
        if below intersects s to the right: Q.insert(intersection(s, below))

    elif event is RIGHT_ENDPOINT of segment s:
        above = T.predecessor(s)
        below = T.successor(s)
        T.delete(s)
        if above and below intersect to the right: Q.insert(intersection(above, below))

    elif event is INTERSECTION of segments s1, s2:
        report(intersection_point)
        T.swap(s1, s2)
        // s1 was above s2, now s2 is above s1
        new_above_s1 = T.predecessor(s1)  // formerly above s2
        new_below_s2 = T.successor(s2)    // formerly below s1
        check new_above_s1 vs s1 for intersection
        check s2 vs new_below_s2 for intersection
        remove stale intersection events
```

**Assumptions (standard form):**
- No vertical segments (handle via slight rotation or special case)
- No three segments meet at one point (handle by processing simultaneously)
- No overlapping collinear segments

**Complexity:** O((n + k) log n) time, O(n + k) space.
For sparse crossings (k << n^2), much better than brute force.

### Approach 3: Bezier Curve Intersection

For edges rendered as cubic bezier curves, line-line intersection doesn't work.

**Method A: Flatten then sweep**
1. Subdivide each bezier into short line segments (adaptive: subdivide until
   segments are within tolerance of the curve)
2. Use Wang's formula to estimate subdivision depth needed
3. Apply Bentley-Ottmann or brute force on the resulting line segments
4. Map intersection parameters back to original curve t-values

**Method B: Recursive subdivision with bounding box rejection**
```
intersect(curve1, curve2, tol):
    if bbox(curve1) does not overlap bbox(curve2): return []
    if curve1 is flat enough AND curve2 is flat enough:
        return line_line_intersect(linearize(curve1), linearize(curve2))
    // Subdivide at midpoint using de Casteljau
    c1a, c1b = subdivide(curve1, 0.5)
    c2a, c2b = subdivide(curve2, 0.5)
    return union(
        intersect(c1a, c2a, tol),
        intersect(c1a, c2b, tol),
        intersect(c1b, c2a, tol),
        intersect(c1b, c2b, tol)
    )
```
Converges rapidly due to convex hull property of bezier curves.

**Method C: Cubic bezier vs line (analytical)**
For intersecting a cubic bezier with a straight line:
1. Compute implicit line equation: Ax + By + C = 0
2. Substitute bezier parametric form into line equation -> cubic polynomial in t
3. Solve cubic via Cardano's method (up to 3 real roots)
4. Filter roots to t in [0, 1] and check line segment bounds
5. Maximum 3 intersections per curve-line pair

**Result data structure:**
Intersections stored as (t1, t2) parameter pairs -- position on curve1 and curve2 respectively.
Physical coordinates computed by evaluating the curve at the parameter value.

### Recommendation for Dagua

For dagua's rendering pipeline (matplotlib/SVG backend, not interactive):
- **Straight edges:** O(E^2) brute force with bounding box pre-filter is fine for <500 edges.
  For larger graphs, Bentley-Ottmann.
- **Bezier edges:** Flatten to line segments first (adaptive subdivision), then use the
  straight-edge algorithm. The flattening tolerance should match the rendering resolution.
- **Store crossings as:** list of (edge_id_1, edge_id_2, point_x, point_y, t1, t2)
  where t1/t2 are parameter positions along each edge's path.
