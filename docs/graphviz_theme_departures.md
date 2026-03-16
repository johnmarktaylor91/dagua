# Graphviz Theme Departure Log

Every difference between `graphviz_strict` (exact Graphviz match) and `graphviz`
(improved variant) is documented here with justification.

## Node Styling

| Parameter | graphviz_strict | graphviz | Justification |
|-----------|----------------|----------|---------------|
| Font family | Times-Roman (serif) | System sans-serif | Serif fonts look dated in technical diagrams; sans-serif is the modern standard |
| Font size | 14pt | 12pt | 14pt is visually heavy; 12pt provides better text-to-node ratio |
| Fill color | #FFFFFF (white) | #FAFBFC (off-white) | Subtle tint adds depth without changing the clean aesthetic |
| Border color | #000000 (black) | #333333 (dark gray) | Pure black borders are harsh; dark gray is softer while remaining clearly visible |
| Border width | 1.0pt | 1.2pt | Slightly heavier line compensates for softer color, maintains crispness on screen |
| Text color | #000000 | #1A1A1A | Matches the softer border palette |
| Padding | (8.0, 4.0) | (9.0, 5.0) | Slightly more breathing room around labels |
| Min width | 54pt | 50pt | Slightly smaller minimum allows compact renders |
| Min height | 36pt | 34pt | Matches reduced minimum width proportionally |
| Input/output fill | #FFFFFF | Subtle tint | Light green for inputs, light red for outputs — aids visual scanning |

## Edge Styling

| Parameter | graphviz_strict | graphviz | Justification |
|-----------|----------------|----------|---------------|
| Color | #000000 | #4A4A4A | Edges should visually recede behind nodes |
| Width | 1.0pt | 1.1pt | Compensates for lighter color |
| Opacity | 1.0 | 0.85 | Creates visual depth — nodes are foreground, edges are connective tissue |
| Label font size | 14pt | 11pt | Edge labels should be subordinate to node labels |

## Cluster Styling

| Parameter | graphviz_strict | graphviz | Justification |
|-----------|----------------|----------|---------------|
| Fill | none (transparent) | #F5F5F5 | Subtle fill clearly delineates group boundaries |
| Border color | #000000 | #BBBBBB | Black cluster borders compete with node borders |
| Corner radius | 0 | 3pt | Very subtle rounding softens the appearance |
| Font size | 14pt | 12pt | Matches node font size |
| Opacity | 1.0 | 0.7 | Clusters are context, not primary content |

## Graph-Level

| Parameter | graphviz_strict | graphviz | Justification |
|-----------|----------------|----------|---------------|
| Background | #FFFFFF | #FAFAFA | Warm white is easier on the eyes |
| Title font size | 14pt | 12pt | Consistency with body text |
