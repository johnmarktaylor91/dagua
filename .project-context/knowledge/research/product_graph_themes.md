# Product Graph/Diagram Theme Research

Date: 2026-03-22
Purpose: Identify products with distinctive graph/diagram aesthetics worth
implementing as dagua themes.

---

## Evaluation Criteria

For each product:
- **Recognizable?** Would someone say "that looks like X" from the theme alone?
- **Distinctive?** Does it have unique visual identity, not just generic boxes?
- **Well-known?** How large is the user base?
- **Implementable?** Can dagua approximate it with colors/shapes/fonts/edges?

---

## Tier 1: Highly Distinctive, Highly Recognizable

### 1. Excalidraw
- **Recognition:** VERY HIGH. The hand-drawn aesthetic is instantly identifiable.
- **Audience:** Massive. Millions of users, ubiquitous in tech blogs/docs.
- **Visual DNA:**
  - Background: white (#FFFFFF) or very light gray
  - Shapes: rectangles/ellipses with rough/sketchy strokes (simulated hand-drawn)
  - Stroke color: #1e1e1e (near-black) by default
  - Fill colors: from Open Color palette (pastel washes):
    - Light red: #ffc9c9 (oc-red-2), Light blue: #a5d8ff (oc-blue-2)
    - Light green: #b2f2bb (oc-green-2), Light yellow: #ffec99 (oc-yellow-2)
    - Light violet: #d0bfff (oc-violet-2), Light orange: #ffd8a8 (oc-orange-2)
    - Stroke variants: oc-*-9 (darkest shade of each hue)
  - Fill style: cross-hatch or hachure (pencil-shading effect)
  - Font: Excalifont/Virgil (hand-drawn), or Comic Shanns
  - Edges: hand-drawn lines with slight wobble, simple arrowheads
  - Corner radius: organic/imperfect
- **Dagua implementation:** Approximate with pastel fills, thin dark strokes,
  hand-drawn font (Virgil/Comic Shanns), possibly rough edge rendering.
  The cross-hatch fill would be the hardest part to replicate.
- **Priority: VERY HIGH** -- most recognizable diagram aesthetic on the internet.

### 2. n8n
- **Recognition:** HIGH among automation/developer community.
- **Audience:** Large. Major open-source workflow automation platform.
- **Visual DNA:**
  - Background: light gray dotted grid canvas (#F5F5F5 area, dot pattern)
  - Nodes: white (#FFFFFF) rounded rectangles with subtle shadow
  - Node border: light gray, with colored left-side accent stripe per node type
  - Brand primary: #EA4B71 (coral pink) for UI accents, not node fills
  - Node type colors: each integration has its own brand color on the accent stripe
  - Font: sans-serif (likely Inter or system font stack)
  - Edges: smooth bezier curves, gray (#B0B0B0 range), no arrowheads typically
  - Connection dots: small circles at node edges (input/output ports)
  - Overall feel: clean, modern, SaaS-minimal with bright accent touches
- **Dagua implementation:** White nodes on light-gray bg, rounded rects,
  colored left accent bar per node, gray bezier edges. Very achievable.
- **Priority: HIGH** -- distinctive node-editor aesthetic, large user base.

### 3. Linear
- **Recognition:** HIGH among developer/PM community. "Linear-style" is a
  recognized design trend (dark, indigo-accented, ultra-minimal).
- **Audience:** Large. Premium project management tool, design-forward brand.
- **Visual DNA:**
  - Background: very dark (#0F0F10 "Woodsmoke" or #1A1A1A)
  - Text: light gray/off-white (#EEEFF1)
  - Accent: indigo/purple (#5E6AD2 Linear signature indigo)
  - Node fill: slightly elevated dark surface (#1E1E20 or #222226)
  - Borders: subtle dark gray (#2A2A2E), nearly invisible
  - Font: Inter (or SF Pro) -- clean, modern sans-serif
  - Edges: thin, muted gray or indigo-tinted
  - Overall feel: dark-mode-native, minimal, "expensive" feel
  - Shadows: very subtle, elevation-based
- **Dagua implementation:** Dark background, dark-surface nodes, indigo accents,
  thin muted edges. Straightforward color-only theme.
- **Priority: HIGH** -- "Linear-style dark mode" is a whole design movement.

### 4. GitHub Actions
- **Recognition:** HIGH. Every developer has seen the workflow visualization.
- **Audience:** Massive. Largest developer platform.
- **Visual DNA:**
  - Background: white (#FFFFFF) or GitHub dark (#0D1117)
  - Nodes: rounded rectangles with status-colored left border
  - Status colors (from Primer design system):
    - Success: green (#1A7F37 light / #3FB950 dark)
    - Failure: red (#CF222E light / #F85149 dark)
    - In progress: yellow/amber (#9A6700 light / #D29922 dark)
    - Neutral/pending: gray (#656D76 light / #8B949E dark)
  - Font: system font stack (-apple-system, BlinkMacSystemFont, Segoe UI)
  - Edges: straight lines with rounded corners at turns, gray
  - Nodes have icon + text layout
  - Overall feel: clean, functional, status-driven
- **Dagua implementation:** Status-colored accents on neutral nodes.
  Both light and dark variants. Very achievable.
- **Priority: HIGH** -- universally recognized by developers.

### 5. Airflow
- **Recognition:** HIGH among data engineers.
- **Audience:** Large. Standard data orchestration tool.
- **Visual DNA:**
  - Background: white (#FFFFFF) with light Airflow blue tints
  - Nodes: rounded rectangles, operator-type colored:
    - BashOperator: #fff (white with border)
    - PythonOperator: ~#3776AB (Python blue-ish)
    - Sensor operators: #e6f1f2 (pale cyan)
    - HTTP operators: #f4a460 (sandy brown/orange)
    - Branch operators: pastel greens
  - Node border: darker shade of fill color
  - Task status overlays: green (success), red (failed), yellow (running),
    pink/purple (retry), gray (queued), light green (scheduled)
  - Font: standard sans-serif
  - Edges: straight/polyline connections between tasks
  - Overall feel: functional, colorful (each operator type = different color),
    resembles a subway map of data pipeline stages
- **Dagua implementation:** Multi-colored nodes by type, status color overlays.
  Need a way to map node categories to colors.
- **Priority: MEDIUM-HIGH** -- distinctive "rainbow DAG" look, but the per-
  operator coloring makes it less of a single "theme" and more of a palette.

---

## Tier 2: Distinctive, Recognizable Within Target Audience

### 6. dbt (data build tool)
- **Recognition:** MEDIUM-HIGH among data professionals.
- **Audience:** Large. Standard data transformation tool.
- **Visual DNA:**
  - Background: white (#FFFFFF)
  - Source nodes: green fills (#22C55E range)
  - Model nodes: blue fills (#3B82F6 range)
  - Test nodes: yellow/amber
  - Exposure nodes: orange
  - Node shape: rounded rectangles with resource-type icon
  - Font: sans-serif (Inter-like)
  - Edges: straight lines, gray, with dependency arrows
  - Layout: left-to-right DAG flow
  - Overall feel: clean data lineage, color = resource type
- **Dagua implementation:** Resource-type color mapping. Achievable.
- **Priority: MEDIUM** -- recognizable to data folks, less so to general audience.

### 7. Dagster
- **Recognition:** MEDIUM among data engineers.
- **Audience:** Medium-large. Modern data orchestrator, growing fast.
- **Visual DNA:**
  - Background: dark mode default (#1C1F25 range, very dark blue-gray)
  - Nodes: dark surface (#252830), rounded rectangles
  - Accent: purple/violet (#7C3AED range) for Dagster brand
  - compute_kind tags: small colored pills below asset names
  - Node border: subtle, dark
  - Font: Inter or similar modern sans-serif
  - Edges: thin, muted, curved
  - Health overlays: green/yellow/red status indicators
  - Overall feel: dark, modern, data-platform-native
- **Dagua implementation:** Dark theme with purple accents. Similar to Linear
  but with purple instead of indigo and data-engineering associations.
- **Priority: MEDIUM** -- solid dark theme, but overlaps with Linear.

### 8. AWS Step Functions
- **Recognition:** MEDIUM-HIGH among cloud/AWS users.
- **Audience:** Large. Major cloud workflow service.
- **Visual DNA:**
  - Background: white (#FFFFFF) or light gray
  - Nodes: rounded rectangles with AWS service icons
  - State type colors:
    - Task states: blue (#2196F3 area)
    - Choice/branch: orange (#FF9900 -- AWS orange)
    - Parallel: teal/cyan
    - Pass/Wait: gray
  - AWS orange (#FF9900) is the dominant brand accent
  - Edges: straight lines with right-angle routing (orthogonal)
  - Overall feel: enterprise, structured, AWS-branded
- **Dagua implementation:** AWS orange accent, orthogonal routing, blue/gray
  nodes. The icon-heavy nature is harder to replicate.
- **Priority: MEDIUM** -- recognizable but very tied to AWS branding.

### 9. Whimsical
- **Recognition:** MEDIUM among designers/PMs.
- **Audience:** Medium. Popular diagramming tool.
- **Visual DNA:**
  - Background: white or soft off-white
  - Nodes: clean rounded rectangles, pastel fills
  - Color palette: soft, muted pastels -- light blue, light green, light
    purple, light pink, light yellow
  - Borders: thin, slightly darker than fill
  - Font: clean sans-serif, likely custom
  - Edges: smooth curves, thin, subtle gray
  - Overall feel: polished, friendly, "designer-grade wireframes"
- **Dagua implementation:** Pastel palette, thin borders, smooth edges.
  Achievable but not radically different from Excalidraw minus the hand-drawn.
- **Priority: MEDIUM** -- pleasant but less distinctive than others.

### 10. Lucidchart
- **Recognition:** MEDIUM-HIGH in enterprise/business context.
- **Audience:** Large. Major diagramming SaaS.
- **Visual DNA:**
  - Background: white with light grid
  - Nodes: standard flowchart shapes (rectangles, diamonds, ovals)
  - Default fill: white or light blue (#E3F2FD range)
  - Borders: medium-weight, blue (#2196F3) or gray
  - Font: gray text, sans-serif
  - Themes: multiple built-in (corporate blue, colorful, minimal)
  - Edges: straight or orthogonal, blue or gray, various arrowheads
  - Overall feel: "corporate presentation diagram"
- **Dagua implementation:** Blue/gray corporate palette. Very standard.
- **Priority: LOW** -- too generic, not distinctive enough for "oh that's
  Lucidchart" recognition.

---

## Tier 3: Known Product but Not a Distinctive Graph Aesthetic

### 11. Notion
- **Recognizable graph aesthetic?** NO.
- Notion doesn't have a native graph view. Database relations exist but
  visualization requires third-party tools (IVGraph). No distinctive
  graph rendering style to capture.
- **Priority: SKIP**

### 12. Figma / FigJam
- **Recognizable graph aesthetic?** PARTIAL.
- FigJam has flowcharts but they're user-styled. Default connectors are
  gray, shapes are colorable. No single "FigJam look" -- it's a canvas tool.
  FigJam sticky notes (pastel yellow, blue, green, pink) are recognizable
  but that's sticky notes, not graphs.
- **Priority: LOW** -- too generic as a graph theme.

### 13. Slack Workflow Builder
- **Recognizable graph aesthetic?** NO.
- Slack's workflow builder is a simple linear step list, not a visual graph.
  No distinctive node/edge rendering.
- **Priority: SKIP**

### 14. Zapier
- **Recognizable graph aesthetic?** WEAK.
- Zapier's visual editor shows a linear step sequence. Zapier Canvas
  (diagramming) is newer and user-styled. The brand uses orange (#FF4A00)
  but the graph rendering isn't distinctive.
- **Priority: SKIP**

### 15. Terraform
- **Recognizable graph aesthetic?** NO (as rendered).
- `terraform graph` outputs DOT format -- the visual output IS Graphviz.
  Third-party tools (Blast Radius, Pluralith) have their own styles but
  those are the third-party tool's aesthetic, not Terraform's.
- **Priority: SKIP**

### 16. Kubernetes
- **Recognizable graph aesthetic?** NO single one.
- K8s topology is rendered by many different dashboards (Kiali, Lens,
  KubeSphere, Grafana). No single "Kubernetes graph look." Status-based
  coloring (green/yellow/red) is universal, not K8s-specific.
- **Priority: SKIP**

### 17. Datadog
- **Recognizable graph aesthetic?** WEAK.
- Service maps use health-colored ring borders (green/yellow/red) and
  purple for inferred services. The Datadog purple (#632CA6) brand is
  recognizable, but the service map itself looks like many monitoring tools.
- **Priority: LOW**

### 18. Grafana
- **Recognizable graph aesthetic?** WEAK for graphs/topology.
- Grafana is known for time-series dashboards (dark theme, green/orange
  lines). Their service graph panel exists but isn't a signature visual.
  The dark theme (#111217 bg, #FF9830 orange accent) is recognizable as
  "monitoring dashboard" but not as a graph layout.
- **Priority: LOW** -- more of a dashboard aesthetic than a graph one.

### 19. Prefect
- **Recognizable graph aesthetic?** WEAK.
- Prefect uses flow run visualizations with status colors but the graph
  rendering itself isn't distinctive enough to identify.
- **Priority: SKIP**

### 20. GitLab CI
- **Recognizable graph aesthetic?** WEAK.
- Pipeline visualization shows stages as columns with connected job nodes.
  Uses GitLab orange (#FC6D26) brand color. The visualization is functional
  but not visually distinctive -- looks like generic CI pipeline rendering.
- **Priority: LOW**

### 21. Luigi
- **Recognizable graph aesthetic?** NO.
- Luigi's web visualizer uses basic colored dots (green=complete,
  yellow=pending, blue=running) on a simple graph. Very bare-bones D3
  force layout. Not distinctive.
- **Priority: SKIP**

### 22. Argo Workflows
- **Recognizable graph aesthetic?** WEAK.
- DAG visualization with status-colored nodes on white background.
  Functional but not visually distinctive. No dark mode as of recent checks.
- **Priority: SKIP**

### 23. Google Cloud Workflows
- **Recognizable graph aesthetic?** NO.
- Basic workflow visualization in Cloud Console. Nothing distinctive.
- **Priority: SKIP**

### 24. Retool / Appsmith
- **Recognizable graph aesthetic?** NO.
- These are app builders, not graph tools. Retool's workflow builder is
  a tree layout but it's not visually distinctive enough to identify.
- **Priority: SKIP**

---

## Prioritized Implementation List

Ranked by: (distinctiveness * recognition * audience_size * implementability)

### MUST IMPLEMENT (instant recognition, huge audience)

1. **`excalidraw`** -- Hand-drawn aesthetic. White bg, pastel Open Color fills,
   near-black strokes, hand-drawn font (Virgil/Comic Shanns). Sketchy edges.
   THE most recognizable diagram style on the internet. Implementation notes:
   rough stroke simulation is the key challenge; without it, use thin dark
   strokes + pastel fills + hand-drawn font for 80% of the effect.

2. **`github`** -- GitHub Actions / Primer. Both light (#FFFFFF bg) and dark
   (#0D1117 bg) variants. Status-colored accents (green/red/yellow/gray) on
   neutral nodes. Clean, functional. Every developer knows this instantly.
   Implementation: two sub-themes (`github-light`, `github-dark`).

3. **`linear`** -- Ultra-dark minimal. #0F0F10 bg, #5E6AD2 indigo accent,
   #EEEFF1 text, dark surface nodes (#1E1E20), barely-visible borders.
   "Linear-style" is now a recognized design movement in SaaS. Modern,
   premium feel.

### SHOULD IMPLEMENT (strong recognition in target audience)

4. **`n8n`** -- Workflow node editor. Light gray dotted grid bg, white rounded
   nodes with colored left accent stripes, gray bezier edges, connection
   port dots. Distinctive node-editor aesthetic. #EA4B71 pink for highlights.

5. **`airflow`** -- Rainbow DAG. White bg, operator-type colored nodes (each
   type gets a distinct pastel), status overlay colors, straight/polyline
   edges. Recognizable "data pipeline" aesthetic.

6. **`dagster`** -- Dark data platform. Similar to Linear but with purple
   (#7C3AED) accent instead of indigo. Dark blue-gray bg. Good complement
   to the Linear theme for data-engineering audience.

### NICE TO HAVE (less distinctive but still useful)

7. **`dbt`** -- Data lineage. White bg, green sources, blue models,
   left-to-right flow. Recognizable to data professionals.

8. **`aws-step-functions`** -- Enterprise workflow. AWS orange (#FF9900)
   accent, blue task nodes, orthogonal edge routing. Recognizable to
   AWS users.

9. **`whimsical`** -- Polished pastels. Clean minimal, soft muted colors,
   thin borders. Pleasant but less distinctive than Excalidraw.

---

## Quick Reference: Key Colors

| Theme | Background | Node Fill | Node Border | Accent | Text | Edge |
|-------|-----------|-----------|-------------|--------|------|------|
| excalidraw | #FFFFFF | pastel washes | #1e1e1e | (per-color) | #1e1e1e | #1e1e1e |
| github-light | #FFFFFF | #FFFFFF | #D1D5DA | #1A7F37 | #1F2328 | #656D76 |
| github-dark | #0D1117 | #161B22 | #30363D | #3FB950 | #E6EDF3 | #484F58 |
| linear | #0F0F10 | #1E1E20 | #2A2A2E | #5E6AD2 | #EEEFF1 | #3A3A3E |
| n8n | #F5F5F5 | #FFFFFF | #DBDBDB | #EA4B71 | #333333 | #B0B0B0 |
| airflow | #FFFFFF | per-type | per-type | #017CEE | #333333 | #999999 |
| dagster | #1C1F25 | #252830 | #2E3138 | #7C3AED | #E0E0E0 | #3A3E48 |
| dbt | #FFFFFF | #E8F4E8 | #22C55E | #FF6600 | #333333 | #CCCCCC |
| aws-sfn | #FFFFFF | #E3F2FD | #2196F3 | #FF9900 | #333333 | #999999 |
| whimsical | #FAFAFA | #EEF2FF | #C7D2E8 | #6366F1 | #374151 | #D1D5DB |

## Fonts by Theme

| Theme | Primary Font | Fallback |
|-------|-------------|----------|
| excalidraw | Virgil / Comic Shanns / Excalifont | cursive |
| github | -apple-system, Segoe UI | system sans-serif |
| linear | Inter | SF Pro, system sans-serif |
| n8n | Inter | system sans-serif |
| airflow | Roboto | system sans-serif |
| dagster | Inter | system sans-serif |
| dbt | Inter | system sans-serif |
| aws-sfn | Amazon Ember | system sans-serif |
| whimsical | (custom sans) | system sans-serif |

## Edge Styles by Theme

| Theme | Routing | Weight | Arrowheads |
|-------|---------|--------|------------|
| excalidraw | hand-drawn curves | thin (~1px) | simple filled triangle |
| github | straight + rounded corners | medium (~1.5px) | none (lines between jobs) |
| linear | subtle curves | very thin (~0.5px) | minimal or none |
| n8n | smooth bezier | thin (~1px) | small or none, port dots instead |
| airflow | straight/polyline | medium (~1.5px) | small triangle |
| dagster | subtle curves | thin (~1px) | small or none |
| dbt | straight | thin (~1px) | small triangle |
| aws-sfn | orthogonal (90deg) | medium (~1.5px) | filled triangle |
| whimsical | smooth curves | thin (~1px) | small filled triangle |

---

## Implementation Notes

1. **Font availability:** Virgil/Excalifont are OFL-licensed and freely
   available. Inter is freely available. Others may need fallbacks.

2. **Excalidraw rough strokes:** Full roughjs-style rendering would require
   path perturbation in the renderer. A simpler approach: use the hand-drawn
   font + pastel fills + thin dark strokes to get 80% of the feel.

3. **n8n port dots:** The connection port indicators (small circles at node
   edges) are a distinctive n8n element. Could be implemented as small
   circular decorations at edge endpoints.

4. **Grid backgrounds:** n8n's dotted grid and Lucidchart's light grid could
   be an optional style property on GraphStyle.

5. **Status color mapping:** GitHub/Airflow/Dagster themes benefit from a
   way to map node categories/states to colors. This aligns with dagua's
   existing category-to-color palette system.

6. **Theme naming:** Use lowercase, no prefix. `dagua.set_theme('excalidraw')`
   reads naturally. For variants: `github-light`, `github-dark`.
