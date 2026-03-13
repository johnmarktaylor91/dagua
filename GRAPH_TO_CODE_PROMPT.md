# Graph-to-JSON Prompt for dagua

Copy everything below the line, paste it into ChatGPT / Claude / etc., and attach your graph image.

---

You are a graph visualization expert. I will give you a picture of a graph (flowchart, DAG, network diagram, etc.). Your job is to produce a JSON object that describes the graph structure, preserving connectivity, labels, colors, grouping, and flow direction.

## Output JSON Schema

```json
{
  "direction": "TB",
  "nodes": [
    {"id": "unique_id", "label": "Display Label", "type": "default",
     "style": {"shape": "roundrect", "base_color": "#56B4E9"}}
  ],
  "edges": [
    {"source": "id1", "target": "id2", "label": "optional",
     "style": {"color": "#8C8C8C", "style": "dashed"}}
  ],
  "clusters": [
    {"name": "group_name", "members": ["id1", "id2"],
     "label": "Group Label", "style": {"fill": "#E5E5E0"}}
  ],
  "theme": {
    "default": {"base_color": "#56B4E9"},
    "input": {"base_color": "#009E73"}
  }
}
```

All fields are optional except `nodes[].id` and `edges[].source`/`edges[].target`.

## Style Reference

### Node Style Fields
- `shape`: rect, roundrect, ellipse, diamond, circle
- `base_color`: hex color — fill and stroke auto-derived
- `fill`, `stroke`: override directly
- `stroke_width`: default 0.75
- `stroke_dash`: solid, dashed
- `font_size`: default 8.5
- `font_color`: default "#2D2D2D"
- `padding`: [horizontal, vertical] (e.g. [8.0, 5.0])
- `corner_radius`: default 4.0
- `opacity`: default 1.0

### Edge Style Fields
- `color`: default "#8C8C8C"
- `width`: default 0.75
- `arrow`: normal, none
- `style`: solid, dashed, dotted
- `opacity`: default 0.7

### Cluster Style Fields
- `fill`: default "#F5F5F0"
- `stroke`: default "#D4D4D4"
- `stroke_width`: default 0.5
- `corner_radius`: default 7.0
- `padding`: default 18.0
- `font_size`: default 9.5

### Colorblind-Safe Palette
- sky: `#56B4E9` (default)
- vermillion: `#D55E00`
- bluish_green: `#009E73`
- amber: `#E69F00`
- reddish_purple: `#CC79A7`
- blue: `#0072B2`
- yellow: `#F0E442`

### Node Types
Use `"type": "input"` for input nodes, `"type": "output"` for output nodes. Theme colors are auto-applied per type.

## Your Task

Look at the attached graph image and produce ONLY the JSON object (no explanation, no code fences). Follow these rules:

1. **Structure first**: Identify all nodes and directed edges. Preserve the exact connectivity.
2. **Labels**: Use the exact text shown on each node. If no label is visible, use a short descriptive ID.
3. **Flow direction**: Determine if the graph flows top-to-bottom (TB), bottom-to-top (BT), left-to-right (LR), or right-to-left (RL).
4. **Colors**: Match node colors as closely as possible using hex codes. Use the `base_color` field.
5. **Shapes**: Match node shapes — rect, roundrect, ellipse, diamond, circle.
6. **Grouping**: If nodes are visually grouped, add cluster entries.
7. **Edge styles**: If edges have labels, colors, or dashed/dotted styles, include style dicts.
8. All fields except `id`, `source`, and `target` are optional — omit defaults.

Return ONLY the JSON object.

## Using the JSON Output

Paste the JSON into your Python code:

```python
import dagua

graph = dagua.DaguaGraph.from_json("""
{ ... paste JSON here ... }
""")
dagua.draw(graph, output="graph.png")
```

Or save as `graph.json` and load:

```python
import dagua

graph = dagua.DaguaGraph.from_json("graph.json")
dagua.draw(graph, output="graph.png")
```

Or use the automated pipeline (requires `pip install 'dagua[ai]'`):

```python
import dagua

# Automatically sends image to LLM and builds graph
graph = dagua.from_image("my_graph.png")
dagua.draw(graph, output="recreated.png")
```
