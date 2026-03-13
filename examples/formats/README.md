# Annotated Graph Spec Examples

These files are deliberately verbose, human-readable examples of the supported
graph data formats.

Policy:
- YAML is the default for hand-authored specs
- JSON is the default for generated or machine-facing artifacts

- `annotated_graph.yaml`
  - best place to learn the structure
  - uses real YAML comments generously
- `annotated_graph.json`
  - same overall shape in JSON form
  - uses `_comment` keys because JSON does not support real comments

Both files are meant to be instructional reference artifacts, not minimal specs.
