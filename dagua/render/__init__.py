"""Render subpackage — multiple output backends.

Re-exports: render(), to_svg().
Renderers accept structured data (positions, elements, styles), not Graph objects.
Graph.render() is a thin wrapper that extracts data and calls the appropriate renderer.
"""
