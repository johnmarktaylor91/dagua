"""Public API for data-coordinate node and cluster borders."""

from dagua.render.borders.annular import annular_path, reverse_closed_path
from dagua.render.borders.collection import add_filled_collections, make_clip_proxy
from dagua.render.borders.dashes import PolylineDashSegment, dash_ribbon_paths, dash_segments
from dagua.render.borders.inset import (
    MAX_BORDER_FRACTION,
    clamp_border_width,
    inset_convex_polygon,
    inset_shape_path,
    inset_star,
)
from dagua.render.borders.shapes import (
    ShapeSpec,
    add_corner_radius,
    build_shape_path,
    closed_path_from_vertices,
    cylinder_path,
    extract_patch_path,
    path_to_closed_vertices,
    polygon_vertices,
    regular_polygon_vertices,
    scale_corner_radius,
    star_vertices,
    triangle_vertices,
)

__all__ = [
    "MAX_BORDER_FRACTION",
    "PolylineDashSegment",
    "ShapeSpec",
    "add_filled_collections",
    "add_corner_radius",
    "annular_path",
    "build_shape_path",
    "clamp_border_width",
    "closed_path_from_vertices",
    "cylinder_path",
    "dash_ribbon_paths",
    "dash_segments",
    "extract_patch_path",
    "inset_convex_polygon",
    "inset_shape_path",
    "inset_star",
    "path_to_closed_vertices",
    "polygon_vertices",
    "regular_polygon_vertices",
    "scale_corner_radius",
    "reverse_closed_path",
    "star_vertices",
    "triangle_vertices",
    "make_clip_proxy",
]
