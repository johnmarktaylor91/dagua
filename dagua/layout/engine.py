"""Core optimization loop — the heart of dagua (~200 lines of PyTorch).

Takes edge_index, node_sizes, groups as tensors (NOT a Graph object).
Initializes positions as learnable parameters, runs optimization with
composite loss from constraints, returns detached position tensor.

Headless design: independently testable without the Graph abstraction.
"""
