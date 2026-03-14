"""Classic educational graph layout algorithms."""

from dagua.layout.classic.fa2 import layout_fa2
from dagua.layout.classic.fr import layout_fr
from dagua.layout.classic.kk import layout_kk
from dagua.layout.classic.stress_sgd import layout_stress_sgd
from dagua.layout.classic.sugiyama import layout_sugiyama

__all__ = ["layout_fa2", "layout_fr", "layout_kk", "layout_stress_sgd", "layout_sugiyama"]
