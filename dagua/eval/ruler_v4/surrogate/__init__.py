"""Manifest-compiled analytic surrogate for RULER V4."""

from dagua.eval.ruler_v4.surrogate.manifest import (
    CompilationRule,
    CompiledSurrogateManifest,
    SurrogateTermTrace,
    compile_surrogate_manifest,
)
from dagua.eval.ruler_v4.surrogate.scorer import (
    SoftScoreResult,
    SoftTermResult,
    score_v4_soft,
)

__all__ = [
    "CompilationRule",
    "CompiledSurrogateManifest",
    "SurrogateTermTrace",
    "SoftScoreResult",
    "SoftTermResult",
    "compile_surrogate_manifest",
    "score_v4_soft",
]
