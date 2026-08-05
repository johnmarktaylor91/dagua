"""DEPRECATED: classic-to-pipeline fidelity validator (dead gate).

This script compared the monolithic classic reimplementations in
``dagua.layout.classic`` against their op-pipeline translations in
``dagua.layout.ops.pipelines``. It has been retired because the
post-consolidation code base broke every one of its assumptions:

- ``_CLASSIC_LAYOUT_SPECS`` import paths now point at
  ``dagua.layout.ops.pipelines.*``, so its string-replace resolution
  compared each pipeline callable against ITSELF (zero fidelity signal);
- before that mattered, it crashed at startup: the ``layout_*_pipeline``
  glob finds two callables per module (e.g. ``layout_fr_pipeline`` and
  ``layout_fr_default_pipeline``), and the hardcoded 105-graph corpus
  expectation no longer holds (the corpus has 129 graphs);
- the classic and pipeline callables have since diverged in both signature
  (``area`` vs ``k``/``networkx_compat``/``fidelity_mode``) and output, so
  the original exact-equality contract is unrecoverable without redesign.

Use the live fidelity gate instead:

    python scripts/compare_reimpl_vs_original.py --help

which compares reimplementation engines against their external references
via the current registry layout. The full historical implementation of
this validator is preserved in git history (see this file before the
deprecation commit).

This stub exits nonzero so any automation still invoking it fails loudly
instead of mistaking a dead gate for a pass.
"""

from __future__ import annotations

import sys

DEPRECATION_MESSAGE = (
    "DEPRECATED: scripts/validate_pipeline_fidelity.py is a retired dead gate "
    "(self-vs-self comparison after the classic->pipelines consolidation; see "
    "module docstring). Use scripts/compare_reimpl_vs_original.py, the live "
    "reimplementation-fidelity gate."
)


def main() -> int:
    """Print the deprecation pointer and exit nonzero.

    Returns
    -------
    int
        Always ``2``: a retired gate must never look like a pass.
    """
    print(DEPRECATION_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
