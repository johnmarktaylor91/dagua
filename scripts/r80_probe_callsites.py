"""Attribute rgg_500's project_overlaps invocations to call sites.

Wraps ``project_overlaps`` with a stack-sampling counter and reruns the
dagua engine on rgg_500 so the r80 sweep regression can be attributed to
gated vs ungated projector call sites (bisection-first doctrine).

Usage: python scripts/r80_probe_callsites.py [graph ...]
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r79_baseline import SEED, TIMEOUT_SECONDS, build_corpus, get_competitor

from dagua.layout import projection as projection_module

call_sites: Counter[str] = Counter()

_orig_project_overlaps = projection_module.project_overlaps


def traced_project_overlaps(*args, **kwargs):
    """Record the nearest non-projection caller frame, then delegate."""
    for frame in reversed(traceback.extract_stack()[:-1]):
        if "projection.py" in frame.filename or "probe_callsites" in frame.filename:
            continue
        call_sites[f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}"] += 1
        break
    return _orig_project_overlaps(*args, **kwargs)


projection_module.project_overlaps = traced_project_overlaps
# Rebind in modules that imported the symbol directly.
from dagua.layout import engine as engine_module  # noqa: E402
from dagua.layout.ops import project as ops_project_module  # noqa: E402

engine_module.project_overlaps = traced_project_overlaps
ops_project_module.project_overlaps = traced_project_overlaps


def main() -> int:
    """Run the call-site attribution probe.

    Returns
    -------
    int
        Process exit status.
    """
    targets = sys.argv[1:] or ["rgg_500"]
    corpus = {g.name: g for g in build_corpus()}
    competitor = get_competitor("dagua")
    for name in targets:
        call_sites.clear()
        t0 = time.perf_counter()
        competitor.layout(corpus[name].graph, timeout=TIMEOUT_SECONDS, seed=SEED)
        elapsed = time.perf_counter() - t0
        print(f"{name}: layout {elapsed:.1f}s | project_overlaps call sites:", flush=True)
        for site, count in call_sites.most_common():
            print(f"  {count:4d}x {site}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
