"""Sprint 0.5: enforce held-out opacity.

The held-out suite is salt-derived and regenerated on demand. `dagua/graphs/
holdout/` may contain ONLY:
- `MANIFEST.json` (topology hashes, family, size; no raw edges).
- `.opaque` marker file (enforces intent).

Any other file (raw topology JSON, .pt tensor, pickled graph) is a process
failure and this test fails CI. Per 03_test_matrix.md "Held-out suite
(fixed, at least 30 graphs -- OPAQUE, NEVER iterated against)".
"""

from __future__ import annotations

from pathlib import Path

import pytest

HOLDOUT_DIR = Path(__file__).resolve().parent.parent / "dagua" / "graphs" / "holdout"
ALLOWED_FILES = frozenset({"MANIFEST.json", ".opaque"})


@pytest.mark.unit
def test_holdout_directory_contains_only_manifest_and_marker():
    assert HOLDOUT_DIR.is_dir(), f"Expected held-out dir at {HOLDOUT_DIR}"
    entries = {p.name for p in HOLDOUT_DIR.iterdir()}
    unexpected = entries - ALLOWED_FILES
    assert not unexpected, (
        f"Held-out opacity violation: unexpected files in {HOLDOUT_DIR}: "
        f"{sorted(unexpected)}. Only MANIFEST.json and .opaque are allowed. "
        f"Graph tensors must be regenerated on demand from the secret salt; "
        f"committing them to git re-creates the overfit vector Sprint 0.5 closed."
    )


@pytest.mark.unit
def test_holdout_manifest_has_hashes_not_edges():
    manifest_path = HOLDOUT_DIR / "MANIFEST.json"
    text = manifest_path.read_text()
    # Simple substring checks: the manifest must not embed raw edge tensors.
    # (Hashes are fine; edge_index tensors would be huge and bypass opacity.)
    forbidden_keys = ("edge_index", "src_nodes", "tgt_nodes", "edges")
    for key in forbidden_keys:
        assert key not in text, (
            f"Held-out MANIFEST.json must not embed raw edge data "
            f"(found key '{key}'). Use topology_sha256_16 only."
        )


@pytest.mark.unit
def test_holdout_regeneration_is_deterministic():
    """Same salt + sprint_tag -> identical topology hashes."""
    from dagua.eval.graph_generator import make_holdout_suite

    _, m1 = make_holdout_suite()
    _, m2 = make_holdout_suite()
    hashes1 = [e["topology_sha256_16"] for e in m1.entries]
    hashes2 = [e["topology_sha256_16"] for e in m2.entries]
    assert hashes1 == hashes2, "Held-out suite not deterministic; salt read is broken"


@pytest.mark.unit
def test_holdout_rolling_never_collides():
    """Different sprint_tag -> different topology hashes."""
    from dagua.eval.graph_generator import make_holdout_suite, make_rolling_suite

    _, mh = make_holdout_suite()
    _, mr = make_rolling_suite(sprint_tag="sprint_smoke_test")
    holdout_hashes = {e["topology_sha256_16"] for e in mh.entries}
    rolling_hashes = {e["topology_sha256_16"] for e in mr.entries}
    # No rolling hash should collide with held-out. (Not impossible by
    # chance for small graphs, but extremely unlikely with sha256 prefixes.)
    collisions = holdout_hashes & rolling_hashes
    assert not collisions, (
        f"Rolling suite collided with held-out: {collisions}. Anti-overfit signal compromised."
    )


@pytest.mark.unit
def test_holdout_rolling_tag_prefix_enforced():
    """Rolling sprint_tag must NOT start with 'holdout_'."""
    from dagua.eval.graph_generator import make_rolling_suite

    with pytest.raises(ValueError, match="holdout_"):
        make_rolling_suite(sprint_tag="holdout_v2")
