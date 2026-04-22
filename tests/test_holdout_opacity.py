"""Sprint 0.5: enforce held-out opacity.

The held-out suite is salt-derived and regenerated on demand. `dagua/graphs/
holdout/` may contain ONLY:
- `MANIFEST.json` (topology hashes, family, size; no raw edges, no seeds).
- `.opaque` marker file (enforces intent).

Any other file (raw topology JSON, .pt tensor, pickled graph) is a process
failure and this test fails CI. Per 03_test_matrix.md "Held-out suite
(fixed, at least 30 graphs -- OPAQUE, NEVER iterated against)".

Round-2 adversarial review (2026-04-22) caught that embedding `seed` in the
manifest broke opacity because seeds reconstruct the graph via
`_FAMILY_SPECS[family](target_n, seed)`. These tests now:
- enforce an EXACT manifest-entry schema (allowlist);
- forbid any field that could reconstruct the topology (seed, edges, etc.);
- scan the raw manifest text for reconstructable payload patterns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

HOLDOUT_DIR = Path(__file__).resolve().parent.parent / "dagua" / "graphs" / "holdout"
ALLOWED_FILES = frozenset({"MANIFEST.json", ".opaque"})

# Exact allowed keys per manifest entry. Any addition is an opacity decision.
ALLOWED_ENTRY_KEYS = frozenset(
    {
        "index",
        "family",
        "target_n",
        "actual_n",
        "actual_e",
        "topology_sha256_10",
    }
)

# Any of these keys appearing in a manifest entry is an opacity bug.
FORBIDDEN_ENTRY_KEYS = frozenset(
    {
        "seed",  # would allow regeneration via _FAMILY_SPECS[family](n, seed)
        "edge_index",
        "edges",
        "src_nodes",
        "tgt_nodes",
        "adjacency",
        "graph",
        "graph_pickle",
        "base64",
    }
)


@pytest.mark.unit
def test_holdout_directory_contains_only_manifest_and_marker():
    assert HOLDOUT_DIR.is_dir(), f"Expected held-out dir at {HOLDOUT_DIR}"
    entries = {p.name for p in HOLDOUT_DIR.iterdir()}
    unexpected = entries - ALLOWED_FILES
    assert not unexpected, (
        f"Held-out opacity violation: unexpected files in {HOLDOUT_DIR}: "
        f"{sorted(unexpected)}. Only MANIFEST.json and .opaque are allowed."
    )


@pytest.mark.unit
def test_holdout_manifest_schema_is_allowlisted():
    """Every manifest entry has exactly the allowed keys; no forbidden fields."""
    manifest = json.loads((HOLDOUT_DIR / "MANIFEST.json").read_text())
    assert "entries" in manifest, "Manifest missing 'entries'"
    assert manifest["entries"], "Manifest entries empty"
    for e in manifest["entries"]:
        keys = set(e.keys())
        missing = ALLOWED_ENTRY_KEYS - keys
        extra = keys - ALLOWED_ENTRY_KEYS
        assert not missing, f"Entry index={e.get('index')} missing required keys: {missing}"
        assert not extra, (
            f"Entry index={e.get('index')} has unexpected keys: {extra}. "
            f"Add to ALLOWED_ENTRY_KEYS only after opacity review."
        )
        forbidden_present = keys & FORBIDDEN_ENTRY_KEYS
        assert not forbidden_present, (
            f"OPACITY BREACH: entry index={e.get('index')} contains "
            f"reconstructable fields {forbidden_present}. These must never "
            f"be committed -- seeds / edge data reconstruct the graph without "
            f"the secret salt."
        )


@pytest.mark.unit
def test_holdout_manifest_raw_text_has_no_reconstructable_payload():
    """Defensive scan of the raw text for patterns that could reconstruct graphs."""
    text = (HOLDOUT_DIR / "MANIFEST.json").read_text()
    for key in FORBIDDEN_ENTRY_KEYS:
        assert f'"{key}"' not in text, (
            f"Held-out MANIFEST.json must not embed '{key}' anywhere "
            f"(found in raw text). Opacity breach."
        )


@pytest.mark.unit
def test_holdout_regeneration_is_deterministic():
    """Same salt + sprint_tag -> identical topology hashes."""
    from dagua.eval.graph_generator import make_holdout_suite

    _, m1 = make_holdout_suite()
    _, m2 = make_holdout_suite()
    hashes1 = [e["topology_sha256_10"] for e in m1.entries]
    hashes2 = [e["topology_sha256_10"] for e in m2.entries]
    assert hashes1 == hashes2, "Held-out suite not deterministic; salt read is broken"


@pytest.mark.unit
def test_holdout_rolling_never_collides():
    """Different sprint_tag -> different topology hashes."""
    from dagua.eval.graph_generator import make_holdout_suite, make_rolling_suite

    _, mh = make_holdout_suite()
    _, mr = make_rolling_suite(sprint_tag="sprint_smoke_test")
    holdout_hashes = {e["topology_sha256_10"] for e in mh.entries}
    rolling_hashes = {e["topology_sha256_10"] for e in mr.entries}
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


@pytest.mark.unit
def test_holdout_tag_prefix_enforced():
    """Holdout sprint_tag MUST start with 'holdout_'. Symmetric with rolling."""
    from dagua.eval.graph_generator import make_holdout_suite

    with pytest.raises(ValueError, match="holdout_"):
        make_holdout_suite(sprint_tag="sprint_1_smoke")


@pytest.mark.unit
def test_holdout_manifest_entries_match_actual_generation():
    """The committed manifest hashes match what the generator actually produces now.

    Guards against salt rotation or generator code change without regenerating
    the committed manifest.
    """
    from dagua.eval.graph_generator import make_holdout_suite

    _, regen = make_holdout_suite()
    committed = json.loads((HOLDOUT_DIR / "MANIFEST.json").read_text())

    regen_hashes = [e["topology_sha256_10"] for e in regen.entries]
    committed_hashes = [e["topology_sha256_10"] for e in committed["entries"]]
    assert regen_hashes == committed_hashes, (
        "Committed MANIFEST.json is stale; regenerate after salt rotation or generator change."
    )
