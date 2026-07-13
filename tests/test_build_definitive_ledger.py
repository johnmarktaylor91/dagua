"""Tests for scripts/build_definitive_ledger.py.

Pure post-processing tests on tiny synthetic fixtures: every tier gate, every
hard rule, and a 30-row smoke run on the real r77 jsonl when present. No layout
engines, no heavy imports; runs in well under 10 seconds.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_definitive_ledger.py"
R77_JSONL = REPO / "eval_output" / "fidelity_definitive" / "per_combo_r77.jsonl"

spec = importlib.util.spec_from_file_location("build_definitive_ledger", SCRIPT)
bdl = importlib.util.module_from_spec(spec)
sys.modules["build_definitive_ledger"] = bdl
spec.loader.exec_module(bdl)


# ---------------------------------------------------------------------------
# Fixture row builders
# ---------------------------------------------------------------------------


def mode_a_row(graph="g", engine="e", **over):
    row = {
        "combo_id": f"{graph}::{engine}",
        "graph": graph,
        "engine": engine,
        "mode": "A",
        "insufficient_data": False,
        "insufficient_reason": None,
        "no_canonical_reference": False,
        "matched_seeds": list(range(100)),
        "mean_diag_B": 0.05,
        "mean_B_offdiag": 0.1,
        "dist_equivalent": None,
        "d_R": None,
        "near_deterministic": False,
        "quality_identical_raw": False,
        "quality_identical_exploratory": False,
        "quality_equivalent_raw": False,
        "quality_superior_distinct": False,
        "git_sha": "deadbeef",
    }
    row.update(over)
    row["combo_id"] = f"{row['graph']}::{row['engine']}"
    return row


def mode_b_row(graph="g", engine="e", d_R=1e-12, **over):
    row = mode_a_row(
        graph=graph, engine=engine, mode="B", d_R=d_R, matched_seeds=[], mean_diag_B=None
    )
    row.update(over)
    row["combo_id"] = f"{row['graph']}::{row['engine']}"
    return row


def tier_of(row, prior=None, cause=None):
    return bdl.assign_tier(row, prior, cause)


# ---------------------------------------------------------------------------
# Tier gates (one test per tier)
# ---------------------------------------------------------------------------


def test_mode_b_bit_exact():
    v = tier_of(mode_b_row(d_R=5e-13))
    assert v["tier"] == "MODE_B_BIT_EXACT"


def test_mode_b_identical_distance():
    assert tier_of(mode_b_row(d_R=5e-3))["tier"] == "MODE_B_IDENTICAL_DISTANCE"


def test_mode_b_close():
    assert tier_of(mode_b_row(d_R=5e-2))["tier"] == "MODE_B_CLOSE"


def test_positional_identical():
    v = tier_of(mode_a_row(mean_diag_B=5e-4, dist_equivalent=True))
    assert v["tier"] == "POSITIONAL_IDENTICAL"


def test_distributional_equivalent():
    v = tier_of(mode_a_row(mean_diag_B=0.2, dist_equivalent=True))
    assert v["tier"] == "DISTRIBUTIONAL_EQUIVALENT"


def test_quality_equivalent_confirmatory():
    v = tier_of(mode_a_row(mean_diag_B=0.2, dist_equivalent=None, quality_identical_raw=True))
    assert v["tier"] == "QUALITY_EQUIVALENT"
    assert v["exploratory"] is False


def test_quality_equivalent_exploratory():
    v = tier_of(
        mode_a_row(mean_diag_B=0.2, dist_equivalent=None, quality_identical_exploratory=True)
    )
    assert v["tier"] == "QUALITY_EQUIVALENT"
    assert v["exploratory"] is True


def test_superior_distinct():
    v = tier_of(mode_a_row(mean_diag_B=0.4, dist_equivalent=None, quality_superior_distinct=True))
    assert v["tier"] == "SUPERIOR_DISTINCT"


def test_divergent_named_cause():
    v = tier_of(mode_a_row(mean_diag_B=0.4, dist_equivalent=None), cause="known packing residual")
    assert v["tier"] == "DIVERGENT_NAMED_CAUSE"
    assert v["named_cause"] == "known packing residual"


def test_divergent_unexplained():
    v = tier_of(mode_a_row(mean_diag_B=0.4, dist_equivalent=None))
    assert v["tier"] == "DIVERGENT_UNEXPLAINED"


def test_insufficient_data():
    v = tier_of(mode_a_row(insufficient_data=True, insufficient_reason="matched_seeds_lt_30"))
    assert v["tier"] == "INSUFFICIENT_DATA"


def test_no_canonical_reference():
    # jsonl flag wins even when the row would otherwise be top-tier.
    v = tier_of(mode_a_row(no_canonical_reference=True, mean_diag_B=1e-6, dist_equivalent=True))
    assert v["tier"] == "NO_CANONICAL_REFERENCE"


def test_r77_carried_routings():
    # Insufficient rows carry AGGREGATE_INSUFFICIENT / NO_CANONICAL priors.
    v = tier_of(mode_a_row(insufficient_data=True), prior="AGGREGATE_INSUFFICIENT")
    assert v["tier"] == "AGGREGATE_INSUFFICIENT"
    v = tier_of(mode_a_row(insufficient_data=True), prior="NO_CANONICAL_REFERENCE")
    assert v["tier"] == "NO_CANONICAL_REFERENCE"
    # Stale-reference carry (sufficient data in jsonl).
    v = tier_of(mode_a_row(dist_equivalent=False), prior="AGGREGATE_STALE_REFERENCE")
    assert v["tier"] == "AGGREGATE_STALE_REFERENCE"


# ---------------------------------------------------------------------------
# Hard rules
# ---------------------------------------------------------------------------


def test_rule_a_dist_equivalent_false_never_above_quality():
    # Attempted promotion: passes the positional-identical bar AND claims
    # quality-identical, but the cloud test failed -> capped at QUALITY_EQUIVALENT.
    v = tier_of(mode_a_row(mean_diag_B=1e-6, dist_equivalent=False, quality_identical_raw=True))
    assert v["tier"] == "QUALITY_EQUIVALENT"
    # Without any quality pass it must go divergent, never positional/distributional.
    v = tier_of(mode_a_row(mean_diag_B=1e-6, dist_equivalent=False))
    assert v["tier"] == "DIVERGENT_UNEXPLAINED"


def test_rule_b_deterministic_large_distance_needs_named_cause():
    # Mode B with large d_R: benign tiers forbidden.
    v = tier_of(mode_b_row(d_R=0.7))
    assert v["tier"] == "DIVERGENT_UNEXPLAINED"
    v = tier_of(mode_b_row(d_R=0.7), cause="drand48 residual")
    assert v["tier"] == "DIVERGENT_NAMED_CAUSE"
    # Mode A near-deterministic with mean_diag_B > 0.5: same rule, even when
    # the mechanical cloud gate would pass.
    row = mode_a_row(
        near_deterministic=True, mean_diag_B=0.9, dist_equivalent=True, quality_identical_raw=True
    )
    v = tier_of(row)
    assert v["tier"] == "DIVERGENT_UNEXPLAINED"
    assert tier_of(row, cause="eigensign floor")["tier"] == "DIVERGENT_NAMED_CAUSE"


def test_rule_b_missing_d_r_mode_b_is_not_benign():
    v = tier_of(mode_b_row(d_R=None))
    assert v["tier"] == "DIVERGENT_UNEXPLAINED"
    assert tier_of(mode_b_row(d_R=float("nan")))["tier"] == "DIVERGENT_UNEXPLAINED"


def test_rule_c_insufficient_seeds_blocks_benign():
    # 20 matched seeds, otherwise perfect -> INSUFFICIENT_DATA, never benign.
    v = tier_of(mode_a_row(matched_seeds=list(range(20)), mean_diag_B=1e-6, dist_equivalent=True))
    assert v["tier"] == "INSUFFICIENT_DATA"


def test_anti_laundering_sticky_named_cause():
    # r77 adjudicated divergent; mechanically passes the cloud gate -> stays divergent.
    row = mode_a_row(mean_diag_B=0.6, dist_equivalent=True)
    v = tier_of(row, prior="DIVERGENT_NAMED_CAUSE", cause="MDS eigenspace floor")
    assert v["tier"] == "DIVERGENT_NAMED_CAUSE"


# ---------------------------------------------------------------------------
# End-to-end on synthetic data (covers rule (d), outputs, exit codes, stale map)
# ---------------------------------------------------------------------------


def synthetic_dataset():
    return [
        mode_b_row(engine="bitexact", d_R=1e-12),
        mode_b_row(engine="identdist", d_R=5e-3),
        mode_b_row(engine="close", d_R=5e-2),
        mode_b_row(engine="bigdist", d_R=0.9),  # named via sidecar below
        mode_a_row(engine="positional", mean_diag_B=1e-5, dist_equivalent=True),
        mode_a_row(engine="cloud", mean_diag_B=0.2, dist_equivalent=True),
        mode_a_row(engine="quality", mean_diag_B=0.2, quality_identical_raw=True),
        mode_a_row(engine="quality_expl", mean_diag_B=0.2, quality_identical_exploratory=True),
        mode_a_row(engine="superior", mean_diag_B=0.4, quality_superior_distinct=True),
        mode_a_row(engine="named", mean_diag_B=0.4),  # named via sidecar below
        mode_a_row(engine="unexplained", mean_diag_B=0.4),
        mode_a_row(
            engine="insufficient", insufficient_data=True, insufficient_reason="matched_seeds_lt_30"
        ),
        mode_a_row(engine="nocanon", no_canonical_reference=True),
        mode_a_row(
            engine="capped", mean_diag_B=1e-6, dist_equivalent=False, quality_identical_raw=True
        ),
        mode_a_row(engine="detbig", near_deterministic=True, mean_diag_B=0.9, dist_equivalent=True),
        mode_a_row(
            engine="fewseeds", matched_seeds=list(range(10)), mean_diag_B=1e-6, dist_equivalent=True
        ),
    ]


def write_inputs(tmp_path, rows, causes=None):
    jsonl = tmp_path / "per_combo.jsonl"
    with open(jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    causes_path = ""
    if causes is not None:
        causes_path = tmp_path / "causes.json"
        causes_path.write_text(json.dumps(causes))
        causes_path = str(causes_path)
    return str(jsonl), causes_path


def test_end_to_end_synthetic(tmp_path):
    rows = synthetic_dataset()
    causes = {
        "g::bigdist": "sugiyama rank-collapse residual",
        "g::named": "packing residual",
        "g::detbig": "eigensign floor",
    }
    jsonl, causes_path = write_inputs(tmp_path, rows, causes)
    out = tmp_path / "out"
    rc = bdl.run(
        [
            "--per-combo",
            jsonl,
            "--output-dir",
            str(out),
            "--causes",
            causes_path,
            "--allow-unexplained",
        ]
    )
    assert rc == 0  # unexplained exists but explicitly allowed

    ledger = [json.loads(ln) for ln in (out / "ledger.jsonl").read_text().splitlines()]
    summary = json.loads((out / "summary.json").read_text())
    md = (out / "LEDGER.md").read_text()

    # Rule (d): exactly one tier per row; counts total the input.
    assert len(ledger) == len(rows)
    assert sum(summary["tier_counts"].values()) == len(rows)

    tiers = {r["combo_id"]: r["tier"] for r in ledger}
    assert tiers["g::bitexact"] == "MODE_B_BIT_EXACT"
    assert tiers["g::identdist"] == "MODE_B_IDENTICAL_DISTANCE"
    assert tiers["g::close"] == "MODE_B_CLOSE"
    assert tiers["g::bigdist"] == "DIVERGENT_NAMED_CAUSE"
    assert tiers["g::positional"] == "POSITIONAL_IDENTICAL"
    assert tiers["g::cloud"] == "DISTRIBUTIONAL_EQUIVALENT"
    assert tiers["g::quality"] == "QUALITY_EQUIVALENT"
    assert tiers["g::quality_expl"] == "QUALITY_EQUIVALENT"
    assert tiers["g::superior"] == "SUPERIOR_DISTINCT"
    assert tiers["g::named"] == "DIVERGENT_NAMED_CAUSE"
    assert tiers["g::unexplained"] == "DIVERGENT_UNEXPLAINED"
    assert tiers["g::insufficient"] == "INSUFFICIENT_DATA"
    assert tiers["g::nocanon"] == "NO_CANONICAL_REFERENCE"
    assert tiers["g::capped"] == "QUALITY_EQUIVALENT"
    assert tiers["g::detbig"] == "DIVERGENT_NAMED_CAUSE"
    assert tiers["g::fewseeds"] == "INSUFFICIENT_DATA"

    # Exploratory split must be visible in counts and LEDGER.md.
    assert summary["tier_counts"]["QUALITY_EQUIVALENT_EXPLORATORY"] == 1
    assert summary["tier_counts"]["QUALITY_EQUIVALENT_CONFIRMATORY"] == 2
    assert "QUALITY_EQUIVALENT_EXPLORATORY" in md
    expl = {r["combo_id"]: r["exploratory"] for r in ledger}
    assert expl["g::quality_expl"] is True and expl["g::quality"] is False

    # Unexplained residual is listed with metrics.
    assert "g::unexplained" in md


def test_unexplained_causes_nonzero_exit(tmp_path):
    rows = [mode_a_row(engine="unexplained", mean_diag_B=0.4)]
    jsonl, _ = write_inputs(tmp_path, rows)
    out = tmp_path / "out"
    rc = bdl.run(["--per-combo", jsonl, "--output-dir", str(out), "--causes", ""])
    assert rc != 0
    # Outputs are still written for inspection.
    assert (out / "ledger.jsonl").exists() and (out / "LEDGER.md").exists()


def test_clean_dataset_exits_zero(tmp_path):
    rows = [
        mode_b_row(engine="bitexact", d_R=1e-12),
        mode_a_row(engine="cloud", dist_equivalent=True),
    ]
    jsonl, _ = write_inputs(tmp_path, rows)
    rc = bdl.run(["--per-combo", jsonl, "--output-dir", str(tmp_path / "out"), "--causes", ""])
    assert rc == 0


def test_stale_map_flags_without_changing_tier(tmp_path):
    rows = [
        mode_a_row(engine="classic_fmmm_steps10", mean_diag_B=1e-5, dist_equivalent=True),
        mode_a_row(engine="classic_gem_iters100", mean_diag_B=1e-5, dist_equivalent=True),
    ]
    jsonl, _ = write_inputs(tmp_path, rows)
    # Fake eval root: fmmm winner dir dated old (mtime set to 1970), gem dir fresh.
    # Each winner dir must contain an `ok` record for its combo so the fail-closed
    # coverage guard counts the row as BACKED (a dir lacking the combo is a coverage gap).
    eval_root = tmp_path / "eval_output"
    dir_combo = {"olddir": "g::classic_fmmm_steps10", "newdir": "g::classic_gem_iters100"}
    for d, cid in dir_combo.items():
        (eval_root / f"benchmark_100seed_{d}").mkdir(parents=True)
        (eval_root / f"benchmark_100seed_{d}" / "results.json").write_text(
            json.dumps({f"{cid}::seed42": {"status": "ok"}})
        )
    os.utime(eval_root / "benchmark_100seed_olddir" / "results.json", (0, 0))  # 1970
    winners = tmp_path / "winners.json"
    winners.write_text(
        json.dumps({"g::classic_fmmm_steps10": "olddir", "g::classic_gem_iters100": "newdir"})
    )
    stale_map = tmp_path / "stale.json"
    stale_map.write_text(json.dumps({"fmmm": "2026-06-22"}))
    out = tmp_path / "out"
    rc = bdl.run(
        [
            "--per-combo",
            jsonl,
            "--output-dir",
            str(out),
            "--causes",
            "",
            "--stale-map",
            str(stale_map),
            "--winners",
            str(winners),
            "--eval-root",
            str(eval_root),
        ]
    )
    assert rc == 0
    ledger = {
        r["combo_id"]: r for r in map(json.loads, (out / "ledger.jsonl").read_text().splitlines())
    }
    fmmm = ledger["g::classic_fmmm_steps10"]
    gem = ledger["g::classic_gem_iters100"]
    assert fmmm["stale_code"] is True and "predates" in fmmm["stale_reason"]
    assert gem["stale_code"] is False
    # Stale flag must NOT change the tier.
    assert fmmm["tier"] == "POSITIONAL_IDENTICAL"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["stale_code_rows"] == 1 and summary["stale_code_benign_rows"] == 1


def test_causes_sidecar_simple_map_format(tmp_path):
    rows = [mode_a_row(engine="named", mean_diag_B=0.4)]
    jsonl, causes_path = write_inputs(tmp_path, rows, {"g::named": "known residual"})
    out = tmp_path / "out"
    rc = bdl.run(["--per-combo", jsonl, "--output-dir", str(out), "--causes", causes_path])
    assert rc == 0
    ledger = [json.loads(ln) for ln in (out / "ledger.jsonl").read_text().splitlines()]
    assert ledger[0]["tier"] == "DIVERGENT_NAMED_CAUSE"
    assert ledger[0]["named_cause"] == "known residual"


# ---------------------------------------------------------------------------
# Smoke test on real r77 data (skipped when the artifact is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not R77_JSONL.exists(), reason="r77 per-combo jsonl not present")
def test_smoke_on_real_r77_sample(tmp_path):
    # Deterministic sample: every 130th row -> ~30 rows across families.
    with open(R77_JSONL) as f:
        sample = [line for line in itertools.islice(f, 0, None, 130)][:30]
    jsonl = tmp_path / "sample.jsonl"
    jsonl.write_text("".join(sample))
    causes = REPO / "eval_output" / "fidelity_definitive_r77" / "OFFICIAL_R77_LEDGER.json"
    out = tmp_path / "out"
    argv = ["--per-combo", str(jsonl), "--output-dir", str(out), "--allow-unexplained"]
    if causes.exists():
        argv += ["--causes", str(causes)]
    else:
        argv += ["--causes", ""]
    rc = bdl.run(argv)
    assert rc == 0
    ledger = [json.loads(ln) for ln in (out / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == len(sample)
    valid = set(bdl.TIER_ORDER)
    assert all(r["tier"] in valid for r in ledger)
    summary = json.loads((out / "summary.json").read_text())
    assert sum(summary["tier_counts"].values()) == len(sample)
