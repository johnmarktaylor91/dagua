#!/usr/bin/env python3
"""Build the definitive, persisted fidelity ledger from r77 per-combo scoring rows.

This script is the PERSISTED, REPRODUCIBLE replacement for the throwaway script
that assigned the r77 ledger dispositions (only a rule summary survived in
eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md). It re-tiers every
(graph, engine-variant) combo with an HONEST taxonomy: each row is labeled with
the STRONGEST claim it actually supports, no more.

The honesty ladder (all three levels are PASSES, ordered by claim strength):
  1. bit/positionally identical        (MODE_B_* / POSITIONAL_IDENTICAL)
  2. statistically indistinguishable   (DISTRIBUTIONAL_EQUIVALENT)
  3. quality indistinguishable         (QUALITY_EQUIVALENT)

Key revision vs r77: the old FIDELITY_IDENTICAL catch-all (2,678 rows) was gated
ONLY on a quality-TOST battery with NO positional comparison. It is replaced by
POSITIONAL_IDENTICAL (per-seed matched Procrustes), DISTRIBUTIONAL_EQUIVALENT
(positional-cloud equivalence), and QUALITY_EQUIVALENT (the honest name for the
quality-battery-only claim).

Pure post-processing: stdlib + numpy only. No network, no layout execution,
deterministic given the same inputs.

Usage:
  python scripts/build_definitive_ledger.py \
      --per-combo eval_output/fidelity_definitive/per_combo_r77.jsonl \
      --output-dir eval_output/fidelity_definitive_ledger \
      [--causes <json>] [--stale-map <json>] [--winners <json>] \
      [--allow-unexplained]

Outputs (in --output-dir):
  ledger.jsonl      one row per combo: tier, exploratory, stale_code, named_cause,
                    prior_disposition, and the key metrics the gates used
  LEDGER.md         human-readable ledger: glossary, old-vs-new counts, headline,
                    stale section, kept-divergent section, unexplained residual
  summary.json      machine-readable counts + gate constants + input hashes
  causes_used.json  the named-cause sidecar actually applied (reproducibility)

Exit status: nonzero if any row lands DIVERGENT_UNEXPLAINED, unless
--allow-unexplained is passed. Outputs are written either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Gate constants (carried from the r77 rule summary in OFFICIAL_R77_LEDGER.md
# and the project's standing per-seed Procrustes rule).
# ---------------------------------------------------------------------------

# Mode-B deterministic-pair ladder (r77 gates, unchanged):
#   d_R < 1e-9  -> MODE_B_BIT_EXACT
#   d_R < 0.01  -> MODE_B_IDENTICAL_DISTANCE
#   d_R < 0.1   -> MODE_B_CLOSE
D_R_BIT_EXACT = 1.0e-9
D_R_IDENTICAL = 1.0e-2
D_R_CLOSE = 1.0e-1

# Per-seed matched Procrustes bar (project standing rule): mean_diag_B < 1e-3.
MEAN_DIAG_B_IDENTICAL = 1.0e-3

# Minimum matched seeds for ANY benign mode-A verdict (mirrors r77's
# matched_seeds_lt_30 insufficiency gate).
MIN_MATCHED_SEEDS = 30

# "Large positional distance" for the deterministic-pair hard rule (b): a
# near-deterministic pair whose per-seed matched Procrustes distance exceeds
# this cannot claim any equivalence tier without a named cause. 0.5 is the
# adversarial audit's own bar for "clearly not the same layout".
LARGE_DISTANCE = 0.5

# Tier names, in precedence order (highest / strongest claim first).
TIER_MODE_B_BIT_EXACT = "MODE_B_BIT_EXACT"
TIER_MODE_B_IDENTICAL_DISTANCE = "MODE_B_IDENTICAL_DISTANCE"
TIER_MODE_B_CLOSE = "MODE_B_CLOSE"
TIER_POSITIONAL_IDENTICAL = "POSITIONAL_IDENTICAL"
TIER_DISTRIBUTIONAL_EQUIVALENT = "DISTRIBUTIONAL_EQUIVALENT"
TIER_QUALITY_EQUIVALENT = "QUALITY_EQUIVALENT"
TIER_SUPERIOR_DISTINCT = "SUPERIOR_DISTINCT"
TIER_DIVERGENT_NAMED_CAUSE = "DIVERGENT_NAMED_CAUSE"
TIER_DIVERGENT_UNEXPLAINED = "DIVERGENT_UNEXPLAINED"
TIER_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
TIER_AGGREGATE_INSUFFICIENT = "AGGREGATE_INSUFFICIENT"
TIER_AGGREGATE_STALE_REFERENCE = "AGGREGATE_STALE_REFERENCE"
TIER_NO_CANONICAL_REFERENCE = "NO_CANONICAL_REFERENCE"

# Equivalence-claiming tiers ("benign" for the hard rules). SUPERIOR_DISTINCT
# is deliberately NOT here: it explicitly claims distinctness, not equivalence.
BENIGN_TIERS = frozenset(
    {
        TIER_MODE_B_BIT_EXACT,
        TIER_MODE_B_IDENTICAL_DISTANCE,
        TIER_MODE_B_CLOSE,
        TIER_POSITIONAL_IDENTICAL,
        TIER_DISTRIBUTIONAL_EQUIVALENT,
        TIER_QUALITY_EQUIVALENT,
    }
)

POSITIONAL_OR_BETTER = (
    TIER_MODE_B_BIT_EXACT,
    TIER_MODE_B_IDENTICAL_DISTANCE,
    TIER_MODE_B_CLOSE,
    TIER_POSITIONAL_IDENTICAL,
)

TIER_ORDER = (
    TIER_MODE_B_BIT_EXACT,
    TIER_MODE_B_IDENTICAL_DISTANCE,
    TIER_MODE_B_CLOSE,
    TIER_POSITIONAL_IDENTICAL,
    TIER_DISTRIBUTIONAL_EQUIVALENT,
    TIER_QUALITY_EQUIVALENT,
    TIER_SUPERIOR_DISTINCT,
    TIER_DIVERGENT_NAMED_CAUSE,
    TIER_DIVERGENT_UNEXPLAINED,
    TIER_AGGREGATE_STALE_REFERENCE,
    TIER_AGGREGATE_INSUFFICIENT,
    TIER_INSUFFICIENT_DATA,
    TIER_NO_CANONICAL_REFERENCE,
)

DEFAULT_PER_COMBO = "eval_output/fidelity_definitive/per_combo_r77.jsonl"
DEFAULT_CAUSES = "eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.json"


# ---------------------------------------------------------------------------
# Small helpers (defensive against null / NaN / missing fields)
# ---------------------------------------------------------------------------


def is_num(x) -> bool:
    """True when x is a real, finite-or-infinite float/int (not None/NaN/bool)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return not (isinstance(x, float) and math.isnan(x))


def as_float(x):
    """Return float(x) when numeric, else None (JSON-safe metric passthrough)."""
    return float(x) if is_num(x) else None


def n_matched(row: dict) -> int:
    """Matched-seed count; prefers explicit n_matched, else len(matched_seeds)."""
    nmt = row.get("n_matched")
    if is_num(nmt):
        return int(nmt)
    seeds = row.get("matched_seeds")
    return len(seeds) if isinstance(seeds, list) else 0


def combo_id(row: dict) -> str:
    cid = row.get("combo_id")
    if isinstance(cid, str) and cid:
        return cid
    return f"{row.get('graph', '?')}::{row.get('engine', '?')}"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Prior / causes sidecar loading
# ---------------------------------------------------------------------------


def load_prior_sidecar(path: str | None) -> tuple[dict, dict]:
    """Load the r77 prior-disposition + named-cause sidecar.

    Accepts either:
      * the full OFFICIAL_R77_LEDGER.json (dict with a "rows" list, each row
        carrying combo_id / disposition / cause), or
      * a simple {combo_id: cause_string} map, or
      * a simple {combo_id: {"disposition": ..., "cause": ...}} map.

    Returns (priors, causes):
      priors: {combo_id: prior_disposition_string}
      causes: {combo_id: named_cause_string}  (specific, adjudicated causes only)
    """
    priors: dict = {}
    causes: dict = {}
    if not path:
        return priors, causes
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        entries = data["rows"]
        for e in entries:
            cid = e.get("combo_id")
            if not cid:
                continue
            disp = e.get("disposition")
            if disp:
                priors[cid] = disp
            # Only causes attached to adjudicated dispositions count as NAMED.
            # (Benign-tier "cause" strings in the r77 ledger are generic gate
            # descriptions like "quality_identical_raw or ...", not causes.)
            if disp in (
                TIER_DIVERGENT_NAMED_CAUSE,
                TIER_AGGREGATE_STALE_REFERENCE,
                TIER_AGGREGATE_INSUFFICIENT,
                TIER_NO_CANONICAL_REFERENCE,
            ):
                cause = e.get("cause")
                if cause:
                    causes[cid] = cause
        return priors, causes

    if isinstance(data, dict):
        for cid, v in data.items():
            if isinstance(v, str):
                causes[cid] = v
            elif isinstance(v, dict):
                if v.get("disposition"):
                    priors[cid] = v["disposition"]
                if v.get("cause"):
                    causes[cid] = v["cause"]
        return priors, causes

    raise ValueError(f"Unrecognized causes sidecar format: {path}")


# ---------------------------------------------------------------------------
# Stale-code provenance guard
# ---------------------------------------------------------------------------


def resolve_dir_date(dir_name: str, eval_root: str) -> date | None:
    """Resolve a winners source-dir name to a date via results.json mtime.

    Tries eval_output/benchmark_100seed_<dir>/results.json first, then
    eval_output/<dir>/results.json. Returns None when unresolvable.
    """
    for cand in (
        os.path.join(eval_root, f"benchmark_100seed_{dir_name}", "results.json"),
        os.path.join(eval_root, dir_name, "results.json"),
    ):
        if os.path.exists(cand):
            return date.fromtimestamp(os.path.getmtime(cand))
    return None


def compute_stale_flags(
    combos: list[str],
    engines: dict,
    stale_map: dict | None,
    winners: dict | None,
    eval_root: str,
) -> dict:
    """Return {combo_id: {"stale_code": bool, "stale_reason": str|None}}.

    A row is stale when its engine name contains a stale-map substring AND its
    winning source dir's date predates that substring's ISO date (i.e. the
    winning result was produced before the engine's code was last fixed).
    Unresolvable dirs are flagged stale (conservative) with a distinct reason.
    """
    out = {cid: {"stale_code": False, "stale_reason": None} for cid in combos}
    if not stale_map or not winners:
        return out
    cutoffs = {}
    for sub, iso in stale_map.items():
        cutoffs[sub] = datetime.strptime(iso, "%Y-%m-%d").date()
    dir_dates: dict = {}
    for cid in combos:
        engine = engines.get(cid, "")
        matched = [(sub, d) for sub, d in cutoffs.items() if sub in engine]
        if not matched:
            continue
        win_dir = winners.get(cid)
        if win_dir is None:
            continue
        if win_dir not in dir_dates:
            dir_dates[win_dir] = resolve_dir_date(win_dir, eval_root)
        ddate = dir_dates[win_dir]
        # Latest cutoff wins when several substrings match.
        sub, cutoff = max(matched, key=lambda t: t[1])
        if ddate is None:
            out[cid] = {
                "stale_code": True,
                "stale_reason": (
                    f"winning dir {win_dir!r} unresolvable; "
                    f"engine code changed {cutoff.isoformat()} ({sub})"
                ),
            }
        elif ddate < cutoff:
            out[cid] = {
                "stale_code": True,
                "stale_reason": (
                    f"winning dir {win_dir!r} dated {ddate.isoformat()} "
                    f"predates engine fix {cutoff.isoformat()} ({sub})"
                ),
            }
    return out


# ---------------------------------------------------------------------------
# Tier assignment (the heart of the ledger)
# ---------------------------------------------------------------------------


def assign_tier(row: dict, prior: str | None, cause: str | None) -> dict:
    """Assign the single strongest honest tier to one per-combo row.

    Returns {"tier": str, "exploratory": bool, "named_cause": str|None,
             "reason": str}.

    Precedence and hard rules:
      * insufficiency / no-canonical / stale-reference routing keeps r77
        semantics (carried via the prior sidecar where the jsonl lacks flags);
      * named-cause adjudications are STICKY: a prior DIVERGENT_NAMED_CAUSE row
        is never promoted to a benign tier by a mechanical gate (anti-laundering,
        r73 lesson) -- promotion requires fresh adjudication, i.e. a new sidecar;
      * (a) dist_equivalent == False caps the row at QUALITY_EQUIVALENT;
      * (b) a deterministic pair (mode B, or near_deterministic with
        mean_diag_B > LARGE_DISTANCE) with large positional distance cannot
        land in a benign tier without a named cause;
      * (c) no benign mode-A tier without >= MIN_MATCHED_SEEDS matched seeds
        (such rows are INSUFFICIENT_DATA, mirroring r77).
    """

    def result(tier, reason, exploratory=False, named=None):
        return {
            "tier": tier,
            "exploratory": bool(exploratory),
            "named_cause": named,
            "reason": reason,
        }

    def divergent(reason_suffix):
        if cause:
            return result(
                TIER_DIVERGENT_NAMED_CAUSE,
                f"divergent ({reason_suffix}); named cause carried",
                named=cause,
            )
        return result(TIER_DIVERGENT_UNEXPLAINED, f"divergent ({reason_suffix}); NO named cause")

    mode = row.get("mode")
    insufficient = bool(row.get("insufficient_data"))
    no_canonical = bool(row.get("no_canonical_reference"))

    # --- Non-verdict routing (r77 semantics) ------------------------------
    if insufficient or mode == "INSUFFICIENT_DATA":
        # r77 routed some insufficient rows to AGGREGATE_INSUFFICIENT (large
        # graphs, population-aggregate only) or NO_CANONICAL_REFERENCE (SFDP
        # parameter no-ops); carry those, else plain INSUFFICIENT_DATA.
        if prior in (TIER_AGGREGATE_INSUFFICIENT, TIER_NO_CANONICAL_REFERENCE):
            return result(prior, f"insufficient data; r77 routing carried ({prior})", named=cause)
        reason = row.get("insufficient_reason") or "insufficient data"
        return result(TIER_INSUFFICIENT_DATA, str(reason))

    if no_canonical or prior == TIER_NO_CANONICAL_REFERENCE:
        return result(
            TIER_NO_CANONICAL_REFERENCE,
            "no canonical reference (jsonl flag or r77 SFDP parameter no-op routing)",
            named=cause,
        )

    if prior == TIER_AGGREGATE_STALE_REFERENCE:
        return result(
            TIER_AGGREGATE_STALE_REFERENCE,
            "reference still resolves to stale seed-era data; no per-seed verdict (r77 carry)",
            named=cause,
        )

    # --- Anti-laundering stickiness ---------------------------------------
    # An adjudicated named divergence is never silently promoted by the
    # mechanical gates below. (7 such mode-A rows pass dist_equivalent on the
    # r77 data; they stay divergent and are surfaced in LEDGER.md.)
    sticky_divergent = prior == TIER_DIVERGENT_NAMED_CAUSE and cause is not None

    # --- Mode B: deterministic-pair ladder (r77 gates) ---------------------
    if mode == "B":
        d_r = row.get("d_R")
        if not sticky_divergent and is_num(d_r):
            if d_r < D_R_BIT_EXACT:
                return result(TIER_MODE_B_BIT_EXACT, f"mode B, d_R={d_r:.3e} < {D_R_BIT_EXACT:g}")
            if d_r < D_R_IDENTICAL:
                return result(
                    TIER_MODE_B_IDENTICAL_DISTANCE, f"mode B, d_R={d_r:.3e} < {D_R_IDENTICAL:g}"
                )
            if d_r < D_R_CLOSE:
                return result(TIER_MODE_B_CLOSE, f"mode B, d_R={d_r:.3e} < {D_R_CLOSE:g}")
        # Hard rule (b): deterministic pair with large distance -> divergent
        # unless a named cause exists.
        d_txt = f"d_R={d_r:.3e}" if is_num(d_r) else "d_R missing"
        return divergent(f"mode B deterministic pair with large positional distance, {d_txt}")

    if mode != "A":
        # Defensive: unknown mode with sufficient-data flag; treat as
        # insufficient rather than inventing a verdict.
        return result(TIER_INSUFFICIENT_DATA, f"unrecognized mode {mode!r}")

    # --- Mode A -------------------------------------------------------------
    nmt = n_matched(row)
    mdb = row.get("mean_diag_B")
    de = row.get("dist_equivalent")
    qi_raw = row.get("quality_identical_raw") is True
    qi_exp = row.get("quality_identical_exploratory") is True
    superior = row.get("quality_superior_distinct") is True
    near_det = row.get("near_deterministic") is True

    # Hard rule (c): benign mode-A verdicts need >= 30 matched seeds.
    if nmt < MIN_MATCHED_SEEDS:
        return result(
            TIER_INSUFFICIENT_DATA,
            f"mode A with only {nmt} matched seeds (< {MIN_MATCHED_SEEDS}); no benign verdict",
        )

    if sticky_divergent:
        return result(
            TIER_DIVERGENT_NAMED_CAUSE,
            "r77 adjudicated named divergence kept (anti-laundering; gates never overturn it)",
            named=cause,
        )

    # Hard rule (b), mode-A form: near-deterministic pair with large per-seed
    # distance. Equivalence tiers are forbidden; SUPERIOR_DISTINCT is allowed
    # because it claims distinctness, not equivalence.
    if near_det and is_num(mdb) and mdb > LARGE_DISTANCE:
        if superior:
            return result(
                TIER_SUPERIOR_DISTINCT,
                f"near-deterministic mean_diag_B={mdb:.3g} > {LARGE_DISTANCE}; distinct/superior",
            )
        return divergent(f"near-deterministic pair with mean_diag_B={mdb:.3g} > {LARGE_DISTANCE}")

    # Hard rule (a): the harness's own positional-cloud verdict failed; the row
    # can never claim better than quality equivalence.
    if de is False:
        if qi_raw:
            return result(
                TIER_QUALITY_EQUIVALENT,
                "dist_equivalent=False caps at quality; quality battery identical (confirmatory)",
            )
        if qi_exp:
            return result(
                TIER_QUALITY_EQUIVALENT,
                "dist_equivalent=False caps at quality; battery identical (exploratory only)",
                exploratory=True,
            )
        if superior:
            return result(
                TIER_SUPERIOR_DISTINCT, "dist_equivalent=False; distinct and quality-superior"
            )
        mdb_txt = f"mean_diag_B={mdb:.3g}" if is_num(mdb) else "mean_diag_B missing"
        return divergent(
            f"dist_equivalent=False and no quality-identical/superior verdict, {mdb_txt}"
        )

    # Tier 3: per-seed matched Procrustes identity (project standing rule).
    if is_num(mdb) and mdb < MEAN_DIAG_B_IDENTICAL:
        return result(
            TIER_POSITIONAL_IDENTICAL,
            f"mode A, {nmt} matched seeds, mean_diag_B={mdb:.3e} < {MEAN_DIAG_B_IDENTICAL:g}",
        )

    # Tier 4: positional-cloud statistical equivalence.
    if de is True:
        return result(
            TIER_DISTRIBUTIONAL_EQUIVALENT, "mode A, dist_equivalent=True (positional-cloud TOST)"
        )

    # Tiers 5/6: quality-battery-only claims (dist_equivalent None/unavailable).
    if qi_raw:
        return result(
            TIER_QUALITY_EQUIVALENT, "quality battery identical (confirmatory); no positional pass"
        )
    if qi_exp:
        return result(
            TIER_QUALITY_EQUIVALENT,
            "quality battery identical (exploratory only; reference too self-dispersed)",
            exploratory=True,
        )

    # Tier 7: distinct but quality-superior.
    if superior:
        return result(TIER_SUPERIOR_DISTINCT, "quality_superior_distinct=True")

    mdb_txt = f"mean_diag_B={mdb:.3g}" if is_num(mdb) else "mean_diag_B missing"
    return divergent(f"fails all equivalence and superiority gates, {mdb_txt}")


# ---------------------------------------------------------------------------
# Ledger assembly
# ---------------------------------------------------------------------------


def build_ledger_rows(
    rows: list[dict],
    priors: dict,
    causes: dict,
    stale_flags: dict,
) -> list[dict]:
    out = []
    for row in rows:
        cid = combo_id(row)
        prior = priors.get(cid)
        cause = causes.get(cid)
        verdict = assign_tier(row, prior, cause)
        stale = stale_flags.get(cid, {"stale_code": False, "stale_reason": None})
        out.append(
            {
                "combo_id": cid,
                "graph": row.get("graph"),
                "engine": row.get("engine"),
                "tier": verdict["tier"],
                "exploratory": verdict["exploratory"],
                "stale_code": stale["stale_code"],
                "stale_reason": stale["stale_reason"],
                "named_cause": verdict["named_cause"],
                "tier_reason": verdict["reason"],
                "prior_disposition": prior,
                "mode": row.get("mode"),
                "n_matched": n_matched(row),
                "mean_diag_B": as_float(row.get("mean_diag_B")),
                "dist_equivalent": row.get("dist_equivalent"),
                "d_R": as_float(row.get("d_R")),
                "near_deterministic": row.get("near_deterministic"),
                "quality_identical_raw": row.get("quality_identical_raw"),
                "quality_identical_exploratory": row.get("quality_identical_exploratory"),
                "quality_equivalent_raw": row.get("quality_equivalent_raw"),
                "quality_superior_distinct": row.get("quality_superior_distinct"),
                "insufficient_reason": row.get("insufficient_reason"),
                "git_sha": row.get("git_sha"),
            }
        )
    return out


def tier_counts(ledger_rows: list[dict]) -> dict:
    """Counts keyed by tier, with QUALITY_EQUIVALENT split by exploratory flag."""
    counts: Counter = Counter()
    for r in ledger_rows:
        if r["tier"] == TIER_QUALITY_EQUIVALENT:
            key = (
                "QUALITY_EQUIVALENT_EXPLORATORY"
                if r["exploratory"]
                else "QUALITY_EQUIVALENT_CONFIRMATORY"
            )
        else:
            key = r["tier"]
        counts[key] += 1
    return dict(counts)


def build_summary(ledger_rows: list[dict], input_hashes: dict, args_echo: dict) -> dict:
    counts = tier_counts(ledger_rows)
    total = len(ledger_rows)
    n_pos_or_better = sum(1 for r in ledger_rows if r["tier"] in POSITIONAL_OR_BETTER)
    n_dist = sum(1 for r in ledger_rows if r["tier"] == TIER_DISTRIBUTIONAL_EQUIVALENT)
    n_quality = sum(1 for r in ledger_rows if r["tier"] == TIER_QUALITY_EQUIVALENT)
    n_stale_benign = sum(1 for r in ledger_rows if r["stale_code"] and r["tier"] in BENIGN_TIERS)
    unexplained = [r["combo_id"] for r in ledger_rows if r["tier"] == TIER_DIVERGENT_UNEXPLAINED]
    return {
        "total_rows": total,
        "tier_counts": counts,
        "headline": {
            "positional_or_better": n_pos_or_better,
            "distributional_equivalent": n_dist,
            "quality_equivalent_only": n_quality,
            "quality_equivalent_exploratory": counts.get("QUALITY_EQUIVALENT_EXPLORATORY", 0),
        },
        "stale_code_rows": sum(1 for r in ledger_rows if r["stale_code"]),
        "stale_code_benign_rows": n_stale_benign,
        "divergent_unexplained": unexplained,
        "gates": {
            "d_R_bit_exact": D_R_BIT_EXACT,
            "d_R_identical_distance": D_R_IDENTICAL,
            "d_R_close": D_R_CLOSE,
            "mean_diag_B_positional_identical": MEAN_DIAG_B_IDENTICAL,
            "min_matched_seeds": MIN_MATCHED_SEEDS,
            "large_distance_deterministic_guard": LARGE_DISTANCE,
        },
        "inputs": input_hashes,
        "args": args_echo,
    }


# ---------------------------------------------------------------------------
# LEDGER.md rendering
# ---------------------------------------------------------------------------

GLOSSARY = [
    (
        TIER_MODE_B_BIT_EXACT,
        "Deterministic pair, similarity-EXACT: residual d_R < 1e-9. d_R is O(2)/"
        "similarity-invariant (translation, uniform scale, rotation, reflection are "
        "factored out); sugiyama rows use the anisotropic residual. This proves the "
        "reimplementation reproduces the reference layout up to a similarity transform "
        "at machine precision -- NOT byte-identical coordinates.",
    ),
    (
        TIER_MODE_B_IDENTICAL_DISTANCE,
        "Deterministic pair with d_R < 0.01: the same layout up to similarity, "
        "within numerical-noise tolerance.",
    ),
    (
        TIER_MODE_B_CLOSE,
        "Deterministic pair with d_R < 0.1: visibly the same layout, small residual.",
    ),
    (
        TIER_POSITIONAL_IDENTICAL,
        "Stochastic engine, per-seed PROOF: >= 30 matched seeds, mean per-seed "
        "matched-Procrustes distance mean_diag_B < 1e-3. Every matched seed produces "
        "the same layout as the reference (up to similarity). Strongest stochastic claim.",
    ),
    (
        TIER_DISTRIBUTIONAL_EQUIVALENT,
        "Stochastic engine, positional-CLOUD equivalence (dist_equivalent=True): the "
        "cross distances between reimplementation and reference layouts are statistically "
        "indistinguishable from the reference's own seed-to-seed spread. Proves the two "
        "samplers draw from indistinguishable layout distributions -- NOT per-seed identity.",
    ),
    (
        TIER_QUALITY_EQUIVALENT,
        "Quality-battery ONLY (honest rename of r77's FIDELITY_IDENTICAL): stress / "
        "crossings / neighborhood-preservation TOST-identical, but NO positional claim "
        "holds (or the positional-cloud test failed outright). Proves the layouts are of "
        "equivalent quality, not that they are the same layouts. Exploratory sub-bucket = "
        "reference too self-dispersed to discriminate; counted separately.",
    ),
    (
        TIER_SUPERIOR_DISTINCT,
        "Distinct layouts, and the reimplementation is quality-superior to the reference. "
        "A distinctness claim, not an equivalence claim.",
    ),
    (
        TIER_DIVERGENT_NAMED_CAUSE,
        "Divergent with an adjudicated, instrument-backed named cause (r77 registry). "
        "Sticky: mechanical gates never promote these without fresh adjudication.",
    ),
    (TIER_DIVERGENT_UNEXPLAINED, "Divergent with NO named cause. Must be zero at sign-off."),
    (
        TIER_AGGREGATE_STALE_REFERENCE,
        "Reference still resolves to a stale seed era; no per-seed verdict exists (r77 carry).",
    ),
    (
        TIER_AGGREGATE_INSUFFICIENT,
        "Large-graph row lacking paired seeds; population-aggregate evidence only (r77 carry).",
    ),
    (TIER_INSUFFICIENT_DATA, "Fewer than 30 matched seeds (or no reference rows); no verdict."),
    (
        TIER_NO_CANONICAL_REFERENCE,
        "No canonical reference exists for the variant (e.g. Graphviz SFDP ignores the "
        "theta/steps/p_neg2 attributes, so those variants have nothing real to match).",
    ),
]


def render_ledger_md(ledger_rows: list[dict], summary: dict, priors: dict) -> str:
    counts = summary["tier_counts"]
    total = summary["total_rows"]
    lines: list[str] = []
    add = lines.append
    add("# Definitive Fidelity Ledger (honest tier taxonomy)")
    add("")
    add(f"Built by `scripts/build_definitive_ledger.py` from {summary['args'].get('per_combo')!r}.")
    add(f"Total rows: {total}.")
    add("")
    add("Every row carries the STRONGEST claim it actually supports, no more. The")
    add("honesty ladder: (1) bit/positionally identical, (2) statistically")
    add("indistinguishable, (3) quality indistinguishable. All three are passes;")
    add("the old FIDELITY_IDENTICAL catch-all (quality-TOST only, no positional")
    add("comparison) is retired in favor of this ladder.")
    add("")

    add("## Honest headline")
    add("")
    h = summary["headline"]
    add(
        f"- Positional-or-better (MODE_B_* + POSITIONAL_IDENTICAL): **{h['positional_or_better']}**"
    )
    add(f"- Distributionally equivalent (positional-cloud): **{h['distributional_equivalent']}**")
    add(
        f"- Quality-equivalent ONLY: **{h['quality_equivalent_only']}** "
        f"(of which exploratory: {h['quality_equivalent_exploratory']})"
    )
    add(f"- Superior-distinct: {counts.get(TIER_SUPERIOR_DISTINCT, 0)}")
    add(f"- Divergent, named cause: {counts.get(TIER_DIVERGENT_NAMED_CAUSE, 0)}")
    add(f"- Divergent, UNEXPLAINED: {counts.get(TIER_DIVERGENT_UNEXPLAINED, 0)}")
    n_nonverdict = (
        counts.get(TIER_INSUFFICIENT_DATA, 0)
        + counts.get(TIER_AGGREGATE_INSUFFICIENT, 0)
        + counts.get(TIER_AGGREGATE_STALE_REFERENCE, 0)
        + counts.get(TIER_NO_CANONICAL_REFERENCE, 0)
    )
    add(f"- No verdict (insufficient / stale-reference / no-canonical): {n_nonverdict}")
    add("")

    add("## Tier counts (new taxonomy)")
    add("")
    add("| Tier | Rows |")
    add("| --- | ---: |")
    display_order = list(TIER_ORDER)
    qi = display_order.index(TIER_QUALITY_EQUIVALENT)
    display_order[qi : qi + 1] = [
        "QUALITY_EQUIVALENT_CONFIRMATORY",
        "QUALITY_EQUIVALENT_EXPLORATORY",
    ]
    for t in display_order:
        add(f"| `{t}` | {counts.get(t, 0)} |")
    add(f"| **total** | **{total}** |")
    add("")

    if priors:
        add("## Old (r77) vs new naming -- migration matrix")
        add("")
        add("Rows counted by (r77 disposition -> new tier). The old FIDELITY_IDENTICAL")
        add("bucket fans out across the honesty ladder; nothing is silently dropped.")
        add("")
        mig: Counter = Counter()
        for r in ledger_rows:
            old = r["prior_disposition"] or "(none)"
            new = r["tier"] + (" (exploratory)" if r["exploratory"] else "")
            mig[(old, new)] += 1
        add("| r77 disposition | New tier | Rows |")
        add("| --- | --- | ---: |")
        for (old, new), c in sorted(mig.items()):
            add(f"| `{old}` | `{new}` | {c} |")
        add("")

    add("## What each tier PROVES (glossary)")
    add("")
    for tier, desc in GLOSSARY:
        add(f"- **`{tier}`** -- {desc}")
    add("")

    add("## Hard rules enforced")
    add("")
    add("- (a) `dist_equivalent == False` caps a row at QUALITY_EQUIVALENT, whatever else passes.")
    add(
        "- (b) a deterministic pair (mode B, or near_deterministic with mean_diag_B > "
        f"{LARGE_DISTANCE}) with large positional distance never lands in a benign tier "
        "without a named cause."
    )
    add(f"- (c) no benign mode-A tier without >= {MIN_MATCHED_SEEDS} matched seeds.")
    add("- (d) every row gets exactly one tier; counts total the input row count.")
    add("- Named-cause adjudications are sticky: mechanical gates never promote a row")
    add("  out of DIVERGENT_NAMED_CAUSE (anti-laundering). See section below.")
    add("")

    kept = [
        r
        for r in ledger_rows
        if r["tier"] == TIER_DIVERGENT_NAMED_CAUSE
        and r["prior_disposition"] == TIER_DIVERGENT_NAMED_CAUSE
        and r["mode"] == "A"
        and r["dist_equivalent"] is True
    ]
    if kept:
        add("## Named-divergent rows that pass the mechanical cloud gate (kept divergent)")
        add("")
        add("These rows pass `dist_equivalent` mechanically, but carry an adjudicated,")
        add("instrument-backed divergence cause from r77. Promotion requires fresh")
        add("adjudication (new sidecar), not a re-run of the gate the adjudication")
        add("already overrode.")
        add("")
        for r in kept:
            add(f"- `{r['combo_id']}` -- {r['named_cause']}")
        add("")

    stale_rows = [r for r in ledger_rows if r["stale_code"]]
    add("## Stale-code provenance guard")
    add("")
    if stale_rows:
        n_benign = summary["stale_code_benign_rows"]
        add(f"{len(stale_rows)} rows won from result dirs that predate the engine's most")
        add(f"recent code fix; {n_benign} of them sit in benign tiers and are pending")
        add("re-bench at HEAD. Flags do NOT change tiers.")
        add("")
        add("| Combo | Tier | Reason |")
        add("| --- | --- | --- |")
        for r in stale_rows:
            add(f"| `{r['combo_id']}` | `{r['tier']}` | {r['stale_reason']} |")
    else:
        add("No stale-code rows flagged (no --stale-map/--winners provided, or none matched).")
    add("")

    unexpl = [r for r in ledger_rows if r["tier"] == TIER_DIVERGENT_UNEXPLAINED]
    add("## Divergent-unexplained residual")
    add("")
    if unexpl:
        add(f"{len(unexpl)} rows support NO honest claim and carry no named cause. These")
        add("are the real residual work; sign-off requires driving this list to zero")
        add("(name the cause with instruments, or fix the engine).")
        add("")
        add("| Combo | mode | n_matched | mean_diag_B | dist_eq | d_R | near_det | quality flags |")
        add("| --- | --- | ---: | ---: | --- | ---: | --- | --- |")
        for r in unexpl:
            mdb = f"{r['mean_diag_B']:.4g}" if r["mean_diag_B"] is not None else "NA"
            d_r = f"{r['d_R']:.4g}" if r["d_R"] is not None else "NA"
            q = (
                f"ident={r['quality_identical_raw']}, expl={r['quality_identical_exploratory']}, "
                f"equiv={r['quality_equivalent_raw']}, sup={r['quality_superior_distinct']}"
            )
            add(
                f"| `{r['combo_id']}` | {r['mode']} | {r['n_matched']} | {mdb} | "
                f"{r['dist_equivalent']} | {d_r} | {r['near_deterministic']} | {q} |"
            )
    else:
        add("None. Every divergent row carries an adjudicated named cause.")
    add("")

    add("## Gate constants")
    add("")
    for k, v in summary["gates"].items():
        add(f"- `{k}` = {v:g}")
    add("")
    add("## Inputs")
    add("")
    for name, meta in summary["inputs"].items():
        add(f"- {name}: `{meta['path']}` (sha256 `{meta['sha256'][:16]}...`)")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--per-combo", default=DEFAULT_PER_COMBO, help="per-combo scoring jsonl")
    ap.add_argument(
        "--output-dir",
        default="eval_output/fidelity_definitive_ledger",
        help="directory for ledger.jsonl / LEDGER.md / summary.json",
    )
    ap.add_argument(
        "--causes",
        default=DEFAULT_CAUSES,
        help=(
            "named-cause / prior-disposition sidecar: either the full "
            "OFFICIAL_R77_LEDGER.json or a {combo_id: cause} map. Pass '' to disable."
        ),
    )
    ap.add_argument(
        "--stale-map", default=None, help="JSON {engine_substring: iso_date} of engine fix dates"
    )
    ap.add_argument("--winners", default=None, help="JSON {combo_id: source_dir} winner map")
    ap.add_argument(
        "--eval-root", default="eval_output", help="root for resolving winner dirs to dates"
    )
    ap.add_argument(
        "--allow-unexplained",
        action="store_true",
        help="exit 0 even when DIVERGENT_UNEXPLAINED rows exist (outputs are written regardless)",
    )
    args = ap.parse_args(argv)

    rows = []
    with open(args.per_combo, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    causes_path = args.causes if args.causes and os.path.exists(args.causes) else None
    if args.causes and not causes_path:
        print(
            f"WARNING: causes sidecar {args.causes!r} not found; proceeding without priors",
            file=sys.stderr,
        )
    priors, causes = load_prior_sidecar(causes_path)

    stale_map = json.load(open(args.stale_map)) if args.stale_map else None
    winners = json.load(open(args.winners)) if args.winners else None
    combos = [combo_id(r) for r in rows]
    engines = {combo_id(r): (r.get("engine") or "") for r in rows}
    stale_flags = compute_stale_flags(combos, engines, stale_map, winners, args.eval_root)

    ledger_rows = build_ledger_rows(rows, priors, causes, stale_flags)

    # Invariant (d): exactly one tier per row, counts total the input.
    assert len(ledger_rows) == len(rows)
    assert sum(tier_counts(ledger_rows).values()) == len(rows)

    input_hashes = {"per_combo": {"path": args.per_combo, "sha256": sha256_of(args.per_combo)}}
    if causes_path:
        input_hashes["causes"] = {"path": causes_path, "sha256": sha256_of(causes_path)}
    if args.stale_map:
        input_hashes["stale_map"] = {"path": args.stale_map, "sha256": sha256_of(args.stale_map)}
    if args.winners:
        input_hashes["winners"] = {"path": args.winners, "sha256": sha256_of(args.winners)}

    args_echo = {
        "per_combo": args.per_combo,
        "causes": causes_path,
        "stale_map": args.stale_map,
        "winners": args.winners,
        "allow_unexplained": args.allow_unexplained,
    }
    summary = build_summary(ledger_rows, input_hashes, args_echo)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ledger.jsonl"), "w", encoding="utf-8") as f:
        for r in ledger_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(args.output_dir, "LEDGER.md"), "w", encoding="utf-8") as f:
        f.write(render_ledger_md(ledger_rows, summary, priors))
    # Reproducibility: persist the causes actually applied alongside the ledger.
    with open(os.path.join(args.output_dir, "causes_used.json"), "w", encoding="utf-8") as f:
        json.dump({"priors": priors, "causes": causes}, f, indent=2, sort_keys=True)
        f.write("\n")

    counts = summary["tier_counts"]
    print(f"Ledger written to {args.output_dir} ({len(ledger_rows)} rows)")
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c:5d}  {t}")

    n_unexplained = counts.get(TIER_DIVERGENT_UNEXPLAINED, 0)
    if n_unexplained and not args.allow_unexplained:
        print(
            f"ERROR: {n_unexplained} DIVERGENT_UNEXPLAINED rows (see LEDGER.md residual section); "
            "sign-off requires zero. Pass --allow-unexplained to override.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
