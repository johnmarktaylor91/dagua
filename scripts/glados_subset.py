"""Blind hash-rank subset selection for the GLaDOS holdout run.

Implements the pre-registered selection rule of PLAN_FABLE_R2.md section 7.2
verbatim: rank every candidate corpus file by
``sha256(seed_string + "\\0" + corpus + "\\0" + filename)`` and take the first
``ceil(fraction * count)`` per corpus (stratification = corpus only).

Selection inputs are corpus + filename ONLY. This module contains NO code path
that opens a candidate file: ranking is a pure function over the name list
(unit-testable on names of nonexistent files). The only file reads are the
FETCHED_FILES.txt candidate list itself and the refuse-to-overwrite existence
check on SUBSET.json; the only writes are SUBSET.json + SEALED_REMAINDER.json.

Holdout opacity: never select on shape, visuals, score, or failure history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_SEED_STRING = "glados-holdout-2026-08-05"
DEFAULT_FRACTION = 0.45
DEFAULT_CORPUS_DIR = Path("eval_output/stdcorpora")
SUPPORTED_SUFFIXES = frozenset({".graph", ".gml", ".graphml", ".mtx"})
CORPUS_NAMES = ("rome", "north", "suitesparse")
RULE_TEXT = "sha256(seed\\0corpus\\0filename) hex ascending; first ceil(fraction*n) per corpus"


def rank_hash(seed_string: str, corpus: str, filename: str) -> str:
    """Return the pre-registered rank hash for one candidate.

    Parameters
    ----------
    seed_string : str
        Frozen selection seed string.
    corpus : str
        Corpus name (``rome``, ``north``, or ``suitesparse``).
    filename : str
        Candidate file basename (``path.name``).

    Returns
    -------
    str
        Hex sha256 digest of ``seed\\0corpus\\0filename``.
    """
    payload = f"{seed_string}\0{corpus}\0{filename}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rank_candidates(
    seed_string: str,
    candidates: Sequence[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Rank candidates per corpus by ascending hex rank hash.

    Pure function over the (corpus, filename) name list: performs no file I/O
    of any kind, so it works identically on names of nonexistent files.

    Parameters
    ----------
    seed_string : str
        Frozen selection seed string.
    candidates : Sequence[Tuple[str, str]]
        ``(corpus, filename)`` pairs.

    Returns
    -------
    Dict[str, List[Tuple[str, str, str]]]
        Per-corpus lists of ``(rank_hash, corpus, filename)`` sorted ascending
        by hex digest.

    Raises
    ------
    ValueError
        If a corpus contains duplicate basenames (position files and resume
        keys would collide downstream -- WP10-F05).
    """
    per_corpus: Dict[str, List[Tuple[str, str, str]]] = {}
    seen: Dict[Tuple[str, str], int] = {}
    duplicates: List[str] = []
    for corpus, filename in candidates:
        key = (corpus, filename)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            duplicates.append(f"{corpus}/{filename}")
        per_corpus.setdefault(corpus, []).append(
            (rank_hash(seed_string, corpus, filename), corpus, filename)
        )
    if duplicates:
        raise ValueError(
            "duplicate basenames within a corpus (would collide in position "
            f"files and resume keys): {sorted(duplicates)}"
        )
    for corpus in per_corpus:
        per_corpus[corpus].sort()
    return per_corpus


def parse_candidates(
    lines: Sequence[str],
    corpus_dir: Path,
) -> Tuple[List[Dict[str, str]], List[Tuple[str, str]]]:
    """Parse FETCHED_FILES.txt lines into candidates and logged exclusions.

    Purely lexical: no path on any line is opened or stat-ed.

    Parameters
    ----------
    lines : Sequence[str]
        Raw lines of the fetched-files list.
    corpus_dir : Path
        Corpus root the listed paths must live under.

    Returns
    -------
    Tuple[List[Dict[str, str]], List[Tuple[str, str]]]
        ``(candidates, excluded)`` where each candidate has ``corpus``,
        ``filename``, and ``relpath`` keys, and each exclusion pairs the raw
        line with a reason.
    """
    prefix = corpus_dir.as_posix().rstrip("/") + "/"
    candidates: List[Dict[str, str]] = []
    excluded: List[Tuple[str, str]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        posix = PurePosixPath(line).as_posix()
        if not posix.startswith(prefix):
            excluded.append((line, f"not under corpus dir {prefix}"))
            continue
        relpath = posix[len(prefix) :]
        parts = PurePosixPath(relpath).parts
        if len(parts) < 2:
            excluded.append((line, "no corpus segment under corpus dir"))
            continue
        corpus = parts[0]
        filename = parts[-1]
        if corpus not in CORPUS_NAMES:
            excluded.append((line, f"unknown corpus segment {corpus!r}"))
            continue
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            excluded.append((line, f"unsupported extension {suffix!r}"))
            continue
        candidates.append({"corpus": corpus, "filename": filename, "relpath": relpath})
    return candidates, excluded


def partition_candidates(
    seed_string: str,
    fraction: float,
    candidates: Sequence[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Dict[str, int]]]:
    """Partition candidates into selected and sealed sets per the frozen rule.

    Parameters
    ----------
    seed_string : str
        Frozen selection seed string.
    fraction : float
        Per-corpus selection fraction.
    candidates : Sequence[Dict[str, str]]
        Candidates from :func:`parse_candidates`.

    Returns
    -------
    Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Dict[str, int]]]
        ``(selected, sealed, per_corpus_counts)``. Entries carry ``corpus``,
        ``filename``, ``relpath``, and ``rank_hash``; both lists are sorted by
        ``(corpus, rank_hash)``.
    """
    relpath_by_key = {
        (candidate["corpus"], candidate["filename"]): candidate["relpath"]
        for candidate in candidates
    }
    ranked = rank_candidates(
        seed_string,
        [(candidate["corpus"], candidate["filename"]) for candidate in candidates],
    )
    selected: List[Dict[str, str]] = []
    sealed: List[Dict[str, str]] = []
    per_corpus: Dict[str, Dict[str, int]] = {}
    for corpus in sorted(ranked):
        ordering = ranked[corpus]
        take = math.ceil(fraction * len(ordering))
        per_corpus[corpus] = {"candidates": len(ordering), "selected": take}
        for index, (digest, _, filename) in enumerate(ordering):
            entry = {
                "corpus": corpus,
                "filename": filename,
                "relpath": relpath_by_key[(corpus, filename)],
                "rank_hash": digest,
            }
            (selected if index < take else sealed).append(entry)
    key = lambda entry: (entry["corpus"], entry["rank_hash"])  # noqa: E731
    return sorted(selected, key=key), sorted(sealed, key=key), per_corpus


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Argument list; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options. Every flag is defaulted so the bare pre-registered
        invocation ``python scripts/glados_subset.py`` is complete.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--fetched-list", type=Path, default=None)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--seed-string", default=DEFAULT_SEED_STRING)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sealed-output", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing SUBSET.json (the merge freeze anchors on its existence).",
    )
    args = parser.parse_args(argv)
    if args.fetched_list is None:
        args.fetched_list = args.corpus_dir / "FETCHED_FILES.txt"
    if args.output is None:
        args.output = args.corpus_dir / "SUBSET.json"
    if args.sealed_output is None:
        args.sealed_output = args.corpus_dir / "SEALED_REMAINDER.json"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run blind subset selection and write both partition files.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Argument list; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any refusal (existing SUBSET.json without
        ``--force``, missing fetched list, duplicate basenames, zero
        candidates).
    """
    args = parse_args(argv)
    if args.output.exists() and not args.force:
        print(
            f"REFUSING to overwrite existing {args.output} (merge freeze anchors on it); "
            "pass --force to regenerate.",
            file=sys.stderr,
        )
        return 1
    if not args.fetched_list.is_file():
        print(f"fetched list not found: {args.fetched_list}", file=sys.stderr)
        return 1
    list_bytes = args.fetched_list.read_bytes()
    candidates, excluded = parse_candidates(
        list_bytes.decode("utf-8", errors="replace").splitlines(), args.corpus_dir
    )
    for line, reason in excluded:
        print(f"excluded candidate: {line} ({reason})", file=sys.stderr)
    if not candidates:
        print("no eligible candidates found in fetched list; aborting", file=sys.stderr)
        return 1
    try:
        selected, sealed, per_corpus = partition_candidates(
            args.seed_string, args.fraction, candidates
        )
    except ValueError as exc:
        print(f"ABORT before writing anything: {exc}", file=sys.stderr)
        return 1

    header: Dict[str, Any] = {
        "seed_string": args.seed_string,
        "fraction": args.fraction,
        "rule": RULE_TEXT,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_dir": str(args.corpus_dir),
        "fetched_list_sha256": hashlib.sha256(list_bytes).hexdigest(),
        "per_corpus": per_corpus,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**header, "selected": selected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.sealed_output.parent.mkdir(parents=True, exist_ok=True)
    args.sealed_output.write_text(
        json.dumps({**header, "sealed": sealed}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = ", ".join(
        f"{corpus}: {counts['selected']}/{counts['candidates']}"
        for corpus, counts in sorted(per_corpus.items())
    )
    print(f"SUBSET written: {args.output} ({summary}); sealed remainder: {args.sealed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
