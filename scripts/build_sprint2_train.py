"""Build the sprint-2 TRAINING corpus: fresh graphs never part of any holdout.

Populates ``eval_output/sprint2_train/`` with graphs disjoint from the 140
allocated stdcorpora files (63 dev + 77 sealed):

- ``rome/`` + ``north/``: ~25 graphs each, drawn deterministically from the
  official graphdrawing.org archives (same host + archives the stdcorpora
  fetch used) by the rule documented in MANIFEST.json.
- ``suitesparse/``: ~15 SuiteSparse HB sparse-mesh matrices from a
  pre-declared name list (square, n<=2000, selected by name/size metadata
  only), none of which are among the 140. NOTE: the sprint-2 brief suggested
  bcspwr02-06, but those are SEALED holdout files -- they are deliberately
  NOT used; the dwt_*/can_* meshes below are the same stringy-mesh class.

Selection rule for rome/north (deterministic, re-runnable):

1. Candidates = archive members with a supported extension
   (.graphml/.gml/.graph), sorted by basename ascending (bytewise).
2. Exclude any candidate whose stem equals the stem of one of the 140
   allocated files (SUBSET.json + SEALED_REMAINDER.json).
3. Accept candidates in sorted order that load cleanly through the holdout
   loaders under the holdout directedness policy with 3 <= nodes <= 2000,
   1 <= edges <= 200000; stop at the per-corpus count.

Every emitted file is recorded in MANIFEST.json with sha256 + provenance.
A hard guard refuses to ever read or copy a sealed file.

Usage::

    python scripts/build_sprint2_train.py            # build (downloads ~20MB)
    python scripts/build_sprint2_train.py --verify   # re-hash existing corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.glados_holdout_run import (  # noqa: E402
    SUPPORTED_LOADERS,
    directedness_policy,
)

DEFAULT_TRAIN_DIR = Path("eval_output/sprint2_train")
DEFAULT_CORPUS_DIR = Path("eval_output/stdcorpora")
ROME_URL = "https://graphdrawing.unipg.it/data/rome-graphml.tgz"
NORTH_URL = "https://graphdrawing.unipg.it/data/north-graphml.tgz"
SUITESPARSE_URL_TEMPLATE = "https://sparse.tamu.edu/MM/HB/{name}.tar.gz"
PER_TEXT_CORPUS = 25
MAX_NODES = 2000
MAX_EDGES = 200_000
MIN_NODES = 3
MIN_EDGES = 1

# Pre-declared SuiteSparse training matrices (HB group): classic structural
# sparse meshes, all square with n<=2000 per collection metadata, chosen by
# NAME/SIZE METADATA ONLY. None appear in stdcorpora (the 140); the sealed
# bcspwr02-06 / bcsstk02-04 / *_bus files are untouchable and NOT here.
SUITESPARSE_TRAIN_NAMES = (
    "can_61",
    "can_73",
    "can_96",
    "can_144",
    "can_161",
    "can_187",
    "can_229",
    "can_256",
    "dwt_59",
    "dwt_66",
    "dwt_72",
    "dwt_87",
    "dwt_162",
    "dwt_193",
    "dwt_209",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments (``None`` = ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--per-corpus", type=int, default=PER_TEXT_CORPUS)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing corpus against MANIFEST.json instead of building",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild over an existing training corpus"
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocated_stems(corpus_dir: Path) -> Dict[str, Set[str]]:
    """Return the per-corpus stems of the 140 allocated (dev+sealed) files.

    Parameters
    ----------
    corpus_dir : Path
        stdcorpora root holding SUBSET.json + SEALED_REMAINDER.json.

    Returns
    -------
    Dict[str, Set[str]]
        ``{corpus: {stem, ...}}`` over dev and sealed allocations combined.
    """
    subset = json.loads((corpus_dir / "SUBSET.json").read_text())
    sealed = json.loads((corpus_dir / "SEALED_REMAINDER.json").read_text())
    stems: Dict[str, Set[str]] = {}
    total = 0
    for entry in list(subset["selected"]) + list(sealed["sealed"]):
        stems.setdefault(entry["corpus"], set()).add(Path(entry["relpath"]).stem)
        total += 1
    if total != 140:
        raise RuntimeError(f"expected 140 allocated files, found {total}")
    return stems


def fetch(url: str, dest: Path) -> None:
    """Download a URL to a file with curl (two attempts).

    Parameters
    ----------
    url : str
        Source URL.
    dest : Path
        Destination file.

    Raises
    ------
    RuntimeError
        When both attempts fail.
    """
    for _attempt in range(2):
        result = subprocess.run(
            ["curl", "-L", "--fail", "--max-time", "300", "-sS", "-o", str(dest), url],
            check=False,
        )
        if result.returncode == 0:
            return
        dest.unlink(missing_ok=True)
    raise RuntimeError(f"download failed after 2 attempts: {url}")


def gate_graph(path: Path, corpus: str) -> Optional[Dict[str, Any]]:
    """Load a candidate through the holdout loaders and apply the size gate.

    Parameters
    ----------
    path : Path
        Candidate graph file.
    corpus : str
        Corpus name (drives the holdout directedness policy).

    Returns
    -------
    Dict[str, Any] | None
        ``{"nodes": ..., "edges": ..., "directed": ...}`` when accepted,
        ``None`` when the file fails to load or falls outside the gate.
    """
    loader = SUPPORTED_LOADERS.get(path.suffix.lower())
    if loader is None:
        return None
    override, _source = directedness_policy(corpus, path)
    try:
        loaded = loader(path, directed_override=override)
    except Exception:  # noqa: BLE001 - candidate rejection, mirrored from load_phase
        return None
    nodes = loaded.graph.num_nodes
    edges = int(loaded.graph.edge_index.shape[1])
    if not (MIN_NODES <= nodes <= MAX_NODES and MIN_EDGES <= edges <= MAX_EDGES):
        return None
    return {"nodes": nodes, "edges": edges, "directed": loaded.directed}


def select_text_corpus(
    archive: Path,
    corpus: str,
    excluded_stems: Set[str],
    dest: Path,
    count: int,
    workdir: Path,
) -> List[Dict[str, Any]]:
    """Select training graphs from a graphdrawing archive deterministically.

    Parameters
    ----------
    archive : Path
        Downloaded ``.tgz`` archive.
    corpus : str
        Corpus name (``rome`` or ``north``).
    excluded_stems : Set[str]
        Stems of allocated (dev+sealed) files to exclude.
    dest : Path
        Output corpus directory.
    count : int
        Number of graphs to accept.
    workdir : Path
        Scratch directory for extraction.

    Returns
    -------
    List[Dict[str, Any]]
        Manifest records for the accepted graphs.
    """
    extract_dir = workdir / corpus
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(extract_dir, filter="data")
    candidates = sorted(
        (
            p
            for p in extract_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_LOADERS
        ),
        key=lambda p: p.name,
    )
    dest.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    seen_stems: Set[str] = set()
    for candidate in candidates:
        if len(records) >= count:
            break
        stem = candidate.stem
        if stem in excluded_stems or stem in seen_stems:
            continue
        gate = gate_graph(candidate, corpus)
        if gate is None:
            continue
        seen_stems.add(stem)
        target = dest / candidate.name
        shutil.copyfile(candidate, target)
        records.append(
            {
                "relpath": f"{corpus}/{candidate.name}",
                "sha256": sha256_file(target),
                "source": f"{ROME_URL if corpus == 'rome' else NORTH_URL}#{candidate.name}",
                **gate,
            }
        )
    if len(records) < count:
        raise RuntimeError(f"{corpus}: only {len(records)}/{count} candidates survived the gate")
    return records


def select_suitesparse(excluded_stems: Set[str], dest: Path, workdir: Path) -> List[Dict[str, Any]]:
    """Download and gate the pre-declared SuiteSparse training matrices.

    Parameters
    ----------
    excluded_stems : Set[str]
        Stems of allocated (dev+sealed) suitesparse files (collision guard).
    dest : Path
        Output corpus directory.
    workdir : Path
        Scratch directory for downloads.

    Returns
    -------
    List[Dict[str, Any]]
        Manifest records for the accepted matrices.
    """
    collisions = sorted(set(SUITESPARSE_TRAIN_NAMES) & excluded_stems)
    if collisions:
        raise RuntimeError(f"pre-declared suitesparse names collide with the 140: {collisions}")
    dest.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for name in SUITESPARSE_TRAIN_NAMES:
        url = SUITESPARSE_URL_TEMPLATE.format(name=name)
        tarball = workdir / f"{name}.tar.gz"
        fetch(url, tarball)
        extract_dir = workdir / name
        with tarfile.open(tarball) as tar:
            tar.extractall(extract_dir, filter="data")
        mtx_files = sorted(extract_dir.rglob("*.mtx"))
        if not mtx_files:
            raise RuntimeError(f"{name}: no .mtx in {url}")
        source_mtx = mtx_files[0]
        gate = gate_graph(source_mtx, "suitesparse")
        if gate is None:
            raise RuntimeError(f"{name}: failed the load/size gate")
        target = dest / f"{name}.mtx"
        shutil.copyfile(source_mtx, target)
        records.append(
            {
                "relpath": f"suitesparse/{name}.mtx",
                "sha256": sha256_file(target),
                "source": url,
                **gate,
            }
        )
    return records


def run_verify(train_dir: Path) -> int:
    """Verify an existing training corpus against its manifest.

    Parameters
    ----------
    train_dir : Path
        Training corpus root.

    Returns
    -------
    int
        0 when every file matches; 1 otherwise.
    """
    manifest_path = train_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    problems: List[str] = []
    for record in manifest["files"]:
        path = train_dir / record["relpath"]
        if not path.is_file():
            problems.append(f"missing: {record['relpath']}")
        elif sha256_file(path) != record["sha256"]:
            problems.append(f"sha drift: {record['relpath']}")
    if problems:
        for line in problems:
            print(line)
        return 1
    print(f"verified {len(manifest['files'])} training files against {manifest_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments (``None`` = ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    if args.verify:
        return run_verify(args.train_dir)

    manifest_path = args.train_dir / "MANIFEST.json"
    if manifest_path.is_file() and not args.force:
        print(f"ERROR: {manifest_path} exists; pass --force to rebuild", file=sys.stderr)
        return 3

    stems = allocated_stems(args.corpus_dir)
    with tempfile.TemporaryDirectory(prefix="sprint2_train_") as scratch:
        workdir = Path(scratch)
        rome_archive = workdir / "rome.tgz"
        north_archive = workdir / "north.tgz"
        fetch(ROME_URL, rome_archive)
        fetch(NORTH_URL, north_archive)
        rome_records = select_text_corpus(
            rome_archive,
            "rome",
            stems.get("rome", set()),
            args.train_dir / "rome",
            args.per_corpus,
            workdir,
        )
        north_records = select_text_corpus(
            north_archive,
            "north",
            stems.get("north", set()),
            args.train_dir / "north",
            args.per_corpus,
            workdir,
        )
        suitesparse_records = select_suitesparse(
            stems.get("suitesparse", set()), args.train_dir / "suitesparse", workdir
        )

    files = rome_records + north_records + suitesparse_records
    manifest = {
        "purpose": (
            "sprint-2 TRAINING graphs: disjoint from the 140 allocated stdcorpora "
            "files (63 dev + 77 sealed); never part of any holdout"
        ),
        "selection_rule": {
            "rome_north": (
                "archive members with supported extensions sorted by basename ascending; "
                "exclude stems of the 140 allocated files; accept in order those that "
                f"load via the holdout loaders+directedness policy with {MIN_NODES}<=N<="
                f"{MAX_NODES} and {MIN_EDGES}<=E<={MAX_EDGES}; first "
                f"{args.per_corpus} per corpus"
            ),
            "suitesparse": (
                "pre-declared HB name list (square, n<=2000 by collection metadata, "
                "name/size metadata only); sealed bcspwr02-06 etc. deliberately excluded"
            ),
        },
        "sources": {
            "rome": ROME_URL,
            "north": NORTH_URL,
            "suitesparse": SUITESPARSE_URL_TEMPLATE,
        },
        "counts": {
            "rome": len(rome_records),
            "north": len(north_records),
            "suitesparse": len(suitesparse_records),
        },
        "files": files,
    }
    args.train_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(
        f"built {len(files)} training graphs -> {args.train_dir} "
        f"(rome {len(rome_records)}, north {len(north_records)}, "
        f"suitesparse {len(suitesparse_records)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
