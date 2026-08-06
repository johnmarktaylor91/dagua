"""GLaDOS holdout runner: score native vs the full engine field on sealed corpora.

Implements the pre-registered protocol of PLAN_FABLE_R2.md section 7 (see
GLADOS_RUNNER_SPEC.md for the authoritative build spec). Pre-registered
invocation (plan 7.3, verbatim):

    python scripts/glados_holdout_run.py --corpus-dir eval_output/stdcorpora \\
      --subset eval_output/stdcorpora/SUBSET.json --output-dir \\
      eval_output/glados_holdout --seed 42 --native-deterministic \\
      --native-timeout 1800 --engine-timeout 300 --workers 10 --resume

Design pins (each backed by a Wave-1 finding):
- V3 scoring is IMPORTED from scripts.native_sprint_score (``score_position``
  with ``ruler="v3"``; NEVER ``score_group``, which silently scores v2 --
  WP10-F15). Champion selection and the tie band come from
  ``best_rows_by_graph`` / ``classify`` (symmetric-G6 + degeneracy handled
  inside; no extra filters).
- Native rows run FIRST, serial, inside ``deterministic_native_runtime`` in a
  spawned child, calling the adapter with ``deterministic_native=True``
  (WP03-F02 seam). Fresh ``LayoutConfig`` + budget ledger per row hold by
  construction: one child process per row, and the adapter builds both inside
  ``layout()`` (WP02A-F02).
- Spawn context for ALL rows (fork-after-torch deadlock -- WP10-F26).
- Per-child RSS watchdog + system memory floor (the r80 ~101GB OOM lesson,
  WP10-F09); parent warn/abort guard reused from r79.
- Per-file load isolation -> LOAD_ERROR rows (WP10-F01); explicit per-corpus
  directedness policy, never path-substring inference (WP10-F02/F04/F16);
  load-sanity gate -> LOAD_SUSPECT (WP10-F03); duplicate-name abort
  (WP10-F05); N<=2000 AND E<=200000 verification with per-row scale-gate
  logging (WP03-F01/CB-5, WP13-F03).
- Resume: seed-aware ``build_record_key`` keys (WP10-F14), fsynced JSONL
  appends with torn-final-line tolerance (WP10-F12), dot-temp cleanup +
  dotfile-ignoring validate (WP10-F13).

Development note: this runner was built and tested on synthetic fixtures only
(tests/fixtures/glados/); real corpora stay off-limits until Phase G (R-12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from scripts.r79_stdcorpora_eval import (  # noqa: E402
    GC_TRIM_INTERVAL_ROWS,
    RSS_ABORT_BYTES,
    RSS_WARN_BYTES,
    current_rss_bytes,
    json_clean,
    release_native_heap,
    safe_component,
)
from scripts.stdcorpora_loaders import (  # noqa: E402
    LoadedGraph,
    _numeric_lines,
    load_gml_file,
    load_graph_file,
    load_graphml_file,
    load_mtx_file,
)

DEFAULT_CORPUS_DIR = Path("eval_output/stdcorpora")
DEFAULT_OUTPUT_DIR = Path("eval_output/glados_holdout")
DEFAULT_SEED = 42
DEFAULT_NATIVE_TIMEOUT = 1800.0
DEFAULT_ENGINE_TIMEOUT = 300.0
DEFAULT_WORKERS = 10
MAX_FIELD_WORKERS = 10  # plan 7.3: field engines run with workers<=10
DEFAULT_SEED_COUNT = 10  # cert DEFAULT_SEED_COUNT (run_benchmark.py)
DEFAULT_MAX_NODES = 2000
# Frozen scale router gates on edges too (CB-5: edge-only gate at E>200K would
# route a dense n<=2000 graph onto the uncertified scale path); the runner
# verifies BOTH bounds and excludes-with-reason anything above either.
DEFAULT_MAX_EDGES = 200_000
DEFAULT_CHILD_RSS_ABORT_GB = 48.0
DEFAULT_MIN_AVAIL_GB = 15.0
# Reject .mtx files whose HEADER declares a dimension above this before ever
# invoking the loader (parent-process wedge guard, dry-well B4-F7 class).
# 25_000 = below the reproduced 50k-node parent wedge, 12.5x the --max-nodes
# 2000 acceptance default (Sol round-2 F4).
MTX_DECLARED_DIM_CAP = 25_000
CHILD_JOIN_GRACE_SECONDS = 60.0
CHILD_POLL_SECONDS = 2.0
CORPUS_NAMES = ("rome", "north", "suitesparse")
SUPPORTED_LOADERS = {
    ".graph": load_graph_file,
    ".gml": load_gml_file,
    ".graphml": load_graphml_file,
    ".mtx": load_mtx_file,
}

# Preflight-required modules: psutil is the memory guard (WP10-F10: silently
# disabled when missing); numba + swiglpk change native-supporting op behavior
# when absent (WP02A-F09/WP-05), so their absence would un-certify the rows.
PREFLIGHT_REQUIRED_MODULES = ("psutil", "numba", "swiglpk")

# A3 engine field, populated from WP-09's corpus-ingestion capability matrix
# (findings/WP-09_FINDINGS.md section 4) at build time: every registered
# cert-pool engine except native itself and the broken-as-coded exclusions
# below. Conditional engines (size caps, planarity gates, tree coercion) stay
# IN the field -- they produce recorded per-row skips or clean error rows,
# which is the honest treatment. --engines-file overrides for dry-runs/gates.
GLADOS_ENGINE_FIELD: Tuple[str, ...] = (
    "backbone",
    "backbone_reimpl",
    "circo_reimpl",
    "classic_classical_mds",
    "classic_davidson_harel",
    "classic_drl",
    "classic_fa2",
    "classic_fcose",
    "classic_fmmm",
    "classic_fr",
    "classic_fr_kk",
    "classic_gem",
    "classic_graphopt",
    "classic_kk",
    "classic_kk_fr",
    "classic_lgl",
    "classic_linlog",
    "classic_maxent_stress",
    "classic_neato",
    "classic_neulay",
    "classic_pivot_mds",
    "classic_rt",
    "classic_sfdp",
    "classic_sgd2_multi",
    "classic_spectral",
    "classic_stress_maj",
    "classic_stress_sgd",
    "classic_sugiyama",
    "classic_tsnet",
    "classic_umap",
    "coregd_reference",
    "coregd_reimpl",
    "cytoscape",
    "cytoscape_fcose",
    "d3_cluster_radial_reimpl",
    "d3_cluster_reimpl",
    "d3_tree_radial_reimpl",
    "d3_tree_reimpl",
    "d3dag",
    "d3force",
    "d3force_reimpl",
    "d3hierarchy",
    "dagre",
    "dagre_reimpl",
    "deepgd_reference",
    "deepgd_reimpl",
    "dot",
    "drgraph_reference",
    "drgraph_reimpl",
    "elk_force",
    "elk_force_reimpl",
    "elk_layered",
    "elk_layered_reimpl",
    "elk_mrtree",
    "elk_mrtree_reimpl",
    "elk_radial",
    "elk_radial_reimpl",
    "elk_stress",
    "elk_stress_reimpl",
    "fa2_ref",
    "fdp",
    "gephi_yifanhu",
    "graphviz_circo",
    "graphviz_dot",
    "graphviz_fdp",
    "graphviz_neato",
    "graphviz_osage",
    "graphviz_sfdp",
    "graphviz_twopi",
    "grip_reference",
    "grip_reimpl",
    "igraph_davidson_harel",
    "igraph_drl",
    "igraph_fr",
    "igraph_graphopt",
    "igraph_kamada_kawai",
    "igraph_lgl",
    "igraph_mds",
    "igraph_rt",
    "igraph_rt_circular",
    "igraph_rt_horizontal",
    "igraph_sugiyama",
    "isom_jung",
    "isom_reimpl",
    "largevis_reference",
    "largevis_reimpl",
    "linlog",
    "mulment_reference",
    "mulment_reimpl",
    "neulay",
    "nnpnet_reference",
    "nnpnet_reimpl",
    "nx_arf",
    "nx_arf_reimpl",
    "nx_bfs",
    "nx_bfs_reimpl",
    "nx_bipartite",
    "nx_bipartite_reimpl",
    "nx_circular",
    "nx_circular_reimpl",
    "nx_kamada_kawai",
    "nx_multipartite",
    "nx_multipartite_reimpl",
    "nx_planar",
    "nx_planar_reimpl",
    "nx_shell",
    "nx_shell_reimpl",
    "nx_spectral",
    "nx_spectral_random_walk",
    "nx_spiral",
    "nx_spiral_reimpl",
    "nx_spring",
    "ogdf_balloon",
    "ogdf_balloon_reimpl",
    "ogdf_bertault",
    "ogdf_bertault_reimpl",
    "ogdf_davidson_harel",
    "ogdf_fmmm",
    "ogdf_fpp",
    "ogdf_fpp_reimpl",
    "ogdf_gem",
    "ogdf_pivot_mds",
    "ogdf_schnyder",
    "ogdf_schnyder_reimpl",
    "ogdf_stress",
    "ogdf_sugiyama",
    "ogdf_sugiyama_reimpl",
    "omega_reference",
    "omega_reimpl",
    "openord",
    "openord_reimpl",
    "osage_reimpl",
    "pacmap",
    "pacmap_reimpl",
    "sgd2",
    "sgd2_mds",
    "sgd2_multi_ref",
    "sklearn_smacof_nonmetric",
    "smacof_nonmetric_reimpl",
    "smartgd_reference",
    "smartgd_reimpl",
    "sparse_stress",
    "sparse_stress_reimpl",
    "tfdp",
    "tfdp_reimpl",
    "tidy_reference",
    "tidy_reimpl",
    "tsne_graph",
    "twopi_reimpl",
    "umap_graph",
    "webcola",
    "word2vecgd",
    "word2vecgd_reimpl",
)

# Engine families absent from the built-in field, with machine-readable
# reasons for the report (spec section 6). These are WP-09's broken-as-coded
# verdicts; re-adding them is conditional on WP-24b's fixes merging (then CC
# updates this constant or passes --engines-file).
GLADOS_ENGINE_EXCLUSIONS: Dict[str, str] = {
    # (2026-08-05 integrator) The three WP-09 broken-as-coded exclusions
    # (largevis_reference, drgraph_reference, tidy_reference) were lifted
    # after the WP-24b adapter fixes merged; they are back in the field
    # above, alongside the newly wired webcola/d3dag families.
}

# Parent-process memory guard knobs (module attributes so tests can
# monkeypatch them; values imported from the r79 harness).
PARENT_GC_TRIM_INTERVAL_ROWS = GC_TRIM_INTERVAL_ROWS
PARENT_RSS_WARN_BYTES = RSS_WARN_BYTES
PARENT_RSS_ABORT_BYTES = RSS_ABORT_BYTES
parent_rss_bytes = current_rss_bytes

_GML_DIRECTED_RE = re.compile(rb"\bdirected\s+([01])\b")
_GRAPHML_EDGEDEFAULT_RE = re.compile(rb"edgedefault\s*=\s*[\"'](directed|undirected)[\"']")


class PreflightError(RuntimeError):
    """Raised when a preflight assertion fails (exit code 4)."""


class LoadPhaseFatal(RuntimeError):
    """Raised on fatal load-phase conditions (exit code 5)."""


@dataclass
class GraphEntry:
    """One loaded corpus graph plus runner-side metadata."""

    name: str
    corpus: str
    relpath: str
    loaded: LoadedGraph
    directed: bool
    directed_source: str
    telemetry: Dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Argument list; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options with all spec defaults baked in so the pre-registered
        plan-7.3 invocation is self-sufficient.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="SUBSET.json path; REQUIRED for the real corpus dir (R-12 guard).",
    )
    parser.add_argument(
        "--allow-full-corpus",
        action="store_true",
        help="Explicitly allow running the real corpus dir without --subset.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--native-deterministic", action="store_true")
    parser.add_argument("--native-timeout", type=float, default=DEFAULT_NATIVE_TIMEOUT)
    parser.add_argument("--engine-timeout", type=float, default=DEFAULT_ENGINE_TIMEOUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help=(
            "Explicitly allow a non-resume run to destroy existing staging/"
            "published run data in the output dir (dry-well B4-F6 guard)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEED_COUNT,
        help="Seed-battery size for stochastic field engines (cert default 10).",
    )
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES)
    parser.add_argument(
        "--engines-file",
        type=Path,
        default=None,
        help="JSON array of engine names overriding the built-in field.",
    )
    parser.add_argument("--score-workers", type=int, default=None)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--child-rss-abort-gb", type=float, default=DEFAULT_CHILD_RSS_ABORT_GB)
    parser.add_argument("--min-avail-gb", type=float, default=DEFAULT_MIN_AVAIL_GB)
    parser.add_argument(
        "--report-context",
        type=Path,
        default=None,
        help='JSON path: {"item4_strict": N, "scale_fix_note": "...", "ruler_ledger_path": "..."}.',
    )
    parser.add_argument(
        "--rss-guard-selftest",
        action="store_true",
        help="Fake one over-ceiling child RSS poll on the first child (guard drill).",
    )
    parser.add_argument(
        "--accept-revision-drift",
        action="store_true",
        help=(
            "Resume escape hatch for the harness-only-hotfix case: KEEP resumed "
            "rows whose git SHA drifted but whose engine source component is "
            "unchanged, disclosing them in results.json and the report instead "
            "of quarantining (R2 B4-Sol-2)."
        ),
    )
    parser.add_argument(
        "--retry-memkills",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Re-run environmental memkill:* ERROR rows on resume (default on; "
            f"capped at {MEMKILL_MAX_RETRIES} retries per row -- R2 B4-F4)."
        ),
    )
    args = parser.parse_args(argv)
    args.workers = max(1, min(int(args.workers), MAX_FIELD_WORKERS))
    if args.score_workers is None:
        args.score_workers = min(args.workers, 10)
    args.score_workers = max(1, int(args.score_workers))
    return args


def git_provenance(repo_root: Path) -> Tuple[str, bool]:
    """Return the current git SHA and dirty flag for the repo root.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.

    Returns
    -------
    Tuple[str, bool]
        ``(sha, dirty)``; SHA is ``"unknown"`` when git is unavailable.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha or "unknown", bool(status.strip())


def preflight(args: argparse.Namespace) -> Dict[str, Any]:
    """Run all hard preflight assertions; abort exit 4 on any failure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.

    Returns
    -------
    Dict[str, Any]
        Provenance payload (module path, git sha/dirty, disk free).

    Raises
    ------
    PreflightError
        On any failed assertion.
    """
    repo_root = Path(__file__).resolve().parents[1]

    import dagua

    module_path = str(Path(dagua.__file__).resolve())
    if not module_path.startswith(str(repo_root)):
        raise PreflightError(
            f"dagua imports from {module_path}, not the repo containing this "
            f"runner ({repo_root}); editable-install contamination"
        )

    import importlib

    for module_name in PREFLIGHT_REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            raise PreflightError(f"required module {module_name!r} not importable: {exc}") from exc

    _preflight_native_env()

    free_bytes = shutil.disk_usage("/").free
    min_bytes = args.min_avail_gb * 1024**3
    if free_bytes < min_bytes:
        raise PreflightError(
            f"/ has {free_bytes / 1024**3:.1f}GB free < required {args.min_avail_gb:.0f}GB"
        )

    corpus_posix = args.corpus_dir.resolve().as_posix()
    if (
        "eval_output/stdcorpora" in corpus_posix
        and args.subset is None
        and not args.allow_full_corpus
    ):
        raise PreflightError(
            f"corpus dir {corpus_posix} looks like the real holdout corpus; "
            "--subset is required (or pass --allow-full-corpus explicitly)"
        )
    if args.subset is not None and not args.subset.is_file():
        raise PreflightError(f"subset file not found: {args.subset}")

    sha, dirty = git_provenance(repo_root)
    return {
        "dagua_module_path": module_path,
        "repo_root": str(repo_root),
        "git_sha": sha,
        "git_dirty": dirty,
        "disk_free_gb": round(free_bytes / 1024**3, 2),
    }


# Environment variables that silently change engine behavior when set; the
# sacred run must start from a clean slate (WP02A-F01 + dry-well B5-F01).
PREFLIGHT_FORBIDDEN_ENV: Dict[str, str] = {
    "DAGUA_NATIVE_DISABLE_W5": "disables the W5 finisher stage and weakens native rows",
    "DAGUA_W5_TELEMETRY_PATH": "arms W5 telemetry writes inside the measured native path",
    "DAGUA_ARM_TELEMETRY_PATH": "arms per-arm telemetry writes inside the measured native path",
    "DAGUA_DISABLE_NUMBA": "silently swaps field-op implementations away from the certified path",
    "DAGUA_SGD2_MULTI_ALLOW_CLONE": "re-enables the sgd2_multi network clone side effect",
    "DAGUA_FDP_TRACE": "arms the fdp/fmmm trace writer (multi-GB dumps) on field rows",
    "NUMBA_DISABLE_JIT": "numba's own kill-switch; swaps field-op numerics off the certified path",
}


def _preflight_native_env() -> None:
    """Assert the native deterministic environment is uncontaminated.

    Checks that every :data:`PREFLIGHT_FORBIDDEN_ENV` variable is unset
    (WP02A-F01; dry-well B5-F01 confirmed telemetry/numba/clone knobs were
    unguarded) and that a fresh ``LayoutConfig`` carries no
    ``_dagua_native_deadline_s`` attribute (class-level contamination would
    put wall-clock reads back on the deterministic path).

    Raises
    ------
    PreflightError
        On any contamination signal.
    """
    for variable, effect in PREFLIGHT_FORBIDDEN_ENV.items():
        value = os.environ.get(variable)
        if value is not None:
            raise PreflightError(f"{variable} is set ({value!r}); unset it -- it {effect}")
    from dagua.config import LayoutConfig

    config = LayoutConfig()
    if hasattr(config, "_dagua_native_deadline_s"):
        raise PreflightError(
            "fresh LayoutConfig already carries _dagua_native_deadline_s; "
            "wall-clock deadline contamination on the deterministic path"
        )


def _native_layout_kwargs(
    timeout_s: float, seed: Optional[int], deterministic: bool
) -> Dict[str, Any]:
    """Return the native adapter call kwargs (pinned by tests).

    Parameters
    ----------
    timeout_s : float
        Native row timeout in seconds.
    seed : int | None
        Run seed.
    deterministic : bool
        Whether the deterministic envelope is active.

    Returns
    -------
    Dict[str, Any]
        Keyword arguments for ``get_competitor("dagua").layout``. When
        deterministic, ``deterministic_native=True`` is passed explicitly
        (WP03-F02 seam); otherwise the kwarg is omitted, mirroring
        run_benchmark's worker.
    """
    kwargs: Dict[str, Any] = {"timeout": timeout_s, "seed": seed}
    if deterministic:
        kwargs["deterministic_native"] = True
    return kwargs


def directedness_policy(corpus: str, path: Path) -> Tuple[Optional[bool], str]:
    """Return the per-corpus directedness override and its provenance.

    Never derives directedness from full-path substrings (the "dagua"
    contains "dag" trap, WP10-F02). Policy table (spec 5.2):

    - rome, suitesparse: undirected, always.
    - north ``.gml``/``.graphml``: explicit direction metadata in the file is
      trusted; metadata absent -> directed (matches the North GraphML
      fallback and its pinned test). Metadata is detected by a cheap scan of
      the file head (4KB; superset of the spec's "first KB").
    - north other formats: ``.mtx`` undirected (sparsity pattern), else
      directed by policy.
    - misc (scratch/dry-run corpora): loader default (no override).

    Parameters
    ----------
    corpus : str
        Runner-classified corpus name.
    path : Path
        Source file path.

    Returns
    -------
    Tuple[bool | None, str]
        ``(directed_override, directed_source)`` where the source is
        ``"format"``, ``"policy"``, or ``"loader_default"``.
    """
    suffix = path.suffix.lower()
    if corpus in {"rome", "suitesparse"}:
        return False, "policy"
    if corpus == "north":
        if suffix == ".mtx":
            return False, "policy"
        if suffix in {".gml", ".graphml"}:
            try:
                head = path.open("rb").read(4096)
            except OSError:
                head = b""
            if suffix == ".gml":
                match = _GML_DIRECTED_RE.search(head)
                if match:
                    return match.group(1) == b"1", "format"
            else:
                match = _GRAPHML_EDGEDEFAULT_RE.search(head)
                if match:
                    return match.group(1) == b"directed", "format"
            return True, "policy"
        return True, "policy"
    return None, "loader_default"


def _classify_corpus(relpath: str) -> str:
    """Classify the corpus from the path relative to the corpus dir.

    Parameters
    ----------
    relpath : str
        POSIX relative path below the corpus dir.

    Returns
    -------
    str
        First path segment when it is a known corpus name, else ``"misc"``.
    """
    parts = Path(relpath).parts
    if len(parts) >= 2 and parts[0] in CORPUS_NAMES:
        return parts[0]
    return "misc"


def _graph_parse_branch(rows: List[List[int]]) -> str:
    """Mirror ``load_graph_file``'s branch choice for telemetry (WP10-F08).

    Parameters
    ----------
    rows : List[List[int]]
        Output of ``_numeric_lines``.

    Returns
    -------
    str
        ``"adjacency_list"``, ``"padded_adjacency"``, or ``"edge_list"``.
    """
    if not rows:
        return "empty"
    header = rows[0]
    if len(header) == 1 and len(rows) >= header[0] + 1:
        return "adjacency_list"
    if len(header) >= 2 and len(rows) >= header[0] + 1:
        return "padded_adjacency"
    return "edge_list"


def _mtx_header_dims(path: Path) -> Optional[Tuple[int, int]]:
    """Read the Matrix Market size row's (rows, cols) for telemetry.

    Parameters
    ----------
    path : Path
        Matrix Market file.

    Returns
    -------
    Tuple[int, int] | None
        Declared dimensions, or ``None`` when unparsable.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
                return None
    except (OSError, ValueError):
        return None
    return None


class _CountingBuildGraph:
    """Context manager wrapping ``stdcorpora_loaders.build_graph`` with counters.

    Loader telemetry (WP10-F08) without editing ``build_graph``: the wrapper
    materializes the edge iterable, counts self-loop / out-of-range /
    duplicate drops using the exact same rules, then delegates to the
    original function with the materialized list.
    """

    def __init__(self) -> None:
        self.record: Dict[str, Any] = {}

    def __enter__(self) -> "_CountingBuildGraph":
        import scripts.stdcorpora_loaders as loaders_mod

        self._loaders_mod = loaders_mod
        self._original = loaders_mod.build_graph
        record = self.record
        original = self._original

        def counting_build_graph(
            name: str,
            corpus: str,
            node_count: int,
            edges: Iterable[Tuple[int, int]],
            directed: bool,
            source_path: Path,
        ) -> LoadedGraph:
            edge_list = list(edges)
            self_loops = 0
            out_of_range = 0
            duplicates = 0
            seen: Set[Tuple[int, int]] = set()
            for source, target in edge_list:
                if source == target:
                    self_loops += 1
                    continue
                if source < 0 or target < 0 or source >= node_count or target >= node_count:
                    out_of_range += 1
                    continue
                key = (source, target) if directed else tuple(sorted((source, target)))
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
            record.update(
                raw_edges=len(edge_list),
                self_loops_dropped=self_loops,
                out_of_range_dropped=out_of_range,
                duplicates_dropped=duplicates,
                edges_kept=len(seen),
            )
            return original(name, corpus, node_count, edge_list, directed, source_path)

        loaders_mod.build_graph = counting_build_graph
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._loaders_mod.build_graph = self._original


def load_phase(
    corpus_dir: Path,
    subset_relpaths: Optional[Set[str]],
    max_nodes: int,
    max_edges: int,
) -> Tuple[List[GraphEntry], List[Dict[str, Any]], Dict[str, Any]]:
    """Load corpus files with per-file isolation, policy, and sanity gates.

    Parameters
    ----------
    corpus_dir : Path
        Corpus root directory.
    subset_relpaths : Set[str] | None
        Blind-subset relpaths to keep; ``None`` keeps every supported file.
    max_nodes : int
        Node-count acceptance bound.
    max_edges : int
        Edge-count acceptance bound (scale-gate verification, CB-5).

    Returns
    -------
    Tuple[List[GraphEntry], List[Dict[str, Any]], Dict[str, Any]]
        ``(entries, load_rows, stats)``: accepted graphs, per-file problem
        rows (LOAD_ERROR / LOAD_SUSPECT / excluded), and counters.

    Raises
    ------
    LoadPhaseFatal
        On duplicate loaded names (WP10-F05) -- exit code 5 upstream.
    """
    entries: List[GraphEntry] = []
    load_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"sealed_skipped": 0, "supported_files": 0}
    seen_relpaths: Set[str] = set()

    for path in sorted(corpus_dir.rglob("*")):
        loader = SUPPORTED_LOADERS.get(path.suffix.lower())
        if loader is None or not path.is_file():
            continue
        stats["supported_files"] += 1
        relpath = path.relative_to(corpus_dir).as_posix()
        if subset_relpaths is not None:
            if relpath not in subset_relpaths:
                stats["sealed_skipped"] += 1
                continue
            seen_relpaths.add(relpath)
        corpus = _classify_corpus(relpath)
        name = f"{corpus}/{path.stem}"
        override, directed_source = directedness_policy(corpus, path)

        telemetry: Dict[str, Any] = {"format": path.suffix.lower().lstrip(".")}
        suffix = path.suffix.lower()
        if suffix == ".graph":
            numeric_rows = _numeric_lines(path)
            telemetry["raw_numeric_rows"] = len(numeric_rows)
            telemetry["parse_branch"] = _graph_parse_branch(numeric_rows)
        elif suffix == ".mtx":
            dims = _mtx_header_dims(path)
            telemetry["mtx_dims"] = list(dims) if dims else None
            telemetry["parse_branch"] = "mtx"
            # Wedge-class pre-guard (dry-well B4-F7 companion): a declared
            # dimension in the billions would pre-allocate that many nodes
            # in the PARENT before any size filter runs. Reject from the
            # header alone, without invoking the loader.
            if dims and max(dims[0], dims[1]) > MTX_DECLARED_DIM_CAP:
                load_rows.append(
                    {
                        "graph": name,
                        "corpus": corpus,
                        "relpath": relpath,
                        "status": "SKIP",
                        "reason": (f"mtx_dims_cap:{dims[0]}x{dims[1]}>{MTX_DECLARED_DIM_CAP}"),
                        "source_path": str(path),
                        "telemetry": telemetry,
                    }
                )
                continue
        else:
            telemetry["raw_numeric_rows"] = None
            telemetry["parse_branch"] = suffix.lstrip(".")

        try:
            with _CountingBuildGraph() as counting:
                loaded = loader(path, directed_override=override)
            telemetry.update(counting.record)
        except Exception as exc:  # noqa: BLE001  (per-file isolation, WP10-F01)
            load_rows.append(
                {
                    "graph": name,
                    "corpus": corpus,
                    "relpath": relpath,
                    "status": "LOAD_ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_path": str(path),
                }
            )
            continue

        if loaded.name != name:
            telemetry["loader_name"] = loaded.name

        nodes = loaded.graph.num_nodes
        edges = int(loaded.graph.edge_index.shape[1])
        base_row = {
            "graph": name,
            "corpus": corpus,
            "relpath": relpath,
            "source_path": str(path),
            "nodes": nodes,
            "edges": edges,
            "telemetry": telemetry,
        }

        dims = telemetry.get("mtx_dims")
        if suffix == ".mtx" and dims and dims[0] != dims[1]:
            load_rows.append(
                {**base_row, "status": "SKIP", "reason": f"nonsquare_mtx:{dims[0]}x{dims[1]}"}
            )
            continue
        raw_rows = telemetry.get("raw_numeric_rows")
        suspect_reason: Optional[str] = None
        if nodes <= 2 and isinstance(raw_rows, int) and raw_rows > 5:
            suspect_reason = (
                f"parsed to {nodes} node(s) from {raw_rows} numeric rows "
                f"(branch {telemetry.get('parse_branch')})"
            )
        elif edges == 0 and nodes > 10:
            suspect_reason = f"parsed to 0 edges with {nodes} nodes"
        if suspect_reason is not None:
            load_rows.append({**base_row, "status": "LOAD_SUSPECT", "reason": suspect_reason})
            continue
        if nodes > max_nodes:
            load_rows.append({**base_row, "status": "SKIP", "reason": f"max_nodes:{nodes}"})
            continue
        if edges > max_edges:
            load_rows.append(
                {**base_row, "status": "SKIP", "reason": f"scale_gate_edges:{edges}>{max_edges}"}
            )
            continue

        entries.append(
            GraphEntry(
                name=name,
                corpus=corpus,
                relpath=relpath,
                loaded=loaded,
                directed=loaded.directed,
                directed_source=directed_source,
                telemetry=telemetry,
            )
        )

    if subset_relpaths is not None:
        for missing in sorted(subset_relpaths - seen_relpaths):
            load_rows.append(
                {
                    "graph": f"{_classify_corpus(missing)}/{Path(missing).stem}",
                    "corpus": _classify_corpus(missing),
                    "relpath": missing,
                    "status": "LOAD_ERROR",
                    "error": "subset file missing from corpus dir",
                    "source_path": str(corpus_dir / missing),
                }
            )

    names_seen: Dict[str, str] = {}
    collisions: List[str] = []
    for entry in entries:
        if entry.name in names_seen:
            collisions.append(f"{entry.name} <- {names_seen[entry.name]} AND {entry.relpath}")
        else:
            names_seen[entry.name] = entry.relpath
    if collisions:
        raise LoadPhaseFatal(
            "duplicate loaded graph names (position files and resume keys would "
            f"collide): {collisions}"
        )
    return entries, load_rows, stats


def engine_availability(engine_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Probe availability for the requested engines (r79 pattern).

    Parameters
    ----------
    engine_names : Sequence[str]
        Engine names to check.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        ``{"available": bool, "reason": str | None}`` keyed by engine.
    """
    from dagua.eval.competitors import get_competitor

    availability: Dict[str, Dict[str, Any]] = {}
    for engine_name in engine_names:
        competitor = get_competitor(engine_name)
        if competitor is None:
            availability[engine_name] = {"available": False, "reason": "adapter not registered"}
            continue
        try:
            available = competitor.available()
        except Exception as exc:  # noqa: BLE001
            availability[engine_name] = {"available": False, "reason": str(exc)}
            continue
        availability[engine_name] = {
            "available": bool(available),
            "reason": None if available else "adapter unavailable",
        }
    return availability


def position_relpath(graph_name: str, engine_name: str, seed: Optional[int]) -> str:
    """Return the seed-aware relative position tensor path.

    Parameters
    ----------
    graph_name : str
        Corpus graph name.
    engine_name : str
        Engine name.
    seed : int | None
        Row seed (``None`` for deterministic row keys).

    Returns
    -------
    str
        POSIX-style path below the output directory.
    """
    seed_part = "deterministic" if seed is None else f"seed{seed}"
    filename = f"{safe_component(graph_name)}__{safe_component(engine_name)}__{seed_part}.pt"
    return str(Path("positions") / filename)


# ---------------------------------------------------------------------------
# Row store (fsynced appends, torn-final-line tolerance, seed-aware keys)
# ---------------------------------------------------------------------------


def rows_path(directory: Path) -> Path:
    """Return the JSONL row-store path under a run directory.

    Parameters
    ----------
    directory : Path
        Staging or published run directory.

    Returns
    -------
    Path
        ``results.rows.jsonl`` path.
    """
    return directory / "results.rows.jsonl"


def append_row(directory: Path, row: Dict[str, Any]) -> None:
    """Append one row with flush + fsync (crash-safe, WP10-F12).

    Parameters
    ----------
    directory : Path
        Run directory containing the row store.
    row : Dict[str, Any]
        Completed row (must carry ``record_key``).

    Returns
    -------
    None
    """
    directory.mkdir(parents=True, exist_ok=True)
    with rows_path(directory).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_clean(row), sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_rows(directory: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Atomically rewrite the JSONL row store to exactly ``rows``.

    Used when resume quarantines rows: quarantined rows must be REMOVED from
    the primary store (they live only in the payload's ``quarantined_rows``)
    so a later resume that re-admits their key reruns from scratch instead
    of resurrecting a row whose tensor was orphaned (Sol round-2 F2).

    Parameters
    ----------
    directory : Path
        Run directory containing the row store.
    rows : Sequence[Dict[str, Any]]
        Rows to keep, in append order.

    Returns
    -------
    None
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = rows_path(directory)
    temp = path.with_name(path.name + ".rewrite")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_clean(row), sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def load_rows_tolerant(directory: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load the JSONL row store, tolerating a torn FINAL line only.

    A malformed final line (SIGKILL mid-append) is truncated away with a
    warning; a malformed interior line is a hard error (WP10-F12).

    Parameters
    ----------
    directory : Path
        Run directory containing the row store.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[str]]
        ``(rows, warnings)``.

    Raises
    ------
    RuntimeError
        On a malformed non-final line.
    """
    path = rows_path(directory)
    if not path.is_file():
        return [], []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                warnings.append(
                    f"torn final JSONL line truncated on resume ({len(stripped)} chars)"
                )
                # Repair ATOMICALLY (dry-well R2 B4-F3): the old in-place
                # write_text could itself be interrupted, leaving an
                # arbitrary prefix of the ONLY row store that is
                # indistinguishable from a valid shorter store. Temp + fsync
                # + rename means a crash mid-repair leaves either the old
                # file (torn line tolerated again next time) or the complete
                # repaired file -- never a silent prefix.
                try:
                    temp = path.with_name(path.name + ".repair")
                    with temp.open("w", encoding="utf-8") as handle:
                        if index:
                            handle.write("\n".join(lines[:index]) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    temp.replace(path)
                except OSError as repair_exc:
                    warnings.append(
                        f"torn-line repair failed ({type(repair_exc).__name__}: "
                        f"{repair_exc}); torn line left in place"
                    )
                break
            raise RuntimeError(
                f"malformed interior JSONL line {index + 1} in {path}: {exc}"
            ) from exc
    return rows, warnings


def dedupe_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Collapse rows to the LAST occurrence per record key.

    Scored rows are appended after their layout rows with the same key; the
    last row for a key is authoritative.

    Parameters
    ----------
    rows : Iterable[Dict[str, Any]]
        Row-store rows in append order.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Latest row per ``record_key``.
    """
    deduped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("record_key")
        if key:
            deduped[str(key)] = row
    return deduped


def clean_child_temps(directory: Path) -> int:
    """Remove stale hidden child-temp files under ``positions/`` (WP10-F13).

    Parameters
    ----------
    directory : Path
        Run directory.

    Returns
    -------
    int
        Number of removed temp files.
    """
    position_dir = directory / "positions"
    if not position_dir.is_dir():
        return 0
    removed = 0
    for path in position_dir.glob(".*"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def compute_revision_markers(engines: Sequence[str], git_sha: str) -> Dict[str, str]:
    """Compute the per-engine run-revision marker (dry-well R2 B4-Sol-2).

    Rows are stamped ``"<git_sha>:<source_component>"`` at creation so resume
    can detect a mid-run implementation hotfix that does NOT touch scoring
    sources (which would leave ``scoring_signature`` unchanged and silently
    mix pre-fix and post-fix layouts under the current run's advertised SHA).

    Components: field rows reuse benchmark.py's ``_adapter_source_signature``
    (the adapter's implementing module + shared base scaffolding); native
    rows use ``_dagua_source_signature`` (all non-eval dagua source, which
    also covers the pipeline code reimpl adapters execute). Any COMMITTED
    hotfix flips the git-SHA component for every row; the source component
    then discriminates harness-only commits (component unchanged -> eligible
    for ``--accept-revision-drift``) from engine-implementation changes
    (component changed -> always quarantined). Uncommitted edits to a file
    outside both component sets (i.e. runner-only edits) change neither --
    which is exactly the class that cannot alter layouts.

    Parameters
    ----------
    engines : Sequence[str]
        Engine names in this run (including ``"dagua"`` when present).
    git_sha : str
        Current git SHA from the preflight provenance block.

    Returns
    -------
    Dict[str, str]
        ``engine -> marker`` map.
    """
    from dagua.eval.benchmark import _adapter_source_signature, _dagua_source_signature

    dagua_component: Optional[str] = None
    markers: Dict[str, str] = {}
    for engine in engines:
        if engine == "dagua":
            if dagua_component is None:
                dagua_component = _dagua_source_signature()
            component = dagua_component
        else:
            component = _adapter_source_signature(engine)
        markers[engine] = f"{git_sha}:{component}"
    return markers


def snapshot_row_store(directory: Path) -> Optional[Path]:
    """Snapshot the row store before any resume-prep mutation (R2 B4-F1).

    On a staging-only resume (crash BEFORE the first publish -- the most
    common sacred-run crash shape) the staging store is the ONLY copy of
    every completed row, and resume prep mutates it (torn-line repair,
    quarantine rewrite). The snapshot is the durable backup; it is removed
    only after a successful publish.

    Parameters
    ----------
    directory : Path
        Run directory containing the row store.

    Returns
    -------
    Path | None
        Snapshot path, or ``None`` when there is no store to snapshot.
    """
    path = rows_path(directory)
    if not path.is_file():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    snapshot = path.with_name(f"{path.name}.pre-resume-{stamp}")
    with path.open("rb") as source, snapshot.open("wb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    return snapshot


def restore_row_store_snapshot(snapshot: Path, directory: Path) -> bool:
    """Restore the row store from a resume snapshot (atomic replace).

    Parameters
    ----------
    snapshot : Path
        Snapshot file from :func:`snapshot_row_store`.
    directory : Path
        Run directory containing the row store.

    Returns
    -------
    bool
        ``True`` when the store was restored.
    """
    if not snapshot.is_file():
        return False
    path = rows_path(directory)
    temp = path.with_name(path.name + ".restore")
    with snapshot.open("rb") as source, temp.open("wb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    temp.replace(path)
    return True


def clean_row_store_snapshots(directory: Path) -> int:
    """Remove resume snapshots after a successful publish.

    Parameters
    ----------
    directory : Path
        Published run directory (snapshots travel with the rename).

    Returns
    -------
    int
        Number of snapshots removed.
    """
    removed = 0
    for snapshot in directory.glob("results.rows.jsonl.pre-resume-*"):
        if snapshot.is_file():
            snapshot.unlink()
            removed += 1
    return removed


# Environmental memkill rows (RSS ceiling / system floor) are resume-retryable
# by default (dry-well R2 B4-F4): a transient load blip must not permanently
# blind native or thin the field. Capped so a genuinely over-budget row cannot
# loop forever.
MEMKILL_MAX_RETRIES = 2


def partition_resumed_rows(
    rows: Sequence[Dict[str, Any]],
    signature: str,
    valid_keys: Set[str],
    current_seed: int,
    tensor_exists: Optional[Any] = None,
    expected_revision: Optional[Any] = None,
    accept_revision_drift: bool = False,
    retry_memkills: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """Split resumed rows into kept vs quarantined (dry-well B4-F3 + Sol).

    A resumed row is QUARANTINED -- recorded but excluded from the tally, its
    key vacated so an in-universe row is re-run/re-scored fresh -- when:

    - its record key is outside the CURRENT subset x field x seed battery
      (Sol B4-3: engines outside the field must never select the champion);
    - its run-revision marker differs from the current one for its engine
      (dry-well R2 B4-Sol-2): a mid-run implementation hotfix invalidates
      the LAYOUT itself, so unlike a scoring change the layout sibling rows
      are NOT rescued -- they carry the same stale marker and quarantine
      with it. With ``accept_revision_drift``, rows whose SOURCE component
      is unchanged (git-SHA-only drift = the allowed harness-only-hotfix
      class) are kept and disclosed instead; a changed source component or a
      missing marker quarantines regardless;
    - it carries a score under a scoring signature other than the current one
      (Sol B3-2; modeled on native_sprint_score's stale-signature rejection).
      A scored row whose layout sibling row survives is rescored from its
      existing tensor, so a mid-run scoring-policy hotfix costs rescoring,
      not layout regeneration;
    - it is a native row generated under a different ``--seed`` than this
      invocation (Sol B5-1: the native key is seedless by pre-registered
      convention, so the seed is checked from the row's recorded
      ``native_child_seed``; rows predating that field are quarantined
      conservatively);
    - it is an environmental ``memkill:*`` ERROR row with fewer than
      :data:`MEMKILL_MAX_RETRIES` recorded retries and ``retry_memkills`` is
      on (dry-well R2 B4-F4: a transient load blip must not permanently
      blind native or thin the field across every future resume).

    Parameters
    ----------
    rows : Sequence[Dict[str, Any]]
        Raw resumed rows (append order, pre-dedupe).
    signature : str
        Current scoring signature.
    valid_keys : Set[str]
        Record keys of the current row universe.
    current_seed : int
        Current run seed.
    tensor_exists : Callable[[str], bool] | None, default=None
        Existence check for a row's ``positions_path`` (relative). When
        provided, kept OK rows must still OWN their tensors: a row whose
        tensor is gone -- e.g. swept to ``.orphaned`` while the row was
        quarantined out-of-universe in an earlier resume, then the engine
        re-added (Sol round-2 F2's reproduced detonation) -- is quarantined
        so its key vacates and the row is recomputed from scratch.
    expected_revision : Callable[[str], str | None] | None, default=None
        Current run-revision marker per engine (``None`` result skips the
        rule for that row, e.g. engines outside the current field).
    accept_revision_drift : bool, default=False
        Keep-and-disclose git-SHA-only revision drift instead of
        quarantining (the explicit harness-only-hotfix escape hatch).
    retry_memkills : bool, default=False
        Vacate environmental memkill rows for re-run (capped per row).

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]
        ``(kept, quarantined, counts, revision_drift_rows)``; quarantined
        rows carry a ``quarantine_reason`` field, counts key on
        ``signature``/``universe``/``native_seed``/``tensor_missing``/
        ``revision``/``memkill_retry``; drift rows are the kept-but-disclosed
        rows under ``accept_revision_drift`` (annotated ``revision_drift``).
    """
    kept: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    revision_drift_rows: List[Dict[str, Any]] = []
    counts = {
        "signature": 0,
        "universe": 0,
        "native_seed": 0,
        "tensor_missing": 0,
        "revision": 0,
        "memkill_retry": 0,
    }
    for row in rows:
        reason: Optional[str] = None
        drift_accepted = False
        expected_marker = (
            expected_revision(str(row.get("engine"))) if expected_revision is not None else None
        )
        row_marker = row.get("run_revision")
        if str(row.get("record_key")) not in valid_keys:
            reason = "outside current subset/field/seed battery"
            counts["universe"] += 1
        elif expected_marker is not None and row_marker != expected_marker:
            # Implementation revision drift (R2 B4-Sol-2). The source
            # component (after the last ':') discriminates harness-only
            # commits from engine-implementation changes.
            row_component = (
                str(row_marker).rsplit(":", 1)[-1] if isinstance(row_marker, str) else None
            )
            expected_component = expected_marker.rsplit(":", 1)[-1]
            if accept_revision_drift and row_component == expected_component:
                drift_accepted = True
            else:
                reason = "implementation revision drift"
                counts["revision"] += 1
        if reason is None:
            if row.get("v3_tiered") is not None and row.get("scoring_signature") != signature:
                reason = "stale scoring signature"
                counts["signature"] += 1
            elif row.get("engine") == "dagua" and row.get("native_child_seed") != current_seed:
                reason = "native row generated under a different --seed"
                counts["native_seed"] += 1
            elif (
                retry_memkills
                and str(row.get("status_detail", "")).startswith("memkill")
                and int(row.get("memkill_retries") or 0) < MEMKILL_MAX_RETRIES
            ):
                reason = "environmental memkill retried on resume"
                counts["memkill_retry"] += 1
            elif (
                tensor_exists is not None
                and row.get("status") == "OK"
                and (not row.get("positions_path") or not tensor_exists(str(row["positions_path"])))
            ):
                # A kept OK row must OWN a present tensor; a null/absent path
                # is as unpublishable as a swept one (Sol R3 F2).
                reason = "positions tensor missing from the row store"
                counts["tensor_missing"] += 1
        if reason is None:
            if drift_accepted:
                row = {**row, "revision_drift": True}
                revision_drift_rows.append(row)
            kept.append(row)
        else:
            quarantined.append({**row, "quarantine_reason": reason})
    return kept, quarantined, counts, revision_drift_rows


def quarantine_orphan_tensors(staging: Path, rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Move position tensors with no owning OK row into ``positions/.orphaned/``.

    Second line of defense for dry-well B4-F2: score-failure rows drop their
    tensors at flip time, but a crash between tensor-save and row-append (or
    a resumed run whose re-attempt recorded SKIP) can still leave a tensor
    that ``validate_store`` would flag as an orphan AFTER publish. Sweep such
    tensors into a dot-prefixed quarantine dir (ignored by the validator)
    right before publishing, preserving them for inspection.

    Parameters
    ----------
    staging : Path
        Staging directory about to be published.
    rows : Sequence[Dict[str, Any]]
        Final (deduped) rows.

    Returns
    -------
    List[str]
        One warning line per quarantined tensor.
    """
    position_dir = staging / "positions"
    if not position_dir.is_dir():
        return []
    ok_paths = {
        str(row.get("positions_path"))
        for row in rows
        if row.get("status") == "OK" and row.get("positions_path")
    }
    warnings: List[str] = []
    orphan_dir = position_dir / ".orphaned"
    for path in sorted(position_dir.glob("*.pt")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relpath = str(Path("positions") / path.name)
        if relpath not in ok_paths:
            orphan_dir.mkdir(exist_ok=True)
            path.replace(orphan_dir / path.name)
            warnings.append(
                f"quarantined orphan tensor without an OK row: positions/.orphaned/{path.name}"
            )
    return warnings


def validate_store(directory: Path, payload: Dict[str, Any]) -> None:
    """Validate OK-row/position one-to-one mapping, ignoring dot temps.

    Parameters
    ----------
    directory : Path
        Published run directory.
    payload : Dict[str, Any]
        Published results payload.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an OK row is missing its tensor or a non-hidden tensor is
        orphaned.
    """
    ok_paths = {
        str(row.get("positions_path"))
        for row in payload.get("rows", [])
        if row.get("status") == "OK" and row.get("positions_path")
    }
    missing = sorted(path for path in ok_paths if not (directory / path).is_file())
    position_dir = directory / "positions"
    on_disk = (
        {
            str(path.relative_to(directory))
            for path in position_dir.glob("*.pt")
            if path.is_file() and not path.name.startswith(".")
        }
        if position_dir.is_dir()
        else set()
    )
    orphaned = sorted(on_disk - ok_paths)
    if missing or orphaned:
        details = []
        if missing:
            details.append(f"missing position files for OK rows: {missing}")
        if orphaned:
            details.append(f"orphaned position files without OK rows: {orphaned}")
        raise RuntimeError("; ".join(details))


# ---------------------------------------------------------------------------
# Row execution (spawn children, RSS watchdog, system floor)
# ---------------------------------------------------------------------------


def _write_child_message(result_path: str, message: Dict[str, Any]) -> None:
    """Write the child's result handshake JSON with fsync.

    Parameters
    ----------
    result_path : str
        Handshake JSON path.
    message : Dict[str, Any]
        Serializable child status payload.

    Returns
    -------
    None
    """
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(message, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_child_message(result_path: Path) -> Optional[Dict[str, Any]]:
    """Load the child's handshake JSON, tolerating torn/empty files.

    A kernel OOM kill or hard crash mid-write leaves a file that exists but
    does not parse; that must become an ERROR row, never an uncaught
    exception that silently kills a worker thread (dry-well B4-F4).

    Parameters
    ----------
    result_path : Path
        Handshake JSON path.

    Returns
    -------
    Dict[str, Any] | None
        Parsed message, or ``None`` when missing/unparseable.
    """
    try:
        with result_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _row_layout_worker(
    graph: Any,
    engine_name: str,
    seed: Optional[int],
    timeout_s: float,
    is_native: bool,
    native_deterministic: bool,
    temp_position_path: str,
    result_path: str,
) -> None:
    """Run one layout row inside an isolated spawn child.

    Native rows apply :func:`deterministic_native_runtime` around the adapter
    call and pass ``deterministic_native=True`` at the call seam (WP03-F02).
    Positions are saved uncast (``detach().cpu()``, cert-store parity)
    through an open file handle so deterministic values produce
    byte-identical files.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    engine_name : str
        Registered competitor name.
    seed : int | None
        Row seed forwarded to the adapter.
    timeout_s : float
        Adapter timeout.
    is_native : bool
        Whether this is the native ("dagua") row.
    native_deterministic : bool
        Whether the deterministic envelope is active (native rows only).
    temp_position_path : str
        Hidden temp tensor path.
    result_path : str
        Handshake JSON path.

    Returns
    -------
    None
    """
    # Own session/process group FIRST: field adapters launch their own
    # binaries (dot, java, node, ogdf_runner, ...), and the parent's RSS/
    # timeout watchdog must be able to kill the ENTIRE tree it measures via
    # killpg, not just this wrapper (Sol WP-25 review, HIGH-1).
    try:
        os.setsid()
    except OSError:
        pass
    try:
        from dagua.eval.competitors import get_competitor

        competitor = get_competitor(engine_name)
        if competitor is None:
            _write_child_message(
                result_path,
                {"status": "ERROR", "runtime_s": 0.0, "error": "adapter not registered"},
            )
            os._exit(0)
        if is_native:
            from scripts.run_benchmark import deterministic_native_runtime

            kwargs = _native_layout_kwargs(timeout_s, seed, native_deterministic)
            with deterministic_native_runtime(native_deterministic, seed):
                result = competitor.layout(graph, **kwargs)
        else:
            result = competitor.layout(graph, timeout=timeout_s, seed=seed)
        if result.pos is None:
            _write_child_message(
                result_path,
                {
                    "status": "ERROR",
                    "runtime_s": float(result.runtime_seconds),
                    "error": result.error or "adapter returned no positions",
                },
            )
            os._exit(0)
        with open(temp_position_path, "wb") as handle:
            torch.save(result.pos.detach().cpu(), handle)
        _write_child_message(
            result_path,
            {
                "status": "OK",
                "runtime_s": float(result.runtime_seconds),
                "error": result.error,
                "temp_positions_path": temp_position_path,
            },
        )
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001
        try:
            _write_child_message(
                result_path,
                {
                    "status": "ERROR",
                    "runtime_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=20),
                },
            )
        except BaseException:  # noqa: BLE001
            pass
        os._exit(1)


def _kill_process_tree(process: mp.process.BaseProcess) -> None:
    """Kill a row child AND every descendant it spawned (Sol HIGH-1).

    The RSS watchdog measures the wrapper plus its recursive descendants
    (:func:`_child_tree_rss_bytes`); containment is only real if the same
    tree dies on memkill/timeout. Two mechanisms, layered:

    1. The row child calls ``os.setsid()`` at startup, so its pgid equals its
       pid and ``os.killpg`` reaches every descendant that did not change its
       own process group (the common case for adapter-spawned binaries).
    2. A psutil descendant snapshot taken BEFORE the group kill is reaped
       individually afterwards, covering descendants that moved themselves to
       another group (psutil guards pid reuse via creation-time identity).

    No-op for a child that already exited cleanly.

    Parameters
    ----------
    process : multiprocessing process
        Row child wrapper.

    Returns
    -------
    None
    """
    import signal

    import psutil

    if process.pid is None or not process.is_alive():
        return
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    process.join(5.0)
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        process.join()
    for descendant in descendants:
        try:
            descendant.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _child_tree_rss_bytes(pid: int) -> Optional[int]:
    """Return RSS of a child process plus all of its descendants.

    Parameters
    ----------
    pid : int
        Child process id.

    Returns
    -------
    int | None
        Total RSS bytes, or ``None`` when the process is gone.
    """
    import psutil

    try:
        process = psutil.Process(pid)
        total = int(process.memory_info().rss)
        for descendant in process.children(recursive=True):
            try:
                total += int(descendant.memory_info().rss)
            except psutil.NoSuchProcess:
                continue
        return total
    except psutil.NoSuchProcess:
        return None


class RowExecutor:
    """Executes layout rows in spawn children with RSS watchdog and grace."""

    def __init__(
        self,
        args: argparse.Namespace,
        staging: Path,
        revision_markers: Optional[Dict[str, str]] = None,
        memkill_retry_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self.args = args
        self.staging = staging
        self.child_rss_abort_bytes = float(args.child_rss_abort_gb) * 1024**3
        # Per-engine run-revision markers stamped onto every row at creation
        # (R2 B4-Sol-2) and per-key environmental-memkill retry counters
        # carried across resumes (R2 B4-F4).
        self.revision_markers = revision_markers or {}
        self.memkill_retry_counts = memkill_retry_counts or {}
        self._selftest_lock = threading.Lock()
        self._selftest_armed = bool(args.rss_guard_selftest)

    def _consume_selftest(self) -> bool:
        """Consume the one-shot RSS-guard selftest arm.

        Returns
        -------
        bool
            ``True`` exactly once when ``--rss-guard-selftest`` was passed.
        """
        with self._selftest_lock:
            armed = self._selftest_armed
            self._selftest_armed = False
            return armed

    def run_row(
        self,
        entry: GraphEntry,
        engine_name: str,
        seed: Optional[int],
        child_seed: Optional[int],
        timeout_s: float,
        is_native: bool,
    ) -> Dict[str, Any]:
        """Run one layout row in an isolated spawn child (WP10-F26/F27).

        Parameters
        ----------
        entry : GraphEntry
            Loaded corpus graph.
        engine_name : str
            Registered competitor name.
        seed : int | None
            Row-key seed (``None`` for deterministic rows).
        child_seed : int | None
            Seed actually passed to the adapter (native rows use the run
            seed under a deterministic ``None`` row key, plan 7.4).
        timeout_s : float
            Adapter timeout; the join deadline adds a 60s grace.
        is_native : bool
            Whether this is the native row.

        Returns
        -------
        Dict[str, Any]
            Completed row.
        """
        base = self._base_row(entry, engine_name, seed)
        if is_native:
            # Sol B5-1: the native row KEY stays seedless (pre-registered
            # plan-7.4 convention), but the child actually runs under the
            # run seed -- record it so resume can quarantine native rows
            # generated under a different --seed instead of silently mixing.
            base["native_child_seed"] = child_seed
        seed_part = "deterministic" if seed is None else f"seed{seed}"
        stem = f".{safe_component(entry.name)}__{safe_component(engine_name)}__{seed_part}"
        temp_path = self.staging / "positions" / f"{stem}.child.pt"
        result_path = self.staging / "positions" / f"{stem}.json"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (temp_path, result_path):
            if path.exists():
                path.unlink()

        context = mp.get_context("spawn")
        process = context.Process(
            target=_row_layout_worker,
            args=(
                entry.loaded.graph,
                engine_name,
                child_seed,
                timeout_s,
                is_native,
                bool(self.args.native_deterministic),
                str(temp_path),
                str(result_path),
            ),
        )
        selftest = self._consume_selftest()
        deadline = timeout_s + CHILD_JOIN_GRACE_SECONDS
        start = time.perf_counter()
        peak_rss = 0
        try:
            process.start()
            while True:
                process.join(CHILD_POLL_SECONDS)
                elapsed = time.perf_counter() - start
                if not process.is_alive():
                    break
                rss = _child_tree_rss_bytes(process.pid) if process.pid else None
                if selftest:
                    rss = int(self.child_rss_abort_bytes) + 1
                    selftest = False
                    print(
                        f"RSS-GUARD-SELFTEST: faking over-ceiling poll for "
                        f"{entry.name} {engine_name}",
                        file=sys.stderr,
                    )
                if rss is not None:
                    peak_rss = max(peak_rss, rss)
                    if rss > self.child_rss_abort_bytes:
                        self._kill(process)
                        self._cleanup(temp_path, result_path)
                        return {
                            **base,
                            "status": "ERROR",
                            "runtime_s": elapsed,
                            "positions_path": None,
                            "error": (
                                f"child RSS {rss / 1024**3:.1f}GB exceeded ceiling "
                                f"{self.args.child_rss_abort_gb:.0f}GB"
                            ),
                            "status_detail": "memkill",
                            "peak_child_rss_gb": round(peak_rss / 1024**3, 2),
                        }
                # Aggregate-memory backstop (dry-well B4-F5): ten children
                # under their per-child ceilings can jointly exhaust the box;
                # re-check the system floor DURING flight, not just at
                # dispatch, so the runner drains before the kernel OOM killer
                # races the per-child watchdogs.
                available = system_available_bytes()
                if available is not None and available < self.args.min_avail_gb * 1024**3:
                    self._kill(process)
                    self._cleanup(temp_path, result_path)
                    return {
                        **base,
                        "status": "ERROR",
                        "runtime_s": elapsed,
                        "positions_path": None,
                        "error": (
                            f"system available memory {available / 1024**3:.1f}GB fell below "
                            f"the {self.args.min_avail_gb:.0f}GB floor while the row ran"
                        ),
                        "status_detail": "memkill:system-floor",
                        "peak_child_rss_gb": round(peak_rss / 1024**3, 2),
                    }
                if elapsed > deadline:
                    self._kill(process)
                    self._cleanup(temp_path, result_path)
                    return {
                        **base,
                        "status": "ERROR",
                        "runtime_s": elapsed,
                        "positions_path": None,
                        "error": "timeout",
                        "status_detail": "timeout",
                    }
            elapsed = time.perf_counter() - start
            exitcode = process.exitcode
            if not result_path.is_file():
                self._cleanup(temp_path, result_path)
                detail = f"child exited {exitcode}" if exitcode is not None else "child exited"
                return {
                    **base,
                    "status": "ERROR",
                    "runtime_s": elapsed,
                    "positions_path": None,
                    "error": detail,
                    "status_detail": detail,
                }
            message = _load_child_message(result_path)
            if message is None:
                # Torn/empty handshake file: the child (or the box) died
                # mid-write -- e.g. a kernel OOM kill outside the runner's
                # own watchdog (dry-well B4-F4). Never let this crash the
                # worker thread; it is an ERROR row like any other crash.
                self._cleanup(temp_path, result_path)
                return {
                    **base,
                    "status": "ERROR",
                    "runtime_s": elapsed,
                    "positions_path": None,
                    "error": f"child result file unreadable (child exited {exitcode})",
                    "status_detail": "harness:child-result-torn",
                }
            result_path.unlink()
            if message.get("status") != "OK":
                self._cleanup(temp_path, result_path)
                error = str(message.get("error") or "child error")
                detail = "timeout" if error.lower() == "timeout" else "adapter error"
                return {
                    **base,
                    "status": "ERROR",
                    "runtime_s": float(message.get("runtime_s") or elapsed),
                    "positions_path": None,
                    "error": error,
                    "status_detail": detail,
                }
            saved = Path(str(message.get("temp_positions_path")))
            if not saved.is_file():
                return {
                    **base,
                    "status": "ERROR",
                    "runtime_s": elapsed,
                    "positions_path": None,
                    "error": "child returned no positions file",
                    "status_detail": "child exited 0",
                }
            relpath = position_relpath(entry.name, engine_name, seed)
            final_path = self.staging / relpath
            final_path.parent.mkdir(parents=True, exist_ok=True)
            saved.replace(final_path)
            row = {
                **base,
                "status": "OK",
                "runtime_s": float(message.get("runtime_s") or elapsed),
                "positions_path": relpath,
                "error": message.get("error"),
            }
            if peak_rss:
                row["peak_child_rss_gb"] = round(peak_rss / 1024**3, 2)
            return row
        finally:
            self._kill(process)
            try:
                process.close()
            except ValueError:
                pass

    def _base_row(self, entry: GraphEntry, engine_name: str, seed: Optional[int]) -> Dict[str, Any]:
        """Build the common row fields for one (graph, engine, seed).

        Parameters
        ----------
        entry : GraphEntry
            Loaded corpus graph.
        engine_name : str
            Engine name.
        seed : int | None
            Row-key seed.

        Returns
        -------
        Dict[str, Any]
            Base row fields.
        """
        from scripts.run_benchmark import build_record_key

        record_key = build_record_key(entry.name, engine_name, seed)
        row = {
            "graph": entry.name,
            "corpus": entry.corpus,
            "engine": engine_name,
            "seed": seed,
            "record_key": record_key,
            "run_revision": self.revision_markers.get(engine_name),
            "nodes": entry.loaded.graph.num_nodes,
            "edges": int(entry.loaded.graph.edge_index.shape[1]),
            "directed": entry.directed,
            "directed_source": entry.directed_source,
            "source_path": str(entry.loaded.source_path),
            "loader_telemetry": entry.telemetry,
        }
        retries = self.memkill_retry_counts.get(record_key)
        if retries is not None:
            row["memkill_retries"] = retries
        return row

    def skip_row(
        self, entry: GraphEntry, engine_name: str, seed: Optional[int], reason: str
    ) -> Dict[str, Any]:
        """Build a machine-readable SKIP row.

        Parameters
        ----------
        entry : GraphEntry
            Loaded corpus graph.
        engine_name : str
            Engine name.
        seed : int | None
            Row-key seed.
        reason : str
            Machine-readable skip reason (e.g. ``unavailable:...``).

        Returns
        -------
        Dict[str, Any]
            SKIP row.
        """
        return {
            **self._base_row(entry, engine_name, seed),
            "status": "SKIP",
            "runtime_s": 0.0,
            "positions_path": None,
            "reason": reason,
        }

    @staticmethod
    def _kill(process: mp.process.BaseProcess) -> None:
        """Kill the whole row-process tree (see :func:`_kill_process_tree`).

        Parameters
        ----------
        process : multiprocessing process
            Child to stop.

        Returns
        -------
        None
        """
        _kill_process_tree(process)

    @staticmethod
    def _cleanup(*paths: Path) -> None:
        """Remove temp files if present.

        Parameters
        ----------
        *paths : Path
            Candidate temp paths.

        Returns
        -------
        None
        """
        for path in paths:
            if path.exists():
                path.unlink()


def system_available_bytes() -> Optional[int]:
    """Return system available memory in bytes (module attr for tests).

    Returns
    -------
    int | None
        ``psutil.virtual_memory().available``, or ``None`` if unavailable.
    """
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


# ---------------------------------------------------------------------------
# Scoring (V3 via imported conventions; spawn pool)
# ---------------------------------------------------------------------------


def _score_pool_initializer(graph_map: Dict[str, Any], signature: str) -> None:
    """Initialize a score-pool worker with the shared graph map + signature.

    Mirrors ``native_sprint_score.init_worker`` exactly by calling it.

    Parameters
    ----------
    graph_map : Dict[str, TestGraph]
        Corpus graphs keyed by name.
    signature : str
        Scoring signature.

    Returns
    -------
    None
    """
    import scripts.native_sprint_score as nss

    nss.init_worker(graph_map, signature)


def _score_pool_task(
    task: Tuple[str, str, str, str],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Score one saved position tensor under the V3 ruler.

    The wrapper passes ``ruler="v3"`` explicitly. NEVER swap this for
    ``native_sprint_score.score_group``: it omits the ruler argument and
    silently scores v2 (WP10-F15).

    Parameters
    ----------
    task : Tuple[str, str, str, str]
        ``(graph_name, engine, absolute_position_path, record_key)``.

    Returns
    -------
    Tuple[str, Dict[str, Any] | None, str | None]
        ``(record_key, score_row, error)``.
    """
    graph_name, engine, path_string, record_key = task
    import scripts.native_sprint_score as nss

    test_graph = nss._WORKER_GRAPHS[graph_name]
    try:
        score = nss.score_position(
            test_graph, path_string, engine, nss._WORKER_SIGNATURE, ruler="v3"
        )
        return record_key, score, None
    except Exception as exc:  # noqa: BLE001
        return record_key, None, f"{type(exc).__name__}: {exc}"


def build_test_graph_map(entries: Sequence[GraphEntry]) -> Dict[str, Any]:
    """Wrap loaded graphs as ``TestGraph`` with tally-critical tags (WP10-F16).

    The ``"undirected"`` tag is what ``is_semantically_directed`` reads; it is
    kept in lockstep with ``graph.is_semantically_directed`` (set by
    ``build_graph``). ``node_sizes`` is guaranteed by ``build_graph``'s
    ``compute_node_sizes`` call.

    Parameters
    ----------
    entries : Sequence[GraphEntry]
        Loaded corpus graphs.

    Returns
    -------
    Dict[str, TestGraph]
        Graph map for the score pool.
    """
    from dagua.eval.graphs import TestGraph

    graph_map: Dict[str, Any] = {}
    for entry in entries:
        tags = {entry.corpus} | ({"undirected"} if not entry.directed else set())
        graph_map[entry.name] = TestGraph(
            name=entry.name,
            graph=entry.loaded.graph,
            tags=tags,
            source="stdcorpora",
            description=str(entry.loaded.source_path),
        )
    return graph_map


# ---------------------------------------------------------------------------
# Tally + report
# ---------------------------------------------------------------------------


def compute_tally(
    entries: Sequence[GraphEntry],
    rows: Sequence[Dict[str, Any]],
    staging: Path,
) -> Dict[str, Any]:
    """Compute the holdout tally with imported champion-selection machinery.

    Parameters
    ----------
    entries : Sequence[GraphEntry]
        Tally-eligible graphs.
    rows : Sequence[Dict[str, Any]]
        Deduped final rows.
    staging : Path
        Run directory (for secondary-diagnostic position loads).

    Returns
    -------
    Dict[str, Any]
        Tally block: overall counts, per-corpus, per-engine field-best
        counts, and per-graph detail (with SECONDARY composite_auto
        diagnostics for champion rows).
    """
    import scripts.native_sprint_score as nss

    # Stable input order: field rows complete on concurrent worker threads,
    # and the imported _selection_key has no tie-break after equal V3 score,
    # so the FIRST exact-tied row wins. Sorting by record_key makes report
    # fields (winning engine, per-engine field-best counts) independent of
    # completion order (Sol WP-25 review, MEDIUM-1).
    scored = sorted(
        (row for row in rows if row.get("status") == "OK" and row.get("v3_tiered") is not None),
        key=lambda row: str(row.get("record_key")),
    )
    native_best = nss.best_rows_by_graph(scored, "v3_tiered", engine="dagua")
    field_best = nss.best_rows_by_graph(scored, "v3_tiered", engine=None)

    entry_by_name = {entry.name: entry for entry in entries}
    overall = {"strictly_best": 0, "tied": 0, "behind": 0, "missing": 0}
    per_corpus: Dict[str, Dict[str, int]] = {}
    per_engine_field_best: Dict[str, int] = {}
    details: List[Dict[str, Any]] = []
    for entry in entries:
        native_row = native_best.get(entry.name)
        field_row = field_best.get(entry.name)
        detail: Dict[str, Any] = {
            "graph": entry.name,
            "corpus": entry.corpus,
            "directed": entry.directed,
            "native_v3_tiered": None if native_row is None else float(native_row["v3_tiered"]),
            "field_v3_tiered": None if field_row is None else float(field_row["v3_tiered"]),
            "field_engine": None if field_row is None else str(field_row["engine"]),
            "field_seed": None if field_row is None else field_row.get("seed"),
        }
        corpus_counts = per_corpus.setdefault(
            entry.corpus, {"strictly_best": 0, "tied": 0, "behind": 0, "missing": 0}
        )
        if native_row is None or field_row is None:
            detail["status"] = "missing"
            detail["missing_side"] = (
                "both"
                if native_row is None and field_row is None
                else ("native" if native_row is None else "field")
            )
            overall["missing"] += 1
            corpus_counts["missing"] += 1
        else:
            delta = float(native_row["v3_tiered"]) - float(field_row["v3_tiered"])
            status = nss.classify(delta)
            detail["delta"] = delta
            detail["status"] = status
            overall[status] += 1
            corpus_counts[status] += 1
            engine = str(field_row["engine"])
            per_engine_field_best[engine] = per_engine_field_best.get(engine, 0) + 1
            _attach_secondary_diagnostics(
                detail, entry_by_name[entry.name], native_row, field_row, staging
            )
        details.append(detail)

    total = len(entries)
    overall_block = {
        **overall,
        "total": total,
        "best_or_tied": overall["strictly_best"] + overall["tied"],
        "tie_band": float(nss.TIE_BAND),
    }
    return {
        "overall": overall_block,
        "per_corpus": per_corpus,
        "per_engine_field_best": dict(sorted(per_engine_field_best.items())),
        "details": details,
    }


def _attach_secondary_diagnostics(
    detail: Dict[str, Any],
    entry: GraphEntry,
    native_row: Dict[str, Any],
    field_row: Dict[str, Any],
    staging: Path,
) -> None:
    """Attach SECONDARY composite_auto diagnostics for champion rows (A2).

    Computed only for the per-graph native-best and field-best rows to bound
    cost; failures are recorded as ``None`` and never affect the tally.

    Parameters
    ----------
    detail : Dict[str, Any]
        Per-graph tally detail to mutate.
    entry : GraphEntry
        Loaded graph.
    native_row : Dict[str, Any]
        Native champion row.
    field_row : Dict[str, Any]
        Field champion row.
    staging : Path
        Run directory.

    Returns
    -------
    None
    """
    from dagua.metrics import composite_auto, evaluate

    for label, row in (("native", native_row), ("field", field_row)):
        key = f"secondary_composite_auto_{label}"
        try:
            positions = torch.load(
                str(staging / str(row["positions_path"])), map_location="cpu", weights_only=True
            )
            metrics = evaluate(entry.loaded.graph, positions, tier="full")
            detail[key] = float(composite_auto(json_clean(metrics), entry.directed))
        except Exception as exc:  # noqa: BLE001
            detail[key] = None
            detail[f"{key}_error"] = f"{type(exc).__name__}: {exc}"


def _facet_deficits(
    native_row: Dict[str, Any], field_row: Dict[str, Any], top_n: int = 3
) -> List[str]:
    """Rank native's V3 facet deficits vs the winning field row.

    Parameters
    ----------
    native_row : Dict[str, Any]
        Full native champion row (with ``v3_facets``).
    field_row : Dict[str, Any]
        Full field champion row.
    top_n : int, default=3
        Number of deficits to report.

    Returns
    -------
    List[str]
        ``"<code>:<delta>"`` strings, most negative first.
    """
    native_facets = native_row.get("v3_facets") or {}
    field_facets = field_row.get("v3_facets") or {}
    deltas: List[Tuple[float, str]] = []
    for code, native_facet in native_facets.items():
        field_facet = field_facets.get(code)
        if not isinstance(native_facet, dict) or not isinstance(field_facet, dict):
            continue
        if not (native_facet.get("applicable") and field_facet.get("applicable")):
            continue
        native_score = native_facet.get("score")
        field_score = field_facet.get("score")
        if native_score is None or field_score is None:
            continue
        deltas.append((float(native_score) - float(field_score), str(code)))
    deltas.sort()
    return [f"{code}:{delta:+.2f}" for delta, code in deltas[:top_n]]


def write_report(
    directory: Path,
    payload: Dict[str, Any],
    full_rows: Dict[str, Dict[str, Any]],
    tally: Dict[str, Any],
    report_context: Dict[str, Any],
    assumptions: Sequence[str],
) -> None:
    """Write GLADOS_RUN_REPORT.md per plan 7.7's field list.

    Parameters
    ----------
    directory : Path
        Run directory to write into.
    payload : Dict[str, Any]
        Results payload (header + slim rows).
    full_rows : Dict[str, Dict[str, Any]]
        Full rows (with facets) keyed by record key.
    tally : Dict[str, Any]
        Tally block from :func:`compute_tally`.
    report_context : Dict[str, Any]
        Orchestrator-provided context (item-4 number, scale-fix note, ruler
        ledger path).
    assumptions : Sequence[str]
        Runner-recorded assumption log entries.

    Returns
    -------
    None
    """
    lines: List[str] = ["# GLaDOS Holdout Run Report", ""]

    # (1) Provenance block.
    subset_info = payload.get("subset") or {}
    disk_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    lines += [
        "## 1. Provenance",
        "",
        f"- Git SHA: `{payload['git_sha']}` (dirty: {payload['git_dirty']})",
        f"- Module-path preflight: `{payload['dagua_module_path']}`",
        f"- Seed: {payload['seed']}; seed battery (stochastic engines): {payload['seeds']}",
        f"- Native deterministic: {payload['native_deterministic']}; "
        f"native timeout {payload['native_timeout']}s; engine timeout {payload['engine_timeout']}s",
        f"- Workers: {payload['workers']}; score workers: {payload['score_workers']}",
        f"- Corpus dir: `{payload['corpus_dir']}`; subset: `{payload.get('subset_path')}` "
        f"(sha256 {payload.get('subset_sha256')})",
        f"- Subset rule: {subset_info.get('rule', 'n/a')} "
        f"(seed string `{subset_info.get('seed_string', 'n/a')}`, "
        f"fraction {subset_info.get('fraction', 'n/a')})",
        f"- Scoring signature: `{payload['scoring_signature']}`",
        f"- Output dir disk usage: {disk_bytes / 1024**2:.1f} MB",
        "",
    ]
    selected = subset_info.get("selected") or []
    if selected:
        lines.append("Selected IDs per corpus:")
        by_corpus: Dict[str, List[str]] = {}
        for item in selected:
            by_corpus.setdefault(item["corpus"], []).append(item["filename"])
        for corpus in sorted(by_corpus):
            names = ", ".join(sorted(by_corpus[corpus]))
            lines.append(f"- {corpus} ({len(by_corpus[corpus])}): {names}")
        lines.append("")
    lines.append("Engine pool (availability):")
    for engine, meta in sorted(payload["engine_availability"].items()):
        status = "available" if meta["available"] else f"UNAVAILABLE ({meta['reason']})"
        lines.append(f"- {engine}: {status}")
    lines.append("")
    if payload.get("engine_exclusions"):
        lines.append("Excluded engine families (built-in, with reasons):")
        for engine, reason in sorted(payload["engine_exclusions"].items()):
            lines.append(f"- {engine}: {reason}")
        lines.append("")

    # (2) Overall tally with the clean-sweep rule.
    overall = tally["overall"]
    lines += [
        "## 2. Overall holdout tally",
        "",
        f"- strictly_best: {overall['strictly_best']}",
        f"- tied: {overall['tied']}",
        f"- behind: {overall['behind']}",
        f"- missing: {overall['missing']}",
        f"- total: {overall['total']} (best_or_tied {overall['best_or_tied']}; "
        f"tie band {overall['tie_band']})",
        "",
    ]
    if overall["total"] > 0 and overall["behind"] == 0 and overall["tied"] == 0:
        lines += [
            "> **SUSPICIOUS: clean sweep on a thin holdout.** Zero behind AND zero",
            "> tied rows is a red flag for measurement error, a starved field, or",
            "> selection leakage -- audit before celebrating.",
            "",
        ]

    # (3) Per-corpus breakdown.
    lines += [
        "## 3. Per-corpus breakdown",
        "",
        "| corpus | strict | tied | behind | missing |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, counts in sorted(tally["per_corpus"].items()):
        lines.append(
            f"| {corpus} | {counts['strictly_best']} | {counts['tied']} | "
            f"{counts['behind']} | {counts['missing']} |"
        )
    lines.append("")

    # (4) Per-engine field-best counts.
    lines += ["## 4. Per-engine field-best counts", ""]
    if tally["per_engine_field_best"]:
        for engine, count in sorted(
            tally["per_engine_field_best"].items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"- {engine}: {count}")
    else:
        lines.append("- (no field-best rows)")
    lines.append("")

    # (5) Every native loss, honestly characterized.
    losses = [detail for detail in tally["details"] if detail.get("status") == "behind"]
    lines += ["## 5. Native losses", ""]
    if losses:
        for detail in sorted(losses, key=lambda item: item.get("delta") or 0.0):
            graph = detail["graph"]
            native_key_row = _find_champion_row(full_rows, graph, "dagua")
            field_key_row = _find_champion_row(
                full_rows, graph, detail.get("field_engine"), detail.get("field_seed")
            )
            deficits = (
                _facet_deficits(native_key_row, field_key_row)
                if native_key_row and field_key_row
                else []
            )
            flags = (native_key_row or {}).get("v3_row_flags") or []
            characterization = (
                f"top facet deficits vs winner: {', '.join(deficits)}"
                if deficits
                else "facet-level deficits unavailable"
            )
            if flags:
                characterization += f"; native row flags: {', '.join(map(str, flags))}"
            lines.append(
                f"- {graph}: behind by {abs(detail['delta']):.2f} to "
                f"{detail['field_engine']} -- {characterization}"
            )
    else:
        lines.append("- none")
    lines.append("")

    # (6) Errors / skips / timeouts / memory events.
    rows = list(full_rows.values())
    errors = [row for row in rows if row.get("status") == "ERROR"]
    skips = [row for row in rows if row.get("status") == "SKIP"]
    timeouts = [row for row in errors if row.get("status_detail") == "timeout"]
    memkills = [row for row in errors if str(row.get("status_detail", "")).startswith("memkill")]
    load_rows = payload.get("load_rows", [])
    quarantined = payload.get("quarantined_rows", [])
    quarantine_reasons: Dict[str, int] = {}
    for row in quarantined:
        reason = str(row.get("quarantine_reason"))
        quarantine_reasons[reason] = quarantine_reasons.get(reason, 0) + 1
    lines += [
        "## 6. Errors, skips, timeouts, memory events",
        "",
        f"- ERROR rows: {len(errors)} (timeouts {len(timeouts)}, memkills {len(memkills)})",
        f"- SKIP rows: {len(skips)}",
        f"- Load-phase problem rows: {len(load_rows)}",
        f"- Quarantined stale resume rows: {len(quarantined)}"
        + (
            " (" + ", ".join(f"{r}: {c}" for r, c in sorted(quarantine_reasons.items())) + ")"
            if quarantine_reasons
            else ""
        ),
        f"- Revision-drift rows KEPT under --accept-revision-drift: "
        f"{len(payload.get('revision_drift_rows', []))}",
        f"- Run aborted: {payload['aborted']} ({payload.get('aborted_reason')})",
        "",
    ]
    markers = payload.get("run_revision_markers", {})
    for row in payload.get("revision_drift_rows", []):
        expected_marker = markers.get(str(row.get("engine")))
        lines.append(
            f"- REVISION DRIFT (kept): {row.get('record_key')} -- row "
            f"{row.get('run_revision')} vs current {expected_marker}"
        )
    for row in errors:
        lines.append(
            f"- ERROR {row['record_key']}: {row.get('status_detail')} -- {row.get('error')}"
        )
    skip_reasons: Dict[str, int] = {}
    for row in skips:
        skip_reasons[str(row.get("reason"))] = skip_reasons.get(str(row.get("reason")), 0) + 1
    for reason, count in sorted(skip_reasons.items()):
        lines.append(f"- SKIP x{count}: {reason}")
    for row in load_rows:
        label = row.get("status")
        lines.append(f"- {label} {row.get('graph')}: {row.get('error') or row.get('reason')}")
    lines.append("")

    # (7) Ruler-bug ledger appendix.
    lines += ["## 7. Ruler-bug ledger (appendix)", ""]
    ledger_path = report_context.get("ruler_ledger_path")
    if ledger_path and Path(str(ledger_path)).is_file():
        lines.append(Path(str(ledger_path)).read_text(encoding="utf-8"))
    else:
        lines.append(
            "(ruler ledger not provided via --report-context; see findings/RULER_BUG_LEDGER.md)"
        )
    lines.append("")

    # (8) Scale-fix disclosure + adopted item-4 number.
    lines += [
        "## 8. Scale-fix disclosure and adopted item-4 number",
        "",
        f"- item4_strict: {report_context.get('item4_strict', 'NOT PROVIDED')}",
        f"- scale_fix_note: {report_context.get('scale_fix_note', 'NOT PROVIDED')}",
        "",
    ]

    # (9) Assumption log.
    lines += ["## 9. Assumption log", ""]
    if assumptions:
        lines += [f"- {item}" for item in assumptions]
    else:
        lines.append("- none recorded")
    lines.append("")

    (directory / "GLADOS_RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_champion_row(
    full_rows: Dict[str, Dict[str, Any]],
    graph: str,
    engine: Optional[str],
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Find the full row backing a champion tally entry.

    Parameters
    ----------
    full_rows : Dict[str, Dict[str, Any]]
        Full rows keyed by record key.
    graph : str
        Graph name.
    engine : str | None
        Engine name.
    seed : int | None, default=None
        Champion row seed.

    Returns
    -------
    Dict[str, Any] | None
        Matching row, best-scored when several match.
    """
    if engine is None:
        return None
    candidates = [
        row
        for row in full_rows.values()
        if row.get("graph") == graph
        and row.get("engine") == engine
        and row.get("v3_tiered") is not None
        and (seed is None or row.get("seed") == seed)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["v3_tiered"]))


# ---------------------------------------------------------------------------
# Publish + archive
# ---------------------------------------------------------------------------


def publish_results(output_dir: Path, staging: Path, payload: Dict[str, Any]) -> None:
    """Publish the staging directory: validate staging FIRST, then swap.

    Structural fix for the dry-well detonation class (Sol round-2 F3): the
    old shape swapped staging into place, deleted the prior run, and only
    THEN validated -- so any missing-tensor state crashed AFTER the last
    valid run was gone. Order now:

    1. ``validate_store`` runs against STAGING. A failure raises with the
       prior published run completely untouched and staging preserved for
       inspection/resume. Only a fully-validated staging dir ever replaces
       the published run.
    2. The swap keeps the prior run as ``.prev`` until the installed output
       re-validates; on swap or re-validation failure the prior run is
       restored (the failed candidate is kept as ``.failed-publish``).

    Parameters
    ----------
    output_dir : Path
        Final output directory.
    staging : Path
        Staging directory containing rows, positions, and the report.
    payload : Dict[str, Any]
        Final results payload.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If staging (or the installed output) fails validation; the prior
        published run is untouched or restored respectively.
    """
    with (staging / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    validate_store(staging, payload)

    previous = output_dir.with_name(f"{output_dir.name}.prev")
    if previous.exists():
        shutil.rmtree(previous)
    moved_prior = False
    if output_dir.exists():
        output_dir.rename(previous)
        moved_prior = True
    try:
        staging.rename(output_dir)
    except BaseException:
        if moved_prior and previous.exists() and not output_dir.exists():
            previous.rename(output_dir)
        raise
    try:
        validate_store(output_dir, payload)
    except BaseException:
        # Restore the prior run; keep the failed candidate for inspection.
        if moved_prior and previous.exists():
            failed = output_dir.with_name(f"{output_dir.name}.failed-publish")
            if failed.exists():
                shutil.rmtree(failed)
            output_dir.rename(failed)
            previous.rename(output_dir)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def archive_run(output_dir: Path, archive_dir: Path) -> Optional[str]:
    """Copy the published run to the archive dir with a sha256 manifest (A-S2).

    Failure to archive is a WARNING, never a run failure.

    Parameters
    ----------
    output_dir : Path
        Published run directory.
    archive_dir : Path
        Archive root; the run lands in ``<archive-dir>/glados_holdout/``.

    Returns
    -------
    str | None
        Warning message on failure, ``None`` on success.
    """
    destination = archive_dir / "glados_holdout"
    temp = archive_dir / f".glados_holdout.tmp-{os.getpid()}"
    previous = archive_dir / ".glados_holdout.prev"
    try:
        # Dry-well R1 B4-F1 (CRITICAL) + Sol round-2 F1: EVERY path this
        # function mutates -- destination, temp, previous -- must pass the
        # overlap predicate against the published run BEFORE any destructive
        # op. Sol's probes aliased output_dir to the hidden temp/.prev
        # siblings and deleted the run through them.
        resolved_output = output_dir.resolve()
        for label, candidate in (
            ("destination", destination),
            ("temp", temp),
            ("previous-archive", previous),
        ):
            resolved = candidate.resolve()
            if (
                resolved == resolved_output
                or resolved in resolved_output.parents
                or resolved_output in resolved.parents
            ):
                return (
                    f"REFUSING to archive: {label} path {resolved} overlaps the "
                    f"published run {resolved_output}; nothing was copied or deleted"
                )
        # Copy to a temp sibling, manifest it, then swap atomically-enough:
        # a failed copy can never destroy the previous archive (additive
        # preservation, A-S2).
        if temp.exists():
            shutil.rmtree(temp)
        shutil.copytree(output_dir, temp)
        manifest_lines = []
        for path in sorted(temp.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest_lines.append(f"{digest}  {path.relative_to(temp).as_posix()}")
        (temp / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        if previous.exists():
            shutil.rmtree(previous)
        moved_previous = False
        if destination.exists():
            destination.rename(previous)
            moved_previous = True
        try:
            temp.rename(destination)
        except BaseException:
            # Roll the old archive back so a failed swap is not destructive.
            if moved_previous and previous.exists() and not destination.exists():
                previous.rename(destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        return None
    except Exception as exc:  # noqa: BLE001
        # Never claim the run is unaffected without checking (B4-F1).
        state = (
            "published run verified intact"
            if (output_dir / "results.json").is_file()
            else "PUBLISHED RUN MAY BE DAMAGED -- inspect immediately"
        )
        return f"archive to {archive_dir} failed ({state}): {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Work plan
# ---------------------------------------------------------------------------


def print_work_plan(
    entries: Sequence[GraphEntry],
    load_rows: Sequence[Dict[str, Any]],
    field_engines: Sequence[str],
    seeds_by_engine: Dict[str, List[Optional[int]]],
    availability: Dict[str, Dict[str, Any]],
    completed_keys: Set[str],
    args: argparse.Namespace,
    include_native: bool,
) -> None:
    """Print the mandatory pre-execution work plan (guardrails-in-tools).

    Parameters
    ----------
    entries : Sequence[GraphEntry]
        Loaded graphs.
    load_rows : Sequence[Dict[str, Any]]
        Load-phase problem rows.
    field_engines : Sequence[str]
        Field engine names.
    seeds_by_engine : Dict[str, List[int | None]]
        Seed battery per engine.
    availability : Dict[str, Dict[str, Any]]
        Availability metadata.
    completed_keys : Set[str]
        Row keys already completed (resume).
    args : argparse.Namespace
        Parsed options.
    include_native : bool
        Whether native rows are part of this run.

    Returns
    -------
    None
    """
    corpus_counts: Dict[str, int] = {}
    directed_count = 0
    for entry in entries:
        corpus_counts[entry.corpus] = corpus_counts.get(entry.corpus, 0) + 1
        directed_count += int(entry.directed)
    native_rows = len(entries) if include_native else 0
    field_rows_per_graph = sum(len(seeds_by_engine[engine]) for engine in field_engines)
    field_rows = len(entries) * field_rows_per_graph
    total = native_rows + field_rows
    unavailable = sorted(
        engine for engine in field_engines if not availability.get(engine, {}).get("available")
    )
    native_ceiling_h = native_rows * args.native_timeout / 3600.0
    field_ceiling_h = field_rows * args.engine_timeout / (3600.0 * max(1, args.workers))
    print("== GLaDOS WORK PLAN ==")
    print(
        f"graphs: {len(entries)} "
        f"({', '.join(f'{corpus}={count}' for corpus, count in sorted(corpus_counts.items()))}; "
        f"directed={directed_count} -- directed rows cost 2-3x referee evals, CB-3)"
    )
    print(f"load problems: {len(load_rows)}")
    print(
        f"engines: native={'yes' if include_native else 'no'}, field={len(field_engines)} "
        f"(unavailable: {len(unavailable)}{': ' + ', '.join(unavailable) if unavailable else ''})"
    )
    print(
        f"rows: native={native_rows} + field={field_rows} "
        f"({field_rows_per_graph}/graph) = {total} planned; "
        f"{len(completed_keys)} already completed (resume)"
    )
    native_mode = (
        "deterministic envelope"
        if args.native_deterministic
        else "WALL-CLOCK -- not deterministic!"
    )
    print(
        f"timeouts: native {args.native_timeout:.0f}s ({native_mode}), "
        f"field {args.engine_timeout:.0f}s, workers {args.workers}"
    )
    print(
        f"runtime CEILING (not estimate): native {native_ceiling_h:.1f}h serial + "
        f"field {field_ceiling_h:.1f}h at {args.workers} workers"
    )
    print(
        f"memory: child ceiling {args.child_rss_abort_gb:.0f}GB, parent abort "
        f"{PARENT_RSS_ABORT_BYTES / 1024**3:.0f}GB, system floor {args.min_avail_gb:.0f}GB"
    )
    print("======================", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class _AbortRun(Exception):
    """Raised internally to unwind execution once a memory guard fires."""


def _parent_guard(state: Dict[str, Any]) -> None:
    """Run the parent-process RSS guard every N completed rows.

    Parameters
    ----------
    state : Dict[str, Any]
        Mutable run state with ``rows_done`` and ``rss_warned`` fields.

    Returns
    -------
    None

    Raises
    ------
    _AbortRun
        When parent RSS crosses the abort ceiling.
    """
    if state["rows_done"] % PARENT_GC_TRIM_INTERVAL_ROWS != 0:
        return
    release_native_heap()
    rss = parent_rss_bytes()
    if rss is None:
        return
    print(
        f"progress: {state['rows_done']} rows, parent RSS={rss / 1024**3:.2f}GB",
        file=sys.stderr,
    )
    if rss >= PARENT_RSS_ABORT_BYTES:
        raise _AbortRun(
            f"parent RSS {rss / 1024**3:.1f}GB >= abort ceiling "
            f"{PARENT_RSS_ABORT_BYTES / 1024**3:.0f}GB after {state['rows_done']} rows"
        )
    if rss >= PARENT_RSS_WARN_BYTES and not state["rss_warned"]:
        state["rss_warned"] = True
        print(
            f"WARNING: parent RSS {rss / 1024**3:.1f}GB >= warn threshold "
            f"{PARENT_RSS_WARN_BYTES / 1024**3:.0f}GB",
            file=sys.stderr,
        )


def _check_system_floor(min_avail_gb: float) -> None:
    """Abort dispatching when system available memory drops below the floor.

    Parameters
    ----------
    min_avail_gb : float
        Minimum available system memory in GB.

    Returns
    -------
    None

    Raises
    ------
    _AbortRun
        When available memory is below the floor.
    """
    available = system_available_bytes()
    if available is not None and available < min_avail_gb * 1024**3:
        raise _AbortRun(
            f"system available memory {available / 1024**3:.1f}GB below floor "
            f"{min_avail_gb:.0f}GB; stopped dispatching"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the GLaDOS holdout protocol.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Argument list; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` ok; ``1`` publish validation/swap failure (prior published run
        untouched or restored, candidate data preserved); ``2`` no graphs;
        ``3`` guarded abort -- memory guard or unexpected harness
        exception -- with partial results published; ``4`` preflight failure
        or fresh-run-over-existing-data refusal; ``5`` load-phase fatal.
    """
    args = parse_args(argv)
    assumptions: List[str] = []
    warnings: List[str] = []

    try:
        provenance = preflight(args)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILURE: {exc}", file=sys.stderr)
        return 4
    print(f"preflight ok: dagua at {provenance['dagua_module_path']}", flush=True)

    subset_relpaths: Optional[Set[str]] = None
    subset_info: Dict[str, Any] = {}
    subset_sha256: Optional[str] = None
    if args.subset is not None:
        subset_bytes = args.subset.read_bytes()
        subset_sha256 = hashlib.sha256(subset_bytes).hexdigest()
        subset_info = json.loads(subset_bytes.decode("utf-8"))
        subset_relpaths = {str(item["relpath"]) for item in subset_info.get("selected", [])}

    report_context: Dict[str, Any] = {}
    if args.report_context is not None and args.report_context.is_file():
        report_context = json.loads(args.report_context.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries, load_rows, load_stats = load_phase(
            args.corpus_dir, subset_relpaths, args.max_nodes, args.max_edges
        )
    except LoadPhaseFatal as exc:
        print(f"LOAD-PHASE FATAL: {exc}", file=sys.stderr)
        return 5
    if not entries and not load_rows:
        print(f"No supported graphs found under {args.corpus_dir}", file=sys.stderr)
        return 2
    if not entries:
        print(
            f"LOAD-PHASE FATAL: all {len(load_rows)} corpus files failed to load or were excluded",
            file=sys.stderr,
        )
        return 5

    for row in load_rows:
        if row.get("status") == "LOAD_SUSPECT":
            assumptions.append(
                f"LOAD_SUSPECT excluded from tally: {row['graph']} ({row['reason']})"
            )
        elif row.get("status") == "SKIP":
            assumptions.append(f"graph excluded: {row['graph']} ({row['reason']})")
    if load_stats["sealed_skipped"]:
        assumptions.append(
            f"{load_stats['sealed_skipped']} corpus files outside the blind subset "
            "were skipped sealed (never loaded)"
        )
    source_counts: Dict[str, int] = {}
    for entry in entries:
        source_counts[entry.directed_source] = source_counts.get(entry.directed_source, 0) + 1
    assumptions.append(
        "directedness policy applied per corpus (never path substrings): "
        + ", ".join(f"{source}={count}" for source, count in sorted(source_counts.items()))
    )
    assumptions.append(
        "word2vecgd and word2vecgd_reimpl are byte-identical twins (WP09-F16); "
        "both stay in the field, so the family is double-represented"
    )

    # Engine field resolution.
    if args.engines_file is not None:
        engines = [str(name) for name in json.loads(args.engines_file.read_text("utf-8"))]
        assumptions.append(f"--engines-file override active: {engines}")
    else:
        engines = ["dagua", *GLADOS_ENGINE_FIELD]
    include_native = "dagua" in engines
    field_engines = [engine for engine in engines if engine != "dagua"]
    availability = engine_availability(engines)

    from scripts.run_benchmark import build_record_key, seeds_for_engine

    seeds_by_engine: Dict[str, List[Optional[int]]] = {
        engine: seeds_for_engine(engine, args.seeds, args.seed) for engine in field_engines
    }

    import scripts.native_sprint_score as nss

    signature = nss.scoring_signature()
    # Per-engine run-revision markers (R2 B4-Sol-2): git SHA + engine source
    # component, stamped onto every row and compared on resume.
    revision_markers = compute_revision_markers(engines, provenance["git_sha"])
    row_store_snapshot: Optional[Path] = None

    # Staging setup + resume.
    staging = args.output_dir.with_name(f"{args.output_dir.name}.tmp")
    if args.resume and not staging.exists() and rows_path(args.output_dir).is_file():
        # Re-open an aborted-and-published partial run COPY-based (Sol R3 F1):
        # the canonical published run stays intact at output_dir until
        # publish_results installs a fully-validated candidate, so a failure
        # anywhere in resume preparation (quarantine rewrite included) leaves
        # the prior valid run untouched. Costs one transient copy of the
        # partial run on disk.
        try:
            shutil.copytree(args.output_dir, staging)
        except Exception as exc:  # noqa: BLE001 - canonical run must survive
            print(
                f"RESUME-PREP FAILED (copy to staging): {type(exc).__name__}: {exc}; "
                "canonical published run untouched",
                file=sys.stderr,
                flush=True,
            )
            shutil.rmtree(staging, ignore_errors=True)
            return 1
        assumptions.append(
            "resume re-opened a previously published partial run "
            "(copy-based; canonical output retained until validated republish)"
        )
    if not args.resume:
        # Dry-well B4-F6: dropping --resume after a crash used to silently
        # rmtree hours of completed rows (up to 63 x 1800s of native work).
        # Destroying existing run data now requires an explicit flag.
        staging_nonempty = staging.exists() and any(staging.iterdir())
        output_has_run = (
            rows_path(args.output_dir).is_file() or (args.output_dir / "results.json").is_file()
        )
        if (staging_nonempty or output_has_run) and not args.force_fresh:
            print(
                "REFUSING to start fresh over existing run data in "
                f"{args.output_dir} (staging populated: {staging_nonempty}, published run "
                f"present: {output_has_run}). Pass --resume to continue it, or "
                "--force-fresh to destroy it and start over.",
                file=sys.stderr,
            )
            return 4
        if staging.exists():
            shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    removed_temps = clean_child_temps(staging)
    if removed_temps:
        warnings.append(f"removed {removed_temps} stale child temp file(s) at startup")
    existing_rows: List[Dict[str, Any]] = []
    if args.resume:
        # Snapshot the ONLY copy of the row store BEFORE any resume-prep
        # mutation (torn-line repair below, quarantine rewrite later):
        # dry-well R2 B4-F1 reproduced the quarantine rewrite gutting a
        # staging-only crashed run (1002 -> 0 bytes) on a mistyped resume
        # invocation. The snapshot is removed only after a successful
        # publish.
        row_store_snapshot = snapshot_row_store(staging)
        existing_rows, torn_warnings = load_rows_tolerant(staging)
        warnings.extend(torn_warnings)

    # Resume-consistency quarantine (dry-well B4-F3, Sol B3-2/B4-3/B5-1,
    # R2 B4-Sol-2, R2 B4-F4): resumed rows must not silently compete when
    # they were produced under a different ruler, row universe, native seed,
    # or implementation revision. Quarantined rows are recorded
    # (results.json + report), excluded from the tally, and their keys
    # vacate so in-universe rows are re-run/re-scored under the CURRENT
    # signature and revision. Layout-only rows (no v3_tiered) pass through
    # and are scored fresh; a quarantined scored row whose layout sibling
    # survives is rescored from its existing tensor -- EXCEPT revision
    # drift, which invalidates the layout itself (siblings carry the same
    # stale marker and quarantine with it).
    quarantined_rows: List[Dict[str, Any]] = []
    revision_drift_rows: List[Dict[str, Any]] = []
    memkill_retry_counts: Dict[str, int] = {}
    if args.resume and existing_rows:
        valid_keys: Set[str] = set()
        for entry in entries:
            if include_native:
                valid_keys.add(build_record_key(entry.name, "dagua", None))
            for engine in field_engines:
                for seed in seeds_by_engine[engine]:
                    valid_keys.add(build_record_key(entry.name, engine, seed))
        existing_rows, quarantined_rows, quarantine_counts, revision_drift_rows = (
            partition_resumed_rows(
                existing_rows,
                signature,
                valid_keys,
                args.seed,
                tensor_exists=lambda relpath: (staging / relpath).is_file(),
                expected_revision=revision_markers.get,
                accept_revision_drift=args.accept_revision_drift,
                retry_memkills=args.retry_memkills,
            )
        )
        for row in quarantined_rows:
            if row.get("quarantine_reason") == "environmental memkill retried on resume":
                key = str(row.get("record_key"))
                memkill_retry_counts[key] = int(row.get("memkill_retries") or 0) + 1
        if quarantined_rows:
            # Quarantine is DURABLE (Sol round-2 F2): rewrite the primary
            # store without the quarantined rows so a later resume that
            # re-admits their key reruns from scratch instead of
            # resurrecting a row whose tensor was orphaned.
            try:
                rewrite_rows(staging, existing_rows)
            except Exception as exc:  # noqa: BLE001 - Sol R3 F1 / R2 B4-F1:
                # restore the pre-resume snapshot so a failed rewrite can
                # never leave a gutted staging-only store behind.
                restored = row_store_snapshot is not None and restore_row_store_snapshot(
                    row_store_snapshot, staging
                )
                print(
                    f"RESUME-PREP FAILED (quarantine rewrite): {type(exc).__name__}: "
                    f"{exc}; row store "
                    f"{'restored from pre-resume snapshot' if restored else 'left as-is'}; "
                    "canonical published run untouched",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            summary = (
                f"QUARANTINE: {len(quarantined_rows)} stale resumed row(s) removed from the "
                f"row store, recorded in quarantined_rows, and re-run where in-universe "
                f"(stale signature: {quarantine_counts['signature']}, outside row universe: "
                f"{quarantine_counts['universe']}, native seed mismatch: "
                f"{quarantine_counts['native_seed']}, tensor missing: "
                f"{quarantine_counts['tensor_missing']}, revision drift: "
                f"{quarantine_counts['revision']}, memkill retries: "
                f"{quarantine_counts['memkill_retry']})"
            )
            print(summary, flush=True)
            warnings.append(summary)
        if revision_drift_rows:
            drift_line = (
                f"REVISION DRIFT ACCEPTED: {len(revision_drift_rows)} resumed row(s) kept "
                "across a git-revision change under --accept-revision-drift; results.json "
                "and the report disclose them"
            )
            print(drift_line, flush=True)
            warnings.append(drift_line)
    row_map: Dict[str, Dict[str, Any]] = dedupe_rows(existing_rows)
    completed_keys: Set[str] = set(row_map)

    print_work_plan(
        entries,
        load_rows,
        field_engines,
        seeds_by_engine,
        availability,
        completed_keys,
        args,
        include_native,
    )

    executor = RowExecutor(args, staging, revision_markers, memkill_retry_counts)
    store_lock = threading.Lock()
    state: Dict[str, Any] = {"rows_done": 0, "rss_warned": False}
    aborted_reason: Optional[str] = None

    def record(row: Dict[str, Any]) -> None:
        with store_lock:
            append_row(staging, row)
            row_map[str(row["record_key"])] = row
            completed_keys.add(str(row["record_key"]))
            state["rows_done"] += 1
            print(f"{row['status']} {row['record_key']}", flush=True)
        _parent_guard(state)

    try:
        # Phase 1: native rows FIRST, serial (plan 7.3). Per-row containment:
        # an unexpected harness exception becomes an ERROR row, never a
        # whole-run crash (dry-well B4-F4).
        if include_native:
            for entry in entries:
                key_row = executor._base_row(entry, "dagua", None)
                if key_row["record_key"] in completed_keys:
                    continue
                try:
                    _check_system_floor(args.min_avail_gb)
                    row = executor.run_row(
                        entry,
                        "dagua",
                        seed=None,
                        child_seed=args.seed,
                        timeout_s=args.native_timeout,
                        is_native=True,
                    )
                except _AbortRun:
                    raise
                except Exception as exc:  # noqa: BLE001
                    row = {
                        **key_row,
                        "native_child_seed": args.seed,
                        "status": "ERROR",
                        "runtime_s": 0.0,
                        "positions_path": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "status_detail": f"harness:{type(exc).__name__}",
                    }
                record(row)

        # Phase 2: field rows, threaded dispatcher over spawn children.
        tasks: List[Tuple[GraphEntry, str, Optional[int]]] = []
        for entry in entries:
            for engine in field_engines:
                for seed in seeds_by_engine[engine]:
                    tasks.append((entry, engine, seed))
        task_lock = threading.Lock()
        task_iter = iter(tasks)
        abort_box: List[str] = []

        def field_worker() -> None:
            from dagua.eval.competitors import get_competitor

            while True:
                with task_lock:
                    task = next(task_iter, None)
                if task is None or abort_box:
                    return
                entry, engine, seed = task
                # Per-task containment (dry-well B4-F4): a worker thread must
                # NEVER die silently -- any unexpected exception (torn child
                # results are handled in run_row; ENOMEM at spawn, ENOSPC at
                # tensor move, ...) becomes an ERROR row and the worker moves
                # on. Only _AbortRun (memory guards) drains the run.
                try:
                    key = executor._base_row(entry, engine, seed)["record_key"]
                    if key in completed_keys:
                        continue
                    meta = availability.get(engine, {"available": False, "reason": "not checked"})
                    competitor = get_competitor(engine)
                    if competitor is None or not meta["available"]:
                        reason = str(meta.get("reason") or "unavailable")
                        record(executor.skip_row(entry, engine, seed, f"unavailable:{reason}"))
                        continue
                    if entry.loaded.graph.num_nodes > competitor.max_nodes:
                        record(
                            executor.skip_row(
                                entry, engine, seed, f"max_nodes:{competitor.max_nodes}"
                            )
                        )
                        continue
                    _check_system_floor(args.min_avail_gb)
                    row = executor.run_row(
                        entry,
                        engine,
                        seed=seed,
                        child_seed=seed,
                        timeout_s=args.engine_timeout,
                        is_native=False,
                    )
                    record(row)
                except _AbortRun as exc:
                    abort_box.append(str(exc))
                    return
                except Exception as exc:  # noqa: BLE001
                    error_row = {
                        **executor._base_row(entry, engine, seed),
                        "status": "ERROR",
                        "runtime_s": 0.0,
                        "positions_path": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "status_detail": f"harness:{type(exc).__name__}",
                    }
                    try:
                        record(error_row)
                    except _AbortRun as abort_exc:
                        abort_box.append(str(abort_exc))
                        return
                    except Exception as record_exc:  # noqa: BLE001
                        # Cannot even record (e.g. ENOSPC on the row store):
                        # drain and take the partial-publish abort path.
                        abort_box.append(
                            f"harness: row store append failed after "
                            f"{type(exc).__name__}: {record_exc}"
                        )
                        return

        threads = [
            threading.Thread(target=field_worker, name=f"glados-field-{index}")
            for index in range(max(1, min(args.workers, len(tasks) or 1)))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if abort_box:
            raise _AbortRun(abort_box[0])

        # Phase 3: scoring (V3 primary).
        graph_map = build_test_graph_map(entries)
        to_score = [
            row
            for row in row_map.values()
            if row.get("status") == "OK"
            and row.get("v3_tiered") is None
            and row.get("positions_path")
            and row.get("graph") in graph_map
        ]
        score_tasks = [
            (
                str(row["graph"]),
                str(row["engine"]),
                str(staging / str(row["positions_path"])),
                str(row["record_key"]),
            )
            for row in to_score
        ]

        def merge_score(
            record_key: str, score: Optional[Dict[str, Any]], error: Optional[str]
        ) -> None:
            row = dict(row_map[record_key])
            if score is not None:
                row.update(score)
            else:
                row["status"] = "ERROR"
                row["error"] = error
                row["status_detail"] = f"score:{error}"
                # Dry-well B4-F2 / Sol B4-1: a score-failure row must not
                # leave its tensor behind -- validate_store treats every
                # tensor without an OK row as an orphan and would crash the
                # run AFTER publish. Drop the tensor, keep the trail.
                relpath = row.get("positions_path")
                if relpath:
                    tensor_path = staging / str(relpath)
                    if tensor_path.is_file():
                        tensor_path.unlink()
                    row["positions_path"] = None
                    row["positions_path_removed"] = str(relpath)
            record(row)

        if score_tasks:
            print(f"scoring {len(score_tasks)} rows (V3, ruler=v3)", flush=True)
            if args.score_workers <= 1:
                _score_pool_initializer(graph_map, signature)
                for task in score_tasks:
                    merge_score(*_score_pool_task(task))
            else:
                context = mp.get_context("spawn")
                with context.Pool(
                    processes=args.score_workers,
                    initializer=_score_pool_initializer,
                    initargs=(graph_map, signature),
                ) as pool:
                    for record_key, score, error in pool.imap_unordered(
                        _score_pool_task, score_tasks
                    ):
                        merge_score(record_key, score, error)
    except _AbortRun as exc:
        aborted_reason = str(exc)
        print(f"ABORT: {aborted_reason}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        # Dry-well B4-F4: an unexpected exception in the serial native phase
        # or the scoring loop must still take the abort-with-partial-publish
        # path (exit 3) instead of crashing with rows stranded in staging.
        aborted_reason = f"harness:{type(exc).__name__}: {exc}"
        print(f"ABORT: {aborted_reason}", file=sys.stderr)
        traceback.print_exc(limit=20, file=sys.stderr)

    # Tally + payload + report + publish (partial on abort).
    tally_entries = entries
    final_rows = list(row_map.values())
    tally = compute_tally(tally_entries, final_rows, staging)

    warnings.extend(quarantine_orphan_tensors(staging, final_rows))
    slim_drop = {"metrics", "v3_facets", "graph_meta", "v3_applicability"}
    slim_rows = [
        {key: value for key, value in row.items() if key not in slim_drop}
        for row in sorted(final_rows, key=lambda row: str(row["record_key"]))
    ]
    slim_quarantined = [
        {key: value for key, value in row.items() if key not in slim_drop}
        for row in sorted(quarantined_rows, key=lambda row: str(row.get("record_key")))
    ]
    slim_revision_drift = [
        {key: value for key, value in row.items() if key not in slim_drop}
        for row in sorted(revision_drift_rows, key=lambda row: str(row.get("record_key")))
    ]
    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **provenance,
        "seed": args.seed,
        "seeds": args.seeds,
        "native_deterministic": bool(args.native_deterministic),
        "native_timeout": args.native_timeout,
        "engine_timeout": args.engine_timeout,
        "workers": args.workers,
        "score_workers": args.score_workers,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
        "corpus_dir": str(args.corpus_dir),
        "subset_path": None if args.subset is None else str(args.subset),
        "subset_sha256": subset_sha256,
        "subset": subset_info,
        "engine_field": engines,
        "engine_exclusions": dict(GLADOS_ENGINE_EXCLUSIONS),
        "engine_availability": availability,
        "scoring_signature": signature,
        "graph_count": len(entries),
        "load_stats": load_stats,
        "load_rows": load_rows,
        "rows": slim_rows,
        "quarantined_rows": slim_quarantined,
        "revision_drift_rows": slim_revision_drift,
        "run_revision_markers": revision_markers,
        "tally": tally,
        "warnings": warnings,
        "assumptions": assumptions,
        "aborted": aborted_reason is not None,
        "aborted_reason": aborted_reason,
    }
    write_report(staging, payload, row_map, tally, report_context, assumptions)
    try:
        publish_results(args.output_dir, staging, payload)
    except Exception as exc:  # noqa: BLE001
        print(
            f"PUBLISH FAILED: {type(exc).__name__}: {exc}. The prior published run is "
            f"untouched (or restored); the candidate data is preserved in {staging} "
            "(or the .failed-publish sibling) for inspection and --resume.",
            file=sys.stderr,
        )
        return 1
    # Publish succeeded: the pre-resume snapshots (which traveled with the
    # staging rename) have served their purpose (R2 B4-F1).
    removed_snapshots = clean_row_store_snapshots(args.output_dir)
    if removed_snapshots:
        print(f"removed {removed_snapshots} pre-resume row-store snapshot(s)", flush=True)
    print(
        f"published {args.output_dir}: tally {tally['overall']}",
        flush=True,
    )

    if args.archive_dir is not None:
        warning = archive_run(args.output_dir, args.archive_dir)
        if warning:
            print(f"WARNING: {warning}", file=sys.stderr)

    return 3 if aborted_reason is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
