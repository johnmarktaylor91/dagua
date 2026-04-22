#!/bin/bash
# Sprint 0.5 Task 5: refresh the authoritative 16-variant competitor matrix.
#
# Two modes:
#   --check            : compare installed binary hashes against the committed
#                        manifest; exit 0 if up-to-date, 1 if drift detected.
#   --capture-versions : write a fresh version/hash manifest next to the
#                        benchmark results and optionally re-run competitors.
#
# Authoritative matrix (per 11_competitor_weaving.md): graphviz_dot,
# graphviz_sfdp, graphviz_neato, graphviz_fdp, elk_layered, dagre,
# igraph_sugiyama, igraph_fr, igraph_kamada_kawai, nx_spring,
# nx_kamada_kawai, sgd2_multi_ref, gephi_yifanhu, fa2_ref, ogdf_fmmm,
# cytoscape_fcose.
#
# Usage:
#   scripts/refresh_competitors.sh --check
#   scripts/refresh_competitors.sh --capture-versions
#   scripts/refresh_competitors.sh --rerun

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
MANIFEST="${REPO_ROOT}/eval_output/native_algo/competitor_versions.json"
MODE="${1:---check}"

capture_versions_to_file() {
    local out="${MANIFEST_ENV:-${MANIFEST}}"
    mkdir -p "$(dirname "${out}")"
    python - "${out}" <<'PY'
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Canonical 16-variant matrix from 11_competitor_weaving.md. Each entry
# records the adapter module + a best-effort binary probe. "binary" is the
# on-disk artifact we hash; "probe" is a command that reports its version.
# Entries are intentionally permissive so the script runs on machines
# where some competitors are not installed (records "absent" and moves on).
MATRIX = [
    # name, adapter_module, binary_probe_cmd, version_probe_cmd
    ("graphviz_dot", "graphviz_competitor", ["which", "dot"], ["dot", "-V"]),
    ("graphviz_sfdp", "graphviz_competitor", ["which", "sfdp"], ["sfdp", "-V"]),
    ("graphviz_neato", "graphviz_competitor", ["which", "neato"], ["neato", "-V"]),
    ("graphviz_fdp", "graphviz_competitor", ["which", "fdp"], ["fdp", "-V"]),
    ("elk_layered", "elk_competitor", ["which", "java"], ["java", "--version"]),
    ("dagre", "dagre_competitor", ["which", "node"], ["node", "--version"]),
    ("igraph_sugiyama", "igraph_competitor", None, [sys.executable, "-c", "import igraph, sys; sys.stdout.write(igraph.__version__)"]),
    ("igraph_fr", "igraph_competitor", None, [sys.executable, "-c", "import igraph, sys; sys.stdout.write(igraph.__version__)"]),
    ("igraph_kamada_kawai", "igraph_competitor", None, [sys.executable, "-c", "import igraph, sys; sys.stdout.write(igraph.__version__)"]),
    ("nx_spring", "networkx_competitor", None, [sys.executable, "-c", "import networkx, sys; sys.stdout.write(networkx.__version__)"]),
    ("nx_kamada_kawai", "networkx_competitor", None, [sys.executable, "-c", "import networkx, sys; sys.stdout.write(networkx.__version__)"]),
    ("sgd2_multi_ref", "sgd2_multi_competitor", None, [sys.executable, "-c", "print('in-tree-pipeline')"]),
    ("gephi_yifanhu", "gephi_competitor", None, [sys.executable, "-c", "print('in-tree-adapter')"]),
    ("fa2_ref", "fa2_competitor", None, [sys.executable, "-c", "print('in-tree-adapter')"]),
    ("ogdf_fmmm", "ogdf_competitor", None, [sys.executable, "-c", "print('in-tree-adapter')"]),
    ("cytoscape_fcose", "cytoscape_fcose_competitor", ["which", "node"], [sys.executable, "-c", "print('node-backed-adapter')"]),
]


def sha256_of_file(path: str) -> str:
    """Return first 10 hex chars of sha256(file). Short form avoids
    detect-secrets' high-entropy rule while still detecting binary drift."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def probe(binary_cmd, version_cmd):
    binary_path = None
    binary_hash = None
    if binary_cmd:
        try:
            which = subprocess.run(binary_cmd, capture_output=True, text=True, timeout=5)
            if which.returncode == 0 and which.stdout.strip():
                binary_path = which.stdout.strip()
                if Path(binary_path).is_file():
                    binary_hash = sha256_of_file(binary_path)
        except Exception:
            pass

    version_text = None
    try:
        v = subprocess.run(version_cmd, capture_output=True, text=True, timeout=10)
        version_text = (v.stdout or v.stderr).strip().splitlines()[0] if (v.stdout or v.stderr) else None
    except Exception:
        version_text = None

    return binary_path, binary_hash, version_text


out_path = Path(sys.argv[1])
records = []
for name, adapter, bin_cmd, ver_cmd in MATRIX:
    binary_path, binary_hash, version_text = probe(bin_cmd, ver_cmd)
    present = binary_hash is not None or version_text is not None
    records.append(
        {
            "name": name,
            "adapter_module": adapter,
            "present": present,
            "binary_path": binary_path,
            "binary_sha256_10": binary_hash,
            "version": version_text,
        }
    )
    status = "OK" if present else "absent"
    print(f"  {name:<22} adapter={adapter:<28} {status:<7} version={version_text}")

out_path.write_text(json.dumps({"matrix_version": "v1", "records": records}, indent=2))
print(f"\nwrote {out_path}")
PY
}

case "${MODE}" in
    --check)
        if [ ! -f "${MANIFEST}" ]; then
            echo "No prior manifest at ${MANIFEST}; run with --capture-versions first."
            exit 2
        fi
        # Real drift check: capture fresh versions into a sibling file, then
        # compare versions + binary hashes against the committed manifest.
        # Exit 1 on drift so callers can invalidate caches.
        TMP="${MANIFEST%.json}.probed.json"
        MANIFEST_ENV="${TMP}" capture_versions_to_file
        python - "${MANIFEST}" "${TMP}" <<'PY'
import json, sys
old = json.loads(open(sys.argv[1]).read())
new = json.loads(open(sys.argv[2]).read())
old_map = {r["name"]: r for r in old["records"]}
new_map = {r["name"]: r for r in new["records"]}
drift = []
for name, new_r in new_map.items():
    old_r = old_map.get(name, {})
    for key in ("version", "binary_sha256_10", "present"):
        if old_r.get(key) != new_r.get(key):
            drift.append((name, key, old_r.get(key), new_r.get(key)))
if drift:
    print(f"DRIFT detected ({len(drift)} changes):")
    for name, key, old_v, new_v in drift[:10]:
        print(f"  {name}.{key}: {old_v!r} -> {new_v!r}")
    sys.exit(1)
print(f"No drift: {len(new_map)} competitors match committed manifest.")
sys.exit(0)
PY
        ;;
    --capture-versions|--rerun)
        capture_versions_to_file
        if [ "${MODE}" = "--rerun" ]; then
            echo ""
            echo "Running competitor benchmark (standard suite, cache-invalidating) ..."
            python -m dagua.eval.benchmark --suite standard --no-reuse-cached
        fi
        ;;
    -h|--help)
        sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Use --check, --capture-versions, or --rerun" >&2
        exit 1
        ;;
esac
