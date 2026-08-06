#!/usr/bin/env bash
# Fetch the GLaDOS standard corpora: 60 Rome + 60 North + 20 SuiteSparse
# (plan 7.1). Deterministic candidate selection (sorted, WP10-F18), one
# format per stem (WP10-F05), square matrices only (WP10-F06), working
# ssgetpy probe with A4 fallback chain (WP10-F17), 15GB floor / 1GB cap
# (WP10-F20), downloads/ cleaned after extraction. FETCHED_FILES.txt is the
# candidate universe: supported extensions only (WP10-F19), written LAST and
# only on success.
set -euo pipefail

OUT_DIR="${1:-eval_output/stdcorpora}"
PYTHON="${PYTHON:-/home/jtaylor/anaconda3/envs/py311/bin/python}"
MIN_FREE_KB=$((15 * 1024 * 1024))
MAX_BYTES=$((1024 * 1024 * 1024))
PER_CORPUS_TEXT=60
SUITESPARSE_COUNT=20

# Fallback-2 matrices: small square Matrix Market files from sparse.tamu.edu,
# pre-declared by NAME/SIZE METADATA ONLY (HB group, all rows==cols, n<=2000;
# blind selection stays intact -- no content was inspected).
SUITESPARSE_FALLBACK_URLS=(
  "https://sparse.tamu.edu/MM/HB/bcspwr01.tar.gz"
  "https://sparse.tamu.edu/MM/HB/bcspwr02.tar.gz"
  "https://sparse.tamu.edu/MM/HB/bcspwr03.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_24.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_61.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_62.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_73.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_96.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_144.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_161.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_187.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_229.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_256.tar.gz"
  "https://sparse.tamu.edu/MM/HB/can_268.tar.gz"
  "https://sparse.tamu.edu/MM/HB/dwt_59.tar.gz"
  "https://sparse.tamu.edu/MM/HB/dwt_66.tar.gz"
  "https://sparse.tamu.edu/MM/HB/dwt_72.tar.gz"
  "https://sparse.tamu.edu/MM/HB/dwt_87.tar.gz"
  "https://sparse.tamu.edu/MM/HB/dwt_162.tar.gz"
  "https://sparse.tamu.edu/MM/HB/dwt_193.tar.gz"
  "https://sparse.tamu.edu/MM/HB/dwt_209.tar.gz"
  "https://sparse.tamu.edu/MM/HB/curtis54.tar.gz"
  "https://sparse.tamu.edu/MM/HB/will57.tar.gz"
  "https://sparse.tamu.edu/MM/HB/ash85.tar.gz"
  "https://sparse.tamu.edu/MM/HB/ibm32.tar.gz"
  "https://sparse.tamu.edu/MM/HB/nos4.tar.gz"
)

check_space() {
  local free_kb
  free_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
  if [ "${free_kb}" -lt "${MIN_FREE_KB}" ]; then
    mkdir -p "${OUT_DIR}"
    cat > "${OUT_DIR}/README.md" <<'EOF'
# Standard Corpora Inputs

Fetch aborted because `/` had less than 15GB free (A-S3 floor).

Expected layout:

- `eval_output/stdcorpora/rome/` with Rome-Lib `.graph` or `.gml` files
- `eval_output/stdcorpora/north/` with North/AT&T DAG `.graph` or `.gml` files
- `eval_output/stdcorpora/suitesparse/` with SuiteSparse Matrix Market `.mtx` files

Official sources:

- Rome-Lib / North graph archive: https://graphdrawing.org/data.html
- SuiteSparse Matrix Collection: https://sparse.tamu.edu/
EOF
    exit 1
  fi
}

fetch_twice() {
  local url="$1"
  local dest="$2"
  local attempt
  for attempt in 1 2; do
    if curl -L --fail --max-time 120 --retry 0 -o "${dest}" "${url}"; then
      return 0
    fi
    rm -f "${dest}"
  done
  return 1
}

write_readme() {
  mkdir -p "${OUT_DIR}"
  cat > "${OUT_DIR}/README.md" <<'EOF'
# Standard Corpora Inputs

Automatic acquisition did not complete. No synthetic replacement graphs were created.

Expected layout:

- `eval_output/stdcorpora/rome/` with a sample of Rome-Lib `.graph` or `.gml` files
- `eval_output/stdcorpora/north/` with a sample of North/AT&T DAG `.graph` or `.gml` files
- `eval_output/stdcorpora/suitesparse/` with small SuiteSparse structural matrices as `.mtx`

Official sources to use manually:

- Rome-Lib / North graph archive: https://graphdrawing.org/data.html
- SuiteSparse Matrix Collection: https://sparse.tamu.edu/

The eval harness accepts dropped-in files with extensions `.graph`, `.gml`,
`.graphml`, and `.mtx`.
EOF
}

# Copy up to $3 files matching pattern $1 into $2, deterministically sorted,
# ONE format per stem (prefer .graph over .gml -- WP10-F05/F18). The pattern
# is matched against paths RELATIVE to downloads/ (find -printf '%P'), never
# the full path: an absolute OUT_DIR under .../projects/dagua/ used to make
# 'dag' match EVERY file via the repo directory name and mis-corpus Rome
# graphs into north/ (dry-well B4-F8, the WP10-F02 trap one layer earlier).
select_text_corpus() {
  local pattern="$1"
  local dest="$2"
  local limit="$3"
  local count=0
  local ext rel stem
  declare -A seen_stems=()
  for ext in graph gml graphml; do
    while IFS= read -r rel; do
      if [ "${count}" -ge "${limit}" ]; then
        break
      fi
      stem="$(basename "${rel}" ".${ext}")"
      if [ -z "${seen_stems[${stem}]:-}" ]; then
        seen_stems[${stem}]=1
        cp "${OUT_DIR}/downloads/${rel}" "${dest}/"
        count=$((count + 1))
      fi
    done < <(find "${OUT_DIR}/downloads" -type f -name "*.${ext}" -printf '%P\n' | grep -Ei "${pattern}" | sort || true)
  done
  echo "selected ${count} files for ${dest}" >&2
}

# SuiteSparse via ssgetpy with the given interpreter. Downloads into
# downloads/ssgetpy, then copies ONLY square .mtx files flat into
# suitesparse/ (WP10-F06: rectangular matrices excluded up front).
fetch_suitesparse_ssgetpy() {
  OUT_DIR="${OUT_DIR}" SUITESPARSE_COUNT="${SUITESPARSE_COUNT}" "${PYTHON}" - <<'PY'
import os
import shutil
from pathlib import Path

import ssgetpy

out_dir = Path(os.environ["OUT_DIR"])
count = int(os.environ["SUITESPARSE_COUNT"])
downloads = out_dir / "downloads" / "ssgetpy"
downloads.mkdir(parents=True, exist_ok=True)
destination = out_dir / "suitesparse"
destination.mkdir(parents=True, exist_ok=True)

matches = ssgetpy.search(rowbounds=(2, 2000), colbounds=(2, 2000), limit=40)
square = [matrix for matrix in matches if matrix.rows == matrix.cols][:count]
if not square:
    raise SystemExit("ssgetpy returned no square matrices")
for matrix in square:
    matrix.download(destpath=str(downloads), extract=True)
copied = 0
for mtx in sorted(downloads.rglob("*.mtx")):
    target = destination / mtx.name
    if not target.exists():
        shutil.copyfile(mtx, target)
        copied += 1
print(f"ssgetpy: copied {copied} square .mtx files")
PY
}

# Fallback 2: curl the pre-declared square-matrix list.
fetch_suitesparse_curl() {
  local url tarball name
  local fetched=0
  mkdir -p "${OUT_DIR}/downloads/ss_curl" "${OUT_DIR}/suitesparse"
  for url in "${SUITESPARSE_FALLBACK_URLS[@]}"; do
    if [ "${fetched}" -ge "${SUITESPARSE_COUNT}" ]; then
      break
    fi
    name="$(basename "${url}" .tar.gz)"
    tarball="${OUT_DIR}/downloads/ss_curl/${name}.tar.gz"
    if fetch_twice "${url}" "${tarball}"; then
      if tar -xzf "${tarball}" -C "${OUT_DIR}/downloads/ss_curl"; then
        find "${OUT_DIR}/downloads/ss_curl/${name}" -type f -name '*.mtx' \
          -exec cp {} "${OUT_DIR}/suitesparse/" \; 2>/dev/null || true
        fetched=$((fetched + 1))
      fi
    fi
  done
  [ "${fetched}" -gt 0 ]
}

# Selection freeze: once the blind subset exists, the candidate universe is
# sealed (plan 7.2 merge-freeze anchor) -- a re-fetch would rewrite it.
if [ -e "${OUT_DIR}/SUBSET.json" ]; then
  echo "REFUSING to fetch: ${OUT_DIR}/SUBSET.json exists (blind selection is frozen)." >&2
  exit 1
fi

check_space
mkdir -p "${OUT_DIR}/downloads" "${OUT_DIR}/rome" "${OUT_DIR}/north" "${OUT_DIR}/suitesparse"

# Single-generation guarantee (dry-well B4-F8): wipe prior corpus files so a
# rerun cannot mix two fetch generations into one candidate universe;
# FETCHED_FILES.txt below then reflects EXACTLY this run's selection.
find "${OUT_DIR}/rome" "${OUT_DIR}/north" "${OUT_DIR}/suitesparse" -type f -delete

ROME_URL="https://graphdrawing.unipg.it/data/rome-graphml.tgz"
NORTH_URL="https://graphdrawing.unipg.it/data/north-graphml.tgz"

# A failed rome/north acquisition is a FAILED fetch: exit non-zero so an
# orchestrator chaining `fetch && subset && run` self-arrests here
# (dry-well B4-F8; the old `exit 0` lied to the chain).
if ! fetch_twice "${ROME_URL}" "${OUT_DIR}/downloads/rome.tar.gz"; then
  write_readme
  exit 1
fi
if ! fetch_twice "${NORTH_URL}" "${OUT_DIR}/downloads/north.tar.gz"; then
  write_readme
  exit 1
fi

tar -xzf "${OUT_DIR}/downloads/rome.tar.gz" -C "${OUT_DIR}/downloads"
tar -xzf "${OUT_DIR}/downloads/north.tar.gz" -C "${OUT_DIR}/downloads"

select_text_corpus 'rome' "${OUT_DIR}/rome" "${PER_CORPUS_TEXT}"
select_text_corpus 'north|dag' "${OUT_DIR}/north" "${PER_CORPUS_TEXT}"

# SuiteSparse: library probe (a Python import, NOT `command -v` -- ssgetpy has
# no console script, WP10-F17), then the A4 fallback chain.
suitesparse_ok=0
if "${PYTHON}" -c 'import ssgetpy' >/dev/null 2>&1; then
  if fetch_suitesparse_ssgetpy; then
    suitesparse_ok=1
  fi
fi
if [ "${suitesparse_ok}" -eq 0 ]; then
  echo "ssgetpy unavailable or failed; trying pip install (A4 fallback 1)" >&2
  if "${PYTHON}" -m pip install --quiet ssgetpy >/dev/null 2>&1 \
      && "${PYTHON}" -c 'import ssgetpy' >/dev/null 2>&1 \
      && fetch_suitesparse_ssgetpy; then
    suitesparse_ok=1
  fi
fi
if [ "${suitesparse_ok}" -eq 0 ]; then
  echo "ssgetpy still failing; curling pre-declared square .mtx list (A4 fallback 2)" >&2
  if fetch_suitesparse_curl; then
    suitesparse_ok=1
  fi
fi
DEGRADED=0
if [ "${suitesparse_ok}" -eq 0 ]; then
  DEGRADED=1
  echo "SuiteSparse acquisition failed on all fallbacks; degrading to 2 corpora (R-7)" >&2
  cat > "${OUT_DIR}/suitesparse/README.md" <<'EOF'
# SuiteSparse sample

Automatic acquisition failed (ssgetpy and the curl fallback list). The run
proceeds with 2 corpora. Download a few small SQUARE Matrix Market `.mtx`
structural matrices from https://sparse.tamu.edu/ and place them here.
EOF
fi

# Downloads are never part of the corpus: clean them BEFORE the size
# accounting so tarballs cannot trip the cap (WP10-F20).
rm -rf "${OUT_DIR}/downloads"

total_bytes="$(du -sb "${OUT_DIR}" | awk '{print $1}')"
if [ "${total_bytes}" -gt "${MAX_BYTES}" ]; then
  write_readme
  echo "Fetched corpus exceeded the 1GB hard cap; removing fetched data." >&2
  find "${OUT_DIR}/rome" "${OUT_DIR}/north" "${OUT_DIR}/suitesparse" -type f ! -name README.md -delete
  exit 1
fi

# The candidate universe (plan 7.1): supported extensions only, sorted,
# written LAST and only on success (WP10-F19).
find "${OUT_DIR}/rome" "${OUT_DIR}/north" "${OUT_DIR}/suitesparse" -type f \
  \( -name '*.graph' -o -name '*.gml' -o -name '*.graphml' -o -name '*.mtx' \) \
  | sort > "${OUT_DIR}/FETCHED_FILES.txt"
echo "FETCHED_FILES.txt: $(wc -l < "${OUT_DIR}/FETCHED_FILES.txt") candidates" >&2
check_space
# Exit 3 = pre-registered SuiteSparse degradation (R-7): the 2-corpora
# universe IS usable (FETCHED_FILES.txt written), but the orchestrator must
# see a non-zero code rather than a silent success (dry-well B4-F8).
if [ "${DEGRADED}" -eq 1 ]; then
  exit 3
fi
