#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-eval_output/stdcorpora}"
MIN_FREE_KB=$((10 * 1024 * 1024))
MAX_BYTES=$((200 * 1024 * 1024))

check_space() {
  local free_kb
  free_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
  if [ "${free_kb}" -lt "${MIN_FREE_KB}" ]; then
    mkdir -p "${OUT_DIR}"
    cat > "${OUT_DIR}/README.md" <<'EOF'
# Standard Corpora Inputs

Fetch aborted because `/` had less than 10GB free.

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

- `eval_output/stdcorpora/rome/` with a small sample of Rome-Lib `.graph` or `.gml` files
- `eval_output/stdcorpora/north/` with a small sample of North/AT&T DAG `.graph` or `.gml` files
- `eval_output/stdcorpora/suitesparse/` with small SuiteSparse structural matrices as `.mtx`

Official sources to use manually:

- Rome-Lib / North graph archive: https://graphdrawing.org/data.html
- SuiteSparse Matrix Collection: https://sparse.tamu.edu/

The eval harness accepts dropped-in files with extensions `.graph`, `.gml`, and `.mtx`.
EOF
}

check_space
mkdir -p "${OUT_DIR}/downloads" "${OUT_DIR}/rome" "${OUT_DIR}/north" "${OUT_DIR}/suitesparse"

before_bytes="$(du -sb "${OUT_DIR}" | awk '{print $1}')"

ROME_URL="https://graphdrawing.org/data/rome/rome.tar.gz"
NORTH_URL="https://graphdrawing.org/data/north/north.tar.gz"

if ! fetch_twice "${ROME_URL}" "${OUT_DIR}/downloads/rome.tar.gz"; then
  write_readme
  exit 0
fi
if ! fetch_twice "${NORTH_URL}" "${OUT_DIR}/downloads/north.tar.gz"; then
  write_readme
  exit 0
fi

tar -xzf "${OUT_DIR}/downloads/rome.tar.gz" -C "${OUT_DIR}/downloads"
tar -xzf "${OUT_DIR}/downloads/north.tar.gz" -C "${OUT_DIR}/downloads"

find "${OUT_DIR}/downloads" -type f \( -name '*.graph' -o -name '*.gml' \) | grep -i rome | head -n 24 | while read -r file; do
  cp "${file}" "${OUT_DIR}/rome/"
done
find "${OUT_DIR}/downloads" -type f \( -name '*.graph' -o -name '*.gml' \) | grep -Ei 'north|dag' | head -n 24 | while read -r file; do
  cp "${file}" "${OUT_DIR}/north/"
done

if command -v ssgetpy >/dev/null 2>&1; then
  OUT_DIR="${OUT_DIR}" .venv/bin/python - <<'PY'
import os
from pathlib import Path
import ssgetpy

out = Path(os.environ["OUT_DIR"]) / "suitesparse"
out.mkdir(parents=True, exist_ok=True)
matches = ssgetpy.search(rowbounds=(2, 2000), colbounds=(2, 2000), limit=8)
for matrix in matches:
    matrix.download(destpath=out, extract=True)
PY
else
  cat > "${OUT_DIR}/suitesparse/README.md" <<'EOF'
# SuiteSparse sample

`ssgetpy` was not installed, so SuiteSparse matrices were not downloaded automatically.
Download a few small Matrix Market `.mtx` structural matrices from https://sparse.tamu.edu/
and place them in this directory.
EOF
fi

after_bytes="$(du -sb "${OUT_DIR}" | awk '{print $1}')"
delta_bytes=$((after_bytes - before_bytes))
if [ "${delta_bytes}" -gt "${MAX_BYTES}" ]; then
  write_readme
  echo "Download exceeded 200MB hard limit; removing fetched data." >&2
  find "${OUT_DIR}/rome" "${OUT_DIR}/north" "${OUT_DIR}/suitesparse" -type f ! -name README.md -delete
  exit 1
fi

find "${OUT_DIR}/rome" "${OUT_DIR}/north" "${OUT_DIR}/suitesparse" -type f | sort > "${OUT_DIR}/FETCHED_FILES.txt"
check_space
