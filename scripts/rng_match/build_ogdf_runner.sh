#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OGDF_TAG="${OGDF_TAG:-foxglove-202510}"
TOOLS_ROOT="${TOOLS_ROOT:-/home/jtaylor/tools}"
OGDF_SRC="${OGDF_SRC:-${TOOLS_ROOT}/ogdf-src}"
OGDF_BUILD="${OGDF_BUILD:-${TOOLS_ROOT}/ogdf-build}"
OGDF_PREFIX="${OGDF_PREFIX:-${TOOLS_ROOT}/ogdf}"
JOBS="${JOBS:-8}"

mkdir -p "${TOOLS_ROOT}"

if [[ -d "${OGDF_SRC}/.git" ]]; then
    git -C "${OGDF_SRC}" fetch --tags origin "${OGDF_TAG}"
    git -C "${OGDF_SRC}" checkout --detach "${OGDF_TAG}"
else
    git clone --branch "${OGDF_TAG}" --depth 1 https://github.com/ogdf/ogdf.git "${OGDF_SRC}"
fi

cmake -S "${OGDF_SRC}" -B "${OGDF_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OGDF_PREFIX}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DOGDF_INCLUDE_CGAL=OFF \
    -DOGDF_SEPARATE_TESTS=OFF \
    -DDOC_INSTALL=OFF
cmake --build "${OGDF_BUILD}" --target install --parallel "${JOBS}"

test -f "${OGDF_PREFIX}/include/ogdf/basic/Graph.h"
test -f "${OGDF_PREFIX}/lib/libOGDF.a"
test -f "${OGDF_PREFIX}/lib/libCOIN.a"

g++ -std=c++17 -O2 "${REPO_ROOT}/scripts/ogdf_runner.cpp" \
    -I"${OGDF_PREFIX}/include" \
    -I"${OGDF_PREFIX}/include/ogdf-release" \
    -L"${OGDF_PREFIX}/lib" \
    -lOGDF -lCOIN -pthread \
    -o "${REPO_ROOT}/scripts/ogdf_runner"

"${REPO_ROOT}/scripts/ogdf_runner" --help >/dev/null

export LD_LIBRARY_PATH="/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}"
python "${REPO_ROOT}/scripts/rng_match/check_engine.py" classic_gem_iters100
python "${REPO_ROOT}/scripts/rng_match/check_engine.py" classic_gem_iters500
python "${REPO_ROOT}/scripts/rng_match/check_engine.py" classic_fmmm_steps100
