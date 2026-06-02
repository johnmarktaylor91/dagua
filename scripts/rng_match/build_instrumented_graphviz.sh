#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tools_dir="${TOOLS_DIR:-$HOME/tools}"
conda_prefix="${CONDA_PREFIX:-$HOME/anaconda3/envs/py311}"
stock_dot="${STOCK_DOT:-$conda_prefix/bin/dot}"

version="7.0.5"
release_url="https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/${version}/graphviz-${version}.tar.gz"
release_sha_url="${release_url}.sha256"
cmake_url="https://gitlab.com/graphviz/graphviz/-/archive/${version}/graphviz-${version}.tar.gz"

release_tar="$tools_dir/graphviz-${version}.tar.gz"
release_sha="$release_tar.sha256"
cmake_tar="$tools_dir/graphviz-${version}-gitlab.tar.gz"
src_dir="$tools_dir/graphviz-${version}-src"
clean_dir="$tools_dir/graphviz-${version}-src.clean"
build_dir="$tools_dir/graphviz-${version}-build"
prefix="$tools_dir/graphviz-${version}-instr"
pkgconfig_dir="$tools_dir/graphviz-${version}-pkgconfig"
patch_file="$repo_root/scripts/rng_match/graphviz_7.0.5_instrumentation.patch"

mkdir -p "$tools_dir" "$pkgconfig_dir"

curl -L "$release_url" -o "$release_tar"
curl -L "$release_sha_url" -o "$release_sha"
(cd "$tools_dir" && sha256sum -c "$(basename "$release_sha")")

# The official release tarball is verified above, but the 7.0.5 release archive
# does not include the CMake project files. Use the matching GitLab tag archive
# for the actual CMake build.
curl -L "$cmake_url" -o "$cmake_tar"
rm -rf "$src_dir" "$clean_dir" "$build_dir" "$prefix"
tar -xzf "$cmake_tar" -C "$tools_dir"
mv "$tools_dir/graphviz-${version}" "$src_dir"
cp -a "$src_dir" "$clean_dir"
(cd "$src_dir" && patch -p1 < "$patch_file")

if [ ! -f "$conda_prefix/lib/pkgconfig/gts.pc" ]; then
  echo "Missing $conda_prefix/lib/pkgconfig/gts.pc; cannot build GTS-backed sfdp." >&2
  exit 1
fi
cp "$conda_prefix/lib/pkgconfig/gts.pc" "$pkgconfig_dir/gts.pc"

gts_cflags="-DHAVE_GTS -I$conda_prefix/include -I$conda_prefix/include/glib-2.0 -I$conda_prefix/lib/glib-2.0/include -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include"
mkdir -p "$build_dir"
PKG_CONFIG_PATH="$pkgconfig_dir" \
LIBRARY_PATH="$conda_prefix/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
cmake -S "$src_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib;$ORIGIN/../lib/graphviz' \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$build_dir/lib/xdot" \
  -DCMAKE_C_FLAGS="$gts_cflags" \
  -DCMAKE_CXX_FLAGS="$gts_cflags" \
  -Dwith_smyrna=OFF \
  -Dwith_gtk=OFF \
  -Dwith_gvedit=OFF \
  -Dwith_lasi=OFF \
  -Dwith_webp=OFF \
  -Dwith_poppler=OFF \
  -Dwith_devil=OFF \
  -Dwith_gdk=OFF \
  -Dwith_gdk_pixbuf=OFF \
  -Dwith_ghostscript=OFF \
  -Dwith_go=OFF \
  -Dwith_java=OFF \
  -Dwith_lua=OFF \
  -Dwith_perl=OFF \
  -Dwith_php=OFF \
  -Dwith_python=OFF \
  -Dwith_r=OFF \
  -Dwith_ruby=OFF \
  -Dwith_tcl=OFF \
  -Dwith_qt=OFF \
  -Dwith_ipsepcola=OFF

set +e
PKG_CONFIG_PATH="$pkgconfig_dir" \
LIBRARY_PATH="$conda_prefix/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
cmake --build "$build_dir" --target install
build_status=$?
cmake --install "$build_dir" --prefix "$prefix"
install_status=$?
set -e

if [ "$build_status" -ne 0 ] || [ "$install_status" -ne 0 ]; then
  echo "Graphviz CMake stopped on auxiliary gvmap/cluster, checking required dot artifacts." >&2
fi
test -x "$prefix/bin/dot_builtins"
test -f "$prefix/lib/graphviz/libgvplugin_neato_layout.so"
test -f "$prefix/lib/graphviz/libgvplugin_core.so"

if [ -f "$prefix/bin/dot" ] && [ ! -f "$prefix/bin/dot.dynamic" ]; then
  mv "$prefix/bin/dot" "$prefix/bin/dot.dynamic"
fi
cat > "$prefix/bin/dot" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
prefix="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_prefix="${CONDA_PREFIX:-$HOME/anaconda3/envs/py311}"
export LD_LIBRARY_PATH="${prefix}/lib:${prefix}/lib/graphviz:${conda_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${prefix}/bin/dot_builtins" "$@"
WRAPPER
chmod +x "$prefix/bin/dot"

"$prefix/bin/dot" -V

python - "$stock_dot" "$prefix/bin/dot" <<'PY'
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

STOCK = Path(sys.argv[1])
INSTR = Path(sys.argv[2])
ENGINES = ["fdp", "neato", "sfdp"]
SEEDS = [1, 2, 3]
GRAPHS = {
    "path8": "graph G { 0--1--2--3--4--5--6--7; }\n",
    "star8": "graph G { 0--1; 0--2; 0--3; 0--4; 0--5; 0--6; 0--7; }\n",
    "grid3x3": "graph G { 0--1--2; 3--4--5; 6--7--8; 0--3--6; 1--4--7; 2--5--8; }\n",
    "cycle6": "graph G { 0--1--2--3--4--5--0; }\n",
    "complete5": "graph G { 0--1; 0--2; 0--3; 0--4; 1--2; 1--3; 1--4; 2--3; 2--4; 3--4; }\n",
    "tree7": "graph G { 0--1; 0--2; 1--3; 1--4; 2--5; 2--6; }\n",
}
NODE_RE = re.compile(r"^node\s+(\S+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+")


def run_dot(dot: Path, engine: str, seed: int, graph: Path) -> bytes:
    """Run Graphviz and return plain-format bytes.

    Parameters
    ----------
    dot
        Graphviz executable path.
    engine
        Layout engine name.
    seed
        Seed used for both `seed` and `start` graph attributes.
    graph
        DOT graph path.

    Returns
    -------
    bytes
        Raw `-Tplain` output.
    """
    env = os.environ.copy()
    env.pop("GV_TRACE", None)
    env.pop("GV_TRACE_FILE", None)
    return subprocess.check_output(
        [str(dot), f"-K{engine}", f"-Gseed={seed}", f"-Gstart={seed}", "-Tplain", str(graph)],
        stderr=subprocess.PIPE,
        env=env,
    )


def positions(plain: bytes) -> dict[str, tuple[float, float]]:
    """Parse node positions from Graphviz plain output.

    Parameters
    ----------
    plain
        Raw `-Tplain` bytes.

    Returns
    -------
    dict[str, tuple[float, float]]
        Node id to `(x, y)` coordinates.
    """
    out: dict[str, tuple[float, float]] = {}
    for line in plain.decode("utf-8").splitlines():
        match = NODE_RE.match(line)
        if match:
            out[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return out


def rmsd(a: dict[str, tuple[float, float]], b: dict[str, tuple[float, float]]) -> float:
    """Compute RMSD across matching node positions.

    Parameters
    ----------
    a
        First node-position mapping.
    b
        Second node-position mapping.

    Returns
    -------
    float
        Root mean square coordinate delta.
    """
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        ax, ay = a[key]
        bx, by = b[key]
        total += (ax - bx) ** 2 + (ay - by) ** 2
    return math.sqrt(total / len(keys))


def main() -> int:
    """Run the stock-vs-instrumented veridical comparison.

    Returns
    -------
    int
        Process status code.
    """
    mismatches: list[str] = []
    max_rmsd = 0.0
    exact_count = 0
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        graph_paths = {}
        for name, text in GRAPHS.items():
            graph_path = tmpdir / f"{name}.dot"
            graph_path.write_text(text, encoding="utf-8")
            graph_paths[name] = graph_path
        for graph_name, graph_path in graph_paths.items():
            for engine in ENGINES:
                for seed in SEEDS:
                    total += 1
                    stock = run_dot(STOCK, engine, seed, graph_path)
                    instr = run_dot(INSTR, engine, seed, graph_path)
                    value = rmsd(positions(stock), positions(instr))
                    max_rmsd = max(max_rmsd, value)
                    if stock == instr:
                        exact_count += 1
                    else:
                        mismatches.append(f"{graph_name} {engine} seed={seed} rmsd={value:.17g}")
    print(f"total={total}")
    print(f"exact_bytes={exact_count}")
    print(f"max_rmsd={max_rmsd:.17g}")
    if mismatches:
        print("mismatches:")
        for item in mismatches:
            print(item)
        return 1
    print("result=PASS bit-for-bit identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
