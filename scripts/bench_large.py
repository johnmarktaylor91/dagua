"""Large-scale layout benchmark — wide DAG.

Usage:
    python scripts/bench_large.py 50m
    python scripts/bench_large.py 100m
    python scripts/bench_large.py 300m
    python scripts/bench_large.py 1b
    python scripts/bench_large.py 10_000_000          # arbitrary node count
    python scripts/bench_large.py 50m --layers 500 --workers 8
    python scripts/bench_large.py 1b --device cuda
"""

import argparse
import atexit
import faulthandler
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import dagua


# ─── Helpers ──────────────────────────────────────────────────────────────────


def rss_gb():
    """Current process RSS in GB (Linux /proc/self/statm)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except Exception:
        return 0.0


def mem(label):
    gc.collect()
    print(f"  [{label}] RSS={rss_gb():.1f} GB", flush=True)


def phase(label: str, t0: float):
    print(f"[phase] {label} @ {time.perf_counter() - t0:.1f}s", flush=True)
    mem(label)


def _default_checkpoint_dir(size: str) -> Path:
    slug = size.strip().lower().replace("/", "_").replace(" ", "_")
    return Path("/tmp") / "dagua_bench_large" / slug


def _checkpoint_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "meta": root / "meta.json",
        "edge_index": root / "edge_index.pt",
        "node_sizes": root / "node_sizes.pt",
        "layer_assignments": root / "layer_assignments.pt",
        "positions": root / "positions.pt",
        "active_run": root / "active_run.json",
    }


def _save_checkpoint_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_graph_checkpoint(paths: dict[str, Path], n: int, layers: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not paths["meta"].exists() or not paths["edge_index"].exists() or not paths["node_sizes"].exists():
        return None
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if meta.get("n") != n or meta.get("layers") != layers:
        return None
    edge_index = torch.load(paths["edge_index"], map_location="cpu")
    node_sizes = torch.load(paths["node_sizes"], map_location="cpu")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return None
    if node_sizes.ndim == 1:
        if node_sizes.shape[0] != n:
            return None
    elif node_sizes.ndim == 2:
        if node_sizes.shape[0] != n or node_sizes.shape[1] not in (1, 2):
            return None
    else:
        return None
    return edge_index, node_sizes


def _load_layer_checkpoint(paths: dict[str, Path], n: int, layers: int) -> torch.Tensor | None:
    if not paths["meta"].exists() or not paths["layer_assignments"].exists():
        return None
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if meta.get("n") != n or meta.get("layers") != layers:
        return None
    layer_assignments = torch.load(paths["layer_assignments"], map_location="cpu")
    if layer_assignments.ndim != 1 or layer_assignments.shape[0] != n:
        return None
    return layer_assignments


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _find_existing_run_pid(size: str) -> int | None:
    """Best-effort process table scan for another large benchmark owner."""
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except Exception:
        return None

    needle = f"scripts/bench_large.py {size}"
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or needle not in line:
            continue
        try:
            pid_str, cmd = line.split(None, 1)
        except ValueError:
            continue
        pid = int(pid_str)
        if pid == os.getpid():
            continue
        cmd = cmd.strip()
        # Ignore wrapper shells such as `bash -c "... python ..."` and only
        # treat real Python owners as canonical benchmark processes.
        if not (
            cmd.startswith("python ")
            or cmd.startswith("python3 ")
            or cmd.startswith("/usr/bin/python ")
            or cmd.startswith("/usr/bin/python3 ")
        ):
            continue
        if _pid_alive(pid):
            return pid
    return None


def _guard_duplicate_run(paths: dict[str, Path], size: str, resume: bool, force_duplicate_run: bool) -> None:
    """Refuse to start a duplicate large run against the same checkpoint root."""
    if force_duplicate_run:
        return
    existing_pid = _find_existing_run_pid(size)
    if existing_pid is not None:
        raise SystemExit(
            "Refusing to start a duplicate large benchmark run for "
            f"{size!r} at {paths['root']}. Existing pid={existing_pid}. "
            "Use --force-duplicate-run only if you really want concurrent runs."
        )
    if not paths["active_run"].exists():
        return
    try:
        payload = json.loads(paths["active_run"].read_text(encoding="utf-8"))
    except Exception:
        return
    pid = int(payload.get("pid", -1))
    if pid <= 0 or pid == os.getpid() or not _pid_alive(pid):
        return
    raise SystemExit(
        "Refusing to start a duplicate large benchmark run for "
        f"{size!r} at {paths['root']}. Existing pid={pid}. "
        "Use --force-duplicate-run only if you really want concurrent runs."
    )


def _register_active_run(paths: dict[str, Path], size: str, resume: bool) -> None:
    payload = {
        "pid": os.getpid(),
        "size": size,
        "resume": resume,
        "checkpoint_root": str(paths["root"]),
    }
    paths["active_run"].parent.mkdir(parents=True, exist_ok=True)
    paths["active_run"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _cleanup() -> None:
        try:
            if paths["active_run"].exists():
                current = json.loads(paths["active_run"].read_text(encoding="utf-8"))
                if int(current.get("pid", -1)) == os.getpid():
                    paths["active_run"].unlink()
        except Exception:
            pass

    atexit.register(_cleanup)


def parse_node_count(s: str) -> int:
    """Parse node count from string: '50m' -> 50_000_000, '1b' -> 1_000_000_000."""
    s = s.strip().lower().replace("_", "").replace(",", "")
    if s.endswith("b"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)


# ─── Presets ──────────────────────────────────────────────────────────────────

PRESETS = {
    "50m": {"n": 50_000_000, "layers": 500},
    "100m": {"n": 100_000_000, "layers": 1000},
    "300m": {"n": 300_000_000, "layers": 1500},
    "1b": {"n": 1_000_000_000, "layers": 1500},
}

# Graphs above this threshold use chunked edge construction to limit peak memory.
CHUNK_THRESHOLD = 200_000_000


def resolve_size_and_layers(size: str, layers_override: int = 0) -> tuple[int, int, int]:
    """Resolve requested size to exact node count, layer count, and layer width."""
    key = size.lower().replace("_", "").replace(",", "")
    if key in PRESETS:
        n = PRESETS[key]["n"]
        layers = layers_override if layers_override > 0 else PRESETS[key]["layers"]
    else:
        n = parse_node_count(size)
        layers = layers_override if layers_override > 0 else max(int(n**0.5 / 10) * 10, 10)

    # Round upward to the next exact multiple so presets like 1b stay at or above
    # the requested size instead of dipping just below it.
    n = ((n + layers - 1) // layers) * layers
    w = n // layers
    return n, layers, w


# ─── Edge construction ────────────────────────────────────────────────────────


def build_edges(n: int, layers: int) -> torch.Tensor:
    """Build wide-DAG edge_index: backbone + ~50% cross-connections per layer."""
    w = n // layers
    e_backbone = n - w

    if n >= CHUNK_THRESHOLD:
        return _build_edges_chunked(n, w, e_backbone)
    else:
        return _build_edges_simple(n, w, e_backbone)


def _build_edges_simple(n: int, w: int, e_backbone: int) -> torch.Tensor:
    idx_dtype = torch.int32 if n <= torch.iinfo(torch.int32).max else torch.long
    src_backbone = torch.arange(0, n - w, dtype=idx_dtype)
    tgt_backbone = src_backbone + w

    cross_mask = torch.rand(n - w) < 0.5
    cross_src = torch.arange(0, n - w, dtype=idx_dtype)[cross_mask]
    cross_offset = torch.randint(0, w, (cross_src.shape[0],), dtype=idx_dtype)
    cross_tgt_layer = cross_src // w + 1
    cross_tgt = cross_tgt_layer * w + cross_offset

    edge_index = torch.stack([
        torch.cat([src_backbone, cross_src]),
        torch.cat([tgt_backbone, cross_tgt]),
    ])
    del src_backbone, tgt_backbone, cross_src, cross_offset, cross_tgt_layer, cross_tgt, cross_mask
    return edge_index


def _build_edges_chunked(n: int, w: int, e_backbone: int) -> torch.Tensor:
    """Build edges in chunks to limit peak memory for very large graphs."""
    idx_dtype = torch.int32 if n <= torch.iinfo(torch.int32).max else torch.long
    cross_prob = 0.5
    e_cross_est = int(e_backbone * cross_prob * 1.02)  # 2% margin
    e_est = e_backbone + e_cross_est

    edge_src = torch.empty(e_est, dtype=idx_dtype)
    edge_tgt = torch.empty(e_est, dtype=idx_dtype)

    # Backbone: node i → node i + w
    edge_src[:e_backbone] = torch.arange(0, e_backbone, dtype=idx_dtype)
    edge_tgt[:e_backbone] = torch.arange(w, n, dtype=idx_dtype)
    mem("backbone")

    # Cross edges in chunks
    write_pos = e_backbone
    chunk_size = 50_000_000
    for start in range(0, e_backbone, chunk_size):
        end = min(start + chunk_size, e_backbone)
        chunk_n = end - start
        mask = torch.rand(chunk_n) < cross_prob
        n_cross = mask.sum().item()
        if n_cross == 0:
            continue
        cross_src = torch.arange(start, end, dtype=idx_dtype)[mask]
        cross_offset = torch.randint(0, w, (n_cross,), dtype=idx_dtype)
        cross_tgt_layer = cross_src // w + 1
        cross_tgt = cross_tgt_layer * w + cross_offset
        edge_src[write_pos:write_pos + n_cross] = cross_src
        edge_tgt[write_pos:write_pos + n_cross] = cross_tgt
        write_pos += n_cross

    edge_index = torch.stack([edge_src[:write_pos], edge_tgt[:write_pos]])
    del edge_src, edge_tgt
    mem("edges done")
    return edge_index


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    faulthandler.enable(all_threads=True)
    sys.stderr.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Large-scale layout benchmark")
    parser.add_argument(
        "size",
        help="Node count: preset name (50m, 100m, 300m, 1b) or number (e.g. 10_000_000, 5m)",
    )
    parser.add_argument("--layers", type=int, default=0, help="Number of layers (0 = auto)")
    parser.add_argument("--workers", type=int, default=4, help="Num parallel workers")
    parser.add_argument("--steps", type=int, default=500, help="Layout optimization steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Optional checkpoint directory for graph-build artifacts and final positions",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse edge_index/node_sizes from the checkpoint dir when available",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
        help="Target layout device",
    )
    parser.add_argument(
        "--force-duplicate-run",
        action="store_true",
        help="Allow multiple concurrent runs against the same checkpoint root.",
    )
    args = parser.parse_args()

    # Resolve size
    n, layers, w = resolve_size_and_layers(args.size, args.layers)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _default_checkpoint_dir(args.size)
    checkpoint_paths = _checkpoint_paths(checkpoint_dir)
    _guard_duplicate_run(checkpoint_paths, args.size, args.resume, args.force_duplicate_run)
    _register_active_run(checkpoint_paths, args.size, args.resume)

    mem("start")
    print(
        f"Building wide DAG: {n:,} nodes, {layers} layers, ~{w:,} nodes/layer on {args.device}...",
        flush=True,
    )
    t0 = time.perf_counter()
    phase("config resolved", t0)

    node_sizes: torch.Tensor
    restored = _load_graph_checkpoint(checkpoint_paths, n, layers) if args.resume else None
    if restored is not None:
        edge_index, node_sizes = restored
        print(f"Restored graph checkpoint from {checkpoint_dir}", flush=True)
        mem("graph restored")
    else:
        edge_index = build_edges(n, layers)
        node_sizes = torch.full((n, 2), 20.0, dtype=torch.float16)
        print(f"Edge index ready: {edge_index.shape[1]:,} edges in {time.perf_counter() - t0:.1f}s", flush=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(edge_index, checkpoint_paths["edge_index"])
        torch.save(node_sizes, checkpoint_paths["node_sizes"])
        _save_checkpoint_meta(
            checkpoint_paths["meta"],
            {
                "size": args.size,
                "n": n,
                "layers": layers,
                "width": w,
                "device": args.device,
                "seed": args.seed,
            },
        )
        print(f"Saved graph checkpoint to {checkpoint_dir}", flush=True)

    layer_assignments = _load_layer_checkpoint(checkpoint_paths, n, layers) if args.resume else None
    if layer_assignments is not None:
        print(f"Restored layering checkpoint from {checkpoint_paths['layer_assignments']}", flush=True)

    g = dagua.DaguaGraph()
    g.num_nodes = n
    g._edge_index_tensor = edge_index
    # Uniform synthetic nodes don't need float32 precision; keep this compact.
    g.node_sizes = node_sizes
    if layer_assignments is not None:
        g._precomputed_layer_assignments = layer_assignments
    else:
        def _save_layer_assignments(layer_tensor: torch.Tensor) -> None:
            torch.save(layer_tensor, checkpoint_paths["layer_assignments"])
            print(f"Saved layering checkpoint to {checkpoint_paths['layer_assignments']}", flush=True)

        g._layer_assignments_callback = _save_layer_assignments
    mem("graph built")

    config = dagua.LayoutConfig(
        device=args.device,
        verbose=True,
        num_workers=args.workers,
        multilevel_threshold=50000,
        multilevel_min_nodes=2000,
        multilevel_coarse_steps=50,
        multilevel_refine_steps=15,
        steps=args.steps,
        seed=args.seed,
    )
    phase("layout start", t0)

    print(f"\nStarting layout (num_workers={config.num_workers})...", flush=True)
    t1 = time.perf_counter()
    pos = dagua.layout(g, config)
    total = time.perf_counter() - t1
    phase("layout finished", t0)
    torch.save(pos, checkpoint_paths["positions"])
    print(f"Saved positions checkpoint to {checkpoint_paths['positions']}", flush=True)

    print(f"\nResult: {pos.shape}", flush=True)
    print(f"Total layout time: {total:.1f}s", flush=True)
    print(f"  x range: [{pos[:, 0].min():.0f}, {pos[:, 0].max():.0f}]", flush=True)
    print(f"  y range: [{pos[:, 1].min():.0f}, {pos[:, 1].max():.0f}]", flush=True)
    mem("done")


if __name__ == "__main__":
    main()
