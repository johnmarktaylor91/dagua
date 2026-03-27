#!/usr/bin/env python3
"""Consolidate individual .pt position tensors into a single HDF5 file.

One-time cost (~80 min for 400K files). Subsequent analysis loads go from
~75 min to ~45 seconds.

Usage:
    python scripts/consolidate_positions_hdf5.py \
        --input eval_output/variant_bench_full \
        --output eval_output/variant_bench_full/positions.h5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate .pt files into HDF5")
    parser.add_argument("--input", type=str, required=True, help="Benchmark output dir")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    results_path = input_dir / "results.json"
    output_path = Path(args.output)

    print(f"Loading results from {results_path}...", file=sys.stderr)
    results = json.loads(results_path.read_text())

    ok_records = [
        (key, record)
        for key, record in results.items()
        if record.get("status") == "ok" and record.get("positions_file")
    ]
    print(f"Found {len(ok_records)} OK records with positions", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    loaded = 0
    skipped = 0

    with h5py.File(str(output_path), "w") as h5:
        for key, record in ok_records:
            pos_path = input_dir / record["positions_file"]
            try:
                tensor = torch.load(pos_path, map_location="cpu")
                if not isinstance(tensor, torch.Tensor):
                    skipped += 1
                    continue
                arr = tensor.detach().float().numpy()
                h5.create_dataset(key, data=arr, compression="gzip", compression_opts=1)
                loaded += 1
            except Exception:
                skipped += 1

            if loaded % 1000 == 0 and loaded > 0:
                elapsed = time.perf_counter() - start
                rate = loaded / elapsed
                eta = (len(ok_records) - loaded) / rate if rate > 0 else 0
                print(
                    f"[consolidate] {loaded}/{len(ok_records)} "
                    f"({100 * loaded / len(ok_records):.1f}%) "
                    f"ETA {eta / 60:.1f} min",
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - start
    print(
        f"Done: {loaded} tensors consolidated, {skipped} skipped, "
        f"{elapsed / 60:.1f} min. Output: {output_path} "
        f"({output_path.stat().st_size / 1024 / 1024:.1f} MB)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
