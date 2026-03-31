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
import os
import sys
import time
from pathlib import Path

import h5py
import torch


def write_hdf5_atomic(
    output_path: Path,
    ok_records: list[tuple[str, dict[str, object]]],
    input_dir: Path,
    start_time: float,
) -> tuple[int, int]:
    """Write consolidated position tensors via a temporary HDF5 file.

    Parameters
    ----------
    output_path : Path
        Final HDF5 destination.
    ok_records : list[tuple[str, dict[str, object]]]
        Successful benchmark records keyed by ``results.json`` key.
    input_dir : Path
        Benchmark artifact root that contains the ``.pt`` tensors.
    start_time : float
        Monotonic timestamp used for progress-rate reporting.

    Returns
    -------
    tuple[int, int]
        Counts of ``(loaded, skipped)`` tensors.
    """
    temp_path = output_path.with_suffix(".h5.tmp")
    temp_path.unlink(missing_ok=True)

    loaded = 0
    skipped = 0
    with h5py.File(str(temp_path), "w") as h5:
        for key, record in ok_records:
            pos_path = input_dir / str(record["positions_file"])
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
                elapsed = time.perf_counter() - start_time
                rate = loaded / elapsed
                eta = (len(ok_records) - loaded) / rate if rate > 0 else 0
                print(
                    f"[consolidate] {loaded}/{len(ok_records)} "
                    f"({100 * loaded / len(ok_records):.1f}%) "
                    f"ETA {eta / 60:.1f} min",
                    file=sys.stderr,
                )

    os.rename(temp_path, output_path)
    return loaded, skipped


def main() -> int:
    """Consolidate benchmark position tensors into one HDF5 cache."""
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
    loaded, skipped = write_hdf5_atomic(
        output_path=output_path,
        ok_records=ok_records,
        input_dir=input_dir,
        start_time=start,
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
