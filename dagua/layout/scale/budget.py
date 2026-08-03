"""Cheap memory-budget guards for scale routing stages."""

from __future__ import annotations

import ctypes
import gc
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

DEFAULT_BUDGET_FRACTION = 0.70
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


@dataclass(frozen=True)
class BudgetCheck:
    """Result of a declared peak-byte budget check.

    Parameters
    ----------
    stage : str
        Name of the guarded scale stage.
    declared_peak_bytes : int
        Estimated peak bytes for the stage.
    available_bytes : int
        Available bytes on the relevant device.
    limit_bytes : int
        Budget limit after applying the guard fraction.
    device : str
        Device family checked by the guard.
    """

    stage: str
    declared_peak_bytes: int
    available_bytes: int
    limit_bytes: int
    device: str


class BudgetGuard:
    """Fail-fast peak-memory guard for scale stages."""

    def __init__(self, *, fraction: float = DEFAULT_BUDGET_FRACTION, device: str = "cpu") -> None:
        """Create a memory budget guard.

        Parameters
        ----------
        fraction : float, default=0.70
            Fraction of available memory a declared stage may consume.
        device : str, default="cpu"
            Device family for availability checks. CUDA checks use
            ``torch.cuda.mem_get_info`` when CUDA is available.

        Returns
        -------
        None
            Initializes an empty check ledger.
        """
        self.fraction = float(fraction)
        self.device = str(device)
        self.checks: Dict[str, BudgetCheck] = {}

    def check(self, stage: str, declared_peak_bytes: int) -> BudgetCheck:
        """Abort if a declared stage peak exceeds the available budget.

        Parameters
        ----------
        stage : str
            Name of the guarded stage.
        declared_peak_bytes : int
            Estimated peak bytes for the stage.

        Returns
        -------
        BudgetCheck
            Recorded check details.
        """
        available = available_bytes(self.device)
        limit = int(available * self.fraction)
        check = BudgetCheck(
            stage=str(stage),
            declared_peak_bytes=int(declared_peak_bytes),
            available_bytes=int(available),
            limit_bytes=limit,
            device=self.device,
        )
        self.checks[str(stage)] = check
        if int(declared_peak_bytes) > limit:
            raise MemoryError(
                f"{stage} declared peak {int(declared_peak_bytes):,} bytes exceeds "
                f"{self.fraction:.0%} budget {limit:,} bytes on {self.device}"
            )
        return check

    def verify_release(self, before_rss_bytes: int, *, min_release_bytes: int = 0) -> int:
        """Collect and verify process RSS after a scale-stage release.

        Parameters
        ----------
        before_rss_bytes : int
            RSS measured before releasing memory.
        min_release_bytes : int, default=0
            Expected minimum RSS decrease. Use ``0`` to only force collection
            and return the observed delta.

        Returns
        -------
        int
            Observed RSS decrease in bytes. Negative means RSS increased.
        """
        gc.collect()
        _malloc_trim()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = rss_bytes()
        released = int(before_rss_bytes) - int(after)
        if min_release_bytes > 0 and released < int(min_release_bytes):
            raise MemoryError(
                f"release verification expected {int(min_release_bytes):,} bytes, "
                f"observed {released:,} bytes"
            )
        return released


def available_bytes(device: str = "cpu") -> int:
    """Return currently available RAM or VRAM bytes.

    Parameters
    ----------
    device : str, default="cpu"
        Device family. CUDA returns free VRAM when available.

    Returns
    -------
    int
        Available bytes.
    """
    if str(device) == "cuda" and torch.cuda.is_available():
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes)
    meminfo = _read_mem_available()
    if meminfo is not None:
        return meminfo
    return max(1, os.sysconf("SC_PHYS_PAGES") * _PAGE_SIZE)


def rss_bytes() -> int:
    """Return current resident set size in bytes.

    Returns
    -------
    int
        RSS for the current process.
    """
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return pages * _PAGE_SIZE
    except (FileNotFoundError, IndexError, ValueError):
        return 0


def _read_mem_available() -> Optional[int]:
    """Read Linux ``MemAvailable`` from ``/proc/meminfo``.

    Returns
    -------
    int or None
        Available RAM in bytes when the proc file is present.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        return None
    return None


def _malloc_trim() -> None:
    """Ask glibc to return free arenas to the operating system.

    Returns
    -------
    None
        Best-effort trim; unsupported platforms are ignored.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError:
        return
