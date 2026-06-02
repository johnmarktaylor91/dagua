"""igraph external RNG helpers used by fidelity pipeline ports."""

from __future__ import annotations

import random


class IgraphPythonRNG:
    """Replicate python-igraph's external RNG bridge over ``random.Random``.

    Parameters
    ----------
    seed : int
        Seed passed to Python's Mersenne Twister implementation.
    """

    def __init__(self, seed: int) -> None:
        """Initialize the bridged Python RNG.

        Parameters
        ----------
        seed : int
            Seed used for the underlying ``random.Random`` instance.
        """
        self._rng = random.Random(seed)

    def integer(self, low: int, high: int) -> int:
        """Return igraph's inclusive bounded integer draw.

        Parameters
        ----------
        low : int
            Inclusive lower bound.
        high : int
            Inclusive upper bound.

        Returns
        -------
        int
            Uniform integer in ``[low, high]`` generated through igraph's
            32-bit Lemire bounded-integer path.
        """
        if high <= low:
            return low
        return low + self._uint32_bounded(high - low + 1)

    def random(self) -> float:
        """Return python-igraph's external ``RNG_UNIF01`` draw.

        Returns
        -------
        float
            A Python ``random()`` value in ``[0, 1)``.
        """
        return self._rng.random()

    def shuffle(self, values: list[int]) -> None:
        """Shuffle values in place with igraph's Fisher-Yates loop.

        Parameters
        ----------
        values : list[int]
            Mutable vector to shuffle.

        Returns
        -------
        None
            The input list is modified in place.
        """
        size = len(values)
        while size > 1:
            swap_index = self.integer(0, size - 1)
            size -= 1
            values[size], values[swap_index] = values[swap_index], values[size]

    def _uint32_bounded(self, range_value: int) -> int:
        """Generate igraph's bounded 32-bit integer.

        Parameters
        ----------
        range_value : int
            Exclusive upper bound for the generated integer.

        Returns
        -------
        int
            Uniform integer in ``[0, range_value)``.
        """
        threshold = ((1 << 32) - range_value) % range_value
        while True:
            value = self._rng.getrandbits(32)
            product = value * range_value
            low_word = product & 0xFFFFFFFF
            if low_word >= threshold:
                return product >> 32


__all__ = ["IgraphPythonRNG"]
