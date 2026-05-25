"""igraph-compatible random number generators for layout fidelity modes."""

from __future__ import annotations

from typing import Optional

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF
_DOUBLE_MANTISSA_BITS = 52
_DOUBLE_MANTISSA_SCALE = float(1 << _DOUBLE_MANTISSA_BITS)


def _rotate_right_u32(value: int, rotation: int) -> int:
    """Rotate a 32-bit integer right.

    Parameters
    ----------
    value : int
        Unsigned 32-bit value to rotate.
    rotation : int
        Rotation count. Only the low five bits are used.

    Returns
    -------
    int
        Rotated unsigned 32-bit value.
    """
    rotation &= 31
    value &= _MASK32
    return ((value >> rotation) | (value << ((-rotation) & 31))) & _MASK32


def _advance_lcg_u64(state: int, delta: int, multiplier: int, increment: int) -> int:
    """Advance a 64-bit LCG by ``delta`` steps.

    Parameters
    ----------
    state : int
        Current unsigned 64-bit state.
    delta : int
        Number of steps to advance.
    multiplier : int
        LCG multiplier.
    increment : int
        LCG increment.

    Returns
    -------
    int
        Advanced unsigned 64-bit state.
    """
    accumulated_multiplier = 1
    accumulated_increment = 0
    current_multiplier = multiplier & _MASK64
    current_increment = increment & _MASK64
    remaining = int(delta) & _MASK64

    while remaining > 0:
        if remaining & 1:
            accumulated_multiplier = (accumulated_multiplier * current_multiplier) & _MASK64
            accumulated_increment = (
                accumulated_increment * current_multiplier + current_increment
            ) & _MASK64
        current_increment = ((current_multiplier + 1) * current_increment) & _MASK64
        current_multiplier = (current_multiplier * current_multiplier) & _MASK64
        remaining >>= 1

    return (accumulated_multiplier * state + accumulated_increment) & _MASK64


class _IgraphRandom32Mixin:
    """Shared igraph extraction helpers for 32-bit generators."""

    def get_u32(self) -> int:
        """Return the next raw 32-bit word.

        Returns
        -------
        int
            Unsigned integer in ``[0, 2**32)``.
        """
        raise NotImplementedError

    def random_bits(self, bits: int) -> int:
        """Return igraph's high-bit-first random bit extraction.

        Parameters
        ----------
        bits : int
            Number of random bits to return.

        Returns
        -------
        int
            Random bits packed into the low bits of the return value.
        """
        if bits < 0:
            raise ValueError("bits must be non-negative")
        if bits == 0:
            return 0
        if bits <= 32:
            return self.get_u32() >> (32 - bits)

        result = 0
        remaining = bits
        while remaining > 32:
            result = (result << 32) + self.get_u32()
            remaining -= 32
        return (result << remaining) + (self.get_u32() >> (32 - remaining))

    def random_bits_uint64(self, bits: int) -> int:
        """Return up to 64 random bits using igraph's packing order.

        Parameters
        ----------
        bits : int
            Number of random bits to return.

        Returns
        -------
        int
            Random bits packed into the low bits of the return value.
        """
        if bits > 64:
            raise ValueError("bits must be at most 64")
        return self.random_bits(bits)

    def random(self) -> float:
        """Draw an igraph ``RNG_UNIF01`` double.

        Returns
        -------
        float
            Uniform value in the half-open interval ``[0, 1)``.
        """
        return self.random_bits_uint64(_DOUBLE_MANTISSA_BITS) / _DOUBLE_MANTISSA_SCALE

    def uniform(self, low: float, high: float) -> float:
        """Draw an igraph ``RNG_UNIF(low, high)`` value.

        Parameters
        ----------
        low : float
            Inclusive lower bound.
        high : float
            Exclusive upper bound unless ``low == high``.

        Returns
        -------
        float
            Uniform floating-point value.
        """
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        if low == high:
            return high

        while True:
            value = self.random() * (high - low) + low
            if value != high:
                return value

    def _bounded_u32(self, range_size: int) -> int:
        """Draw an unbiased bounded unsigned 32-bit integer.

        Parameters
        ----------
        range_size : int
            Exclusive upper bound in ``(0, 2**32]``.

        Returns
        -------
        int
            Integer in ``[0, range_size)``.
        """
        if range_size <= 0 or range_size > (1 << 32):
            raise ValueError("range_size must be in (0, 2**32]")
        threshold = ((-range_size) & _MASK32) % range_size
        while True:
            value = self.random_bits(32)
            product = value * range_size
            low = product & _MASK32
            if low >= threshold:
                return (product >> 32) & _MASK32

    def _bounded_u64(self, range_size: int) -> int:
        """Draw an unbiased bounded unsigned 64-bit integer.

        Parameters
        ----------
        range_size : int
            Exclusive upper bound in ``(0, 2**64]``.

        Returns
        -------
        int
            Integer in ``[0, range_size)``.
        """
        if range_size <= 0 or range_size > (1 << 64):
            raise ValueError("range_size must be in (0, 2**64]")
        threshold = ((-range_size) & _MASK64) % range_size
        while True:
            value = self.random_bits_uint64(64)
            product = value * range_size
            low = product & _MASK64
            if low >= threshold:
                return (product >> 64) & _MASK64

    def randint(self, low: int, high: int) -> int:
        """Draw an igraph inclusive bounded integer.

        Parameters
        ----------
        low : int
            Inclusive lower bound.
        high : int
            Inclusive upper bound.

        Returns
        -------
        int
            Integer in ``[low, high]``.
        """
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        if high == low:
            return low

        range_size = high - low + 1
        if range_size <= (1 << 32):
            return low + self._bounded_u32(range_size)
        return low + self._bounded_u64(range_size)

    def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        """Draw an integer from a Python-style half-open range.

        Parameters
        ----------
        start : int
            Start value, or exclusive stop when ``stop`` is omitted.
        stop : int, optional
            Exclusive stop value.
        step : int, default=1
            Range step. Only positive steps are supported.

        Returns
        -------
        int
            Integer selected from ``range(start, stop, step)``.
        """
        if step <= 0:
            raise ValueError("step must be positive")
        if stop is None:
            low = 0
            high = start
        else:
            low = start
            high = stop
        width = high - low
        if width <= 0:
            raise ValueError("empty range for randrange")
        if step == 1:
            return self.randint(low, high - 1)

        count = (width + step - 1) // step
        return low + step * self.randint(0, count - 1)


class IgraphPCG32(_IgraphRandom32Mixin):
    """Pure-Python port of igraph's current default PCG32 generator."""

    _MULTIPLIER = 6364136223846793005
    _INITIAL_STATE = 0x853C49E6748FEA9B
    _INITIAL_INCREMENT = 0xDA3E39CB94B95BDB
    _DEFAULT_SEQUENCE = _INITIAL_INCREMENT >> 1

    def __init__(self, seed: int = 0) -> None:
        """Create a seeded PCG32 stream.

        Parameters
        ----------
        seed : int, default=0
            igraph seed value. Zero selects PCG's compiled default sequence.
        """
        self.state = 0
        self.increment = 0
        self.seed(seed)

    def seed(self, seed: int) -> None:
        """Seed the generator with igraph's PCG32 convention.

        Parameters
        ----------
        seed : int
            igraph seed value.

        Returns
        -------
        None
            The generator state is reset in place.
        """
        init_sequence = self._DEFAULT_SEQUENCE if seed == 0 else int(seed)
        self.state = 0
        self.increment = ((init_sequence & _MASK64) << 1 | 1) & _MASK64
        self._step()
        self.state = (self.state + self._INITIAL_STATE) & _MASK64
        self._step()

    def _step(self) -> None:
        """Advance the underlying PCG LCG by one step.

        Returns
        -------
        None
            The internal state is advanced in place.
        """
        self.state = (self.state * self._MULTIPLIER + self.increment) & _MASK64

    def advance(self, delta: int) -> None:
        """Advance the generator by ``delta`` raw draws.

        Parameters
        ----------
        delta : int
            Number of raw PCG steps to skip.

        Returns
        -------
        None
            The internal state is advanced in place.
        """
        if delta < 0:
            raise ValueError("delta must be non-negative")
        self.state = _advance_lcg_u64(self.state, delta, self._MULTIPLIER, self.increment)

    def get_u32(self) -> int:
        """Return the next raw PCG32 output word.

        Returns
        -------
        int
            Unsigned integer in ``[0, 2**32)``.
        """
        old_state = self.state
        self._step()
        xorshifted = (((old_state >> 18) ^ old_state) >> 27) & _MASK32
        rotation = old_state >> 59
        return _rotate_right_u32(xorshifted, rotation)


class IgraphMT19937(_IgraphRandom32Mixin):
    """Pure-Python port of igraph's legacy MT19937 generator."""

    _N = 624
    _M = 397
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF
    _MATRIX_A = 0x9908B0DF

    def __init__(self, seed: int = 0) -> None:
        """Create a seeded MT19937 stream.

        Parameters
        ----------
        seed : int, default=0
            igraph seed value. Zero maps to igraph's legacy default seed 4357.
        """
        self.mt = [0] * self._N
        self.index = self._N
        self.seed(seed)

    def seed(self, seed: int) -> None:
        """Seed the MT19937 state.

        Parameters
        ----------
        seed : int
            igraph seed value.

        Returns
        -------
        None
            The generator state is reset in place.
        """
        resolved_seed = 4357 if seed == 0 else int(seed)
        self.mt = [0] * self._N
        self.mt[0] = resolved_seed & _MASK32
        for index in range(1, self._N):
            previous = self.mt[index - 1]
            self.mt[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & _MASK32
        self.index = self._N

    def _twist(self) -> None:
        """Regenerate the MT19937 state array.

        Returns
        -------
        None
            The internal state array is updated in place.
        """
        for index in range(self._N - self._M):
            value = (self.mt[index] & self._UPPER_MASK) | (self.mt[index + 1] & self._LOWER_MASK)
            magic = self._MATRIX_A if value & 1 else 0
            self.mt[index] = (self.mt[index + self._M] ^ (value >> 1) ^ magic) & _MASK32
        for index in range(self._N - self._M, self._N - 1):
            value = (self.mt[index] & self._UPPER_MASK) | (self.mt[index + 1] & self._LOWER_MASK)
            magic = self._MATRIX_A if value & 1 else 0
            self.mt[index] = (self.mt[index + (self._M - self._N)] ^ (value >> 1) ^ magic) & _MASK32
        value = (self.mt[self._N - 1] & self._UPPER_MASK) | (self.mt[0] & self._LOWER_MASK)
        magic = self._MATRIX_A if value & 1 else 0
        self.mt[self._N - 1] = (self.mt[self._M - 1] ^ (value >> 1) ^ magic) & _MASK32
        self.index = 0

    def advance(self, delta: int) -> None:
        """Advance the generator by ``delta`` raw draws.

        Parameters
        ----------
        delta : int
            Number of raw MT19937 words to skip.

        Returns
        -------
        None
            The internal state is advanced in place.
        """
        if delta < 0:
            raise ValueError("delta must be non-negative")
        for _ in range(delta):
            self.get_u32()

    def get_u32(self) -> int:
        """Return the next raw tempered MT19937 output word.

        Returns
        -------
        int
            Unsigned integer in ``[0, 2**32)``.
        """
        if self.index >= self._N:
            self._twist()

        value = self.mt[self.index]
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        self.index += 1
        return value & _MASK32


def make_igraph_default_rng(seed: int) -> IgraphPCG32:
    """Create the RNG used by this igraph checkout's layout call paths.

    Parameters
    ----------
    seed : int
        igraph seed value.

    Returns
    -------
    IgraphPCG32
        Seeded igraph default RNG stream.
    """
    return IgraphPCG32(seed=seed)


__all__ = ["IgraphMT19937", "IgraphPCG32", "make_igraph_default_rng"]
