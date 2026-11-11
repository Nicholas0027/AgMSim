"""Named, independent random streams.

Reproducible comparisons need more than one global seed: if switching the
vehicles' decision model consumed random numbers from the same generator that
draws travel demand, the two experimental arms would face different demand and
the comparison would be confounded. AgMSim therefore derives one independent
stream per *purpose* (``demand``, ``fleet``, one per agent, ...)
from the run seed. Arms that share a seed share demand, personas and car
ownership exactly (common random numbers), whatever the policies do.

Only the standard library generator is used; its core methods are stable
across Python versions for a given seed.
"""
from __future__ import annotations

import hashlib
import math
import random


class RngStreams:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._streams: dict[str, random.Random] = {}

    def get(self, *name: object) -> random.Random:
        """Return the stream called ``name`` (created on first use)."""
        key = "/".join(str(part) for part in name)
        stream = self._streams.get(key)
        if stream is None:
            digest = hashlib.sha256(f"{self.seed}|{key}".encode()).digest()
            stream = random.Random(int.from_bytes(digest[:8], "big"))
            self._streams[key] = stream
        return stream


def lognormal(rng: random.Random, median: float, sigma: float) -> float:
    """Sample a log-normal variate parameterised by its median."""
    return median * math.exp(sigma * rng.gauss(0.0, 1.0))


def weighted_choice(rng: random.Random, items: list, weights: list[float]):
    """Pick one item with probability proportional to its weight."""
    total = sum(weights)
    if total <= 0:
        return items[rng.randrange(len(items))]
    x = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if x < acc:
            return item
    return items[-1]
