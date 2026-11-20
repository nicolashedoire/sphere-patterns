"""Fuse signed evidence for two hypotheses as observations arrive."""

from math import isfinite
from typing import NamedTuple


class Decision(NamedTuple):
    early: int
    final: int
    revised: bool
    reads: int
    final_margin: float


def _label(margin: float, threshold: float) -> int:
    if margin == 0.0 or abs(margin) < threshold:
        return 0
    return 1 if margin > 0.0 else -1


def decide(
    first: float,
    second: float | None = None,
    *,
    first_threshold: float,
    final_threshold: float,
    second_weight: float = 1.0,
    second_ready: bool = True,
) -> Decision:
    """Return early and eventual labels (+1 first hypothesis, -1 second, 0 abstain).

    Ignore a second observation withheld by the observation cutoff or lacking a
    usable measurement. A ready measurement adds signed evidence to the first.
    A revision changes or retracts an early commitment. Resolving an early
    abstention is tracked by the labels and does not count as a revision.
    Reads count usable observations.
    """
    policy = (first_threshold, final_threshold, second_weight)
    if any(not isfinite(value) or value < 0.0 for value in policy):
        raise ValueError("Decision thresholds and weight must be finite and nonnegative")
    if not isfinite(first):
        raise ValueError("Initial signed evidence must be finite")

    early = _label(first, first_threshold)
    if not second_ready or second is None or not isfinite(second):
        return Decision(early, early, False, 1, first)

    margin = first + second_weight * second
    final = _label(margin, final_threshold)
    return Decision(early, final, early != 0 and final != early, 2, margin)
