"""Statistical analysis helpers: percentiles, ABBA pairing, bootstrap 95% CI, early stop."""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))
    return float(ordered[idx])


def summarize_samples(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stdev": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20261123,
    statistic: str = "median",
) -> dict[str, float | bool]:
    """Paired bootstrap confidence interval on B - A deltas.

    Matches the ABBA protocol: positive delta means B > A; excludes_zero is true
    when ci_low > 0 or ci_high < 0.
    """
    if not deltas:
        return {
            "n": 0,
            "estimate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "half_width": 0.0,
            "excludes_zero": False,
        }
    n = len(deltas)
    stat_fn = statistics.median if statistic == "median" else statistics.fmean
    point = float(stat_fn(deltas))
    if n == 1:
        return {
            "n": 1,
            "estimate": point,
            "ci_low": point,
            "ci_high": point,
            "half_width": 0.0,
            "excludes_zero": False,
        }
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot.append(float(stat_fn(sample)))
    boot.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(math.floor((alpha / 2.0) * (n_resamples - 1))))
    hi_idx = min(n_resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * (n_resamples - 1))))
    ci_low = boot[lo_idx]
    ci_high = boot[hi_idx]
    return {
        "n": n,
        "estimate": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "half_width": 0.5 * (ci_high - ci_low),
        "excludes_zero": (ci_low > 0.0) or (ci_high < 0.0),
    }


def pair_abba_blocks(
    arm_a_blocks: Sequence[float],
    arm_b_blocks: Sequence[float],
) -> list[float]:
    """Compute paired (B - A) deltas across ABBA rounds.

    Input streams may be interleaved [A0, A1, ...] and [B0, B1, ...] or already
    grouped per block; pairs block i of B minus block i of A.
    """
    count = min(len(arm_a_blocks), len(arm_b_blocks))
    return [float(arm_b_blocks[i]) - float(arm_a_blocks[i]) for i in range(count)]


def should_stop_early(
    deltas: Sequence[float],
    *,
    min_pairs: int = 4,
    target_half_width: float = 0.15,
    n_resamples: int = 1000,
) -> tuple[bool, dict[str, float | bool]]:
    """Sequential stopping rule: stop once CI excludes 0 or half_width <= target."""
    ci = bootstrap_ci(deltas, n_resamples=n_resamples)
    if len(deltas) < min_pairs:
        return False, ci
    decisive = bool(ci["excludes_zero"]) or (
        float(ci["half_width"]) <= target_half_width
        and float(ci["ci_low"]) <= 0.0 <= float(ci["ci_high"])
    )
    return decisive, ci


def filter_aa_spread_ok(
    delta: float,
    aa_spread: float,
    multiplier: float = 2.0,
) -> bool:
    """Confirm observed delta clears the A/A noise floor (2x-spread guard)."""
    return abs(delta) >= multiplier * abs(aa_spread)


def trim_transient(values: Iterable[float], drop_first: int = 1) -> list[float]:
    items = list(values)
    return items[drop_first:] if len(items) > drop_first else items
