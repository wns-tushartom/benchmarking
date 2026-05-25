from __future__ import annotations

import math
import random
from typing import Iterable, Sequence, Tuple


def recall_at_k(hits: Sequence[bool], k: int) -> float:
    return 1.0 if any(hits[:k]) else 0.0


def precision_at_k(hits: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for h in hits[:k] if h) / k


def mrr(hits: Sequence[bool]) -> float:
    for i, hit in enumerate(hits, 1):
        if hit:
            return 1.0 / i
    return 0.0


def dcg_at_k(hits: Sequence[bool], k: int) -> float:
    return sum((1.0 if hit else 0.0) / math.log2(i + 2) for i, hit in enumerate(hits[:k]))


def ndcg_at_k(hits: Sequence[bool], k: int) -> float:
    ideal = sorted(hits, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    return dcg_at_k(hits, k) / ideal_dcg if ideal_dcg else 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_ci(values: Sequence[float], iterations: int = 200, seed: int = 42, alpha: float = 0.05) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(max(1, iterations)):
        draw = [rng.choice(vals) for _ in vals]
        samples.append(mean(draw))
    return percentile(samples, alpha / 2), percentile(samples, 1 - alpha / 2)
