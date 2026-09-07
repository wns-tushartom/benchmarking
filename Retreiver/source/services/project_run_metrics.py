"""Pure metric calculations for one project matrix combination."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class QueryMetricInput:
    """One completed query's relevance ranks and operational latency."""

    query_id: str
    label_applies: bool
    relevant_ranks: tuple[int, ...]
    relevant_corpus_count: int
    retrieval_latency_s: float
    rerank_latency_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be non-empty text")
        if not isinstance(self.label_applies, bool):
            raise ValueError("label_applies must be boolean")
        if not isinstance(self.relevant_ranks, tuple) or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
            for rank in self.relevant_ranks
        ):
            raise ValueError("relevant_ranks must contain positive integers")
        if len(self.relevant_ranks) != len(set(self.relevant_ranks)):
            raise ValueError("relevant_ranks must not contain duplicates")
        if (
            isinstance(self.relevant_corpus_count, bool)
            or not isinstance(self.relevant_corpus_count, int)
            or self.relevant_corpus_count < 0
        ):
            raise ValueError("relevant_corpus_count must be a nonnegative integer")
        if self.relevant_corpus_count < len(self.relevant_ranks):
            raise ValueError("relevant corpus count is inconsistent with relevant ranks")
        if not self.label_applies and (
            self.relevant_ranks or self.relevant_corpus_count != 0
        ):
            raise ValueError("unlabelled queries cannot carry relevance values")
        for value in (self.retrieval_latency_s, self.rerank_latency_s):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError("query latency must be finite and nonnegative")


def _ndcg_at_k(query: QueryMetricInput, k: int) -> float:
    ranks = tuple(rank for rank in query.relevant_ranks if rank <= k)
    dcg = math.fsum(1.0 / math.log2(rank + 1) for rank in ranks)
    ideal_count = min(query.relevant_corpus_count, k)
    if ideal_count == 0:
        return 0.0
    idcg = math.fsum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg


def summarize_query_metrics(
    inputs: Iterable[QueryMetricInput],
    *,
    k: int,
) -> dict[str, int | float | None]:
    """Summarize labelled quality and all-query operational latency."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    queries = tuple(inputs)
    if any(not isinstance(query, QueryMetricInput) for query in queries):
        raise ValueError("inputs must contain QueryMetricInput values")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query_id in metric inputs")

    labelled = tuple(query for query in queries if query.label_applies)
    unlabelled_count = len(queries) - len(labelled)
    average_latency = (
        math.fsum(
            float(query.retrieval_latency_s) + float(query.rerank_latency_s)
            for query in queries
        )
        / len(queries)
        if queries
        else None
    )

    if not labelled:
        recall = mrr = ndcg = None
    else:
        recall = math.fsum(
            1.0 if any(rank <= k for rank in query.relevant_ranks) else 0.0
            for query in labelled
        ) / len(labelled)
        mrr = math.fsum(
            1.0 / min(rank for rank in query.relevant_ranks if rank <= k)
            if any(rank <= k for rank in query.relevant_ranks)
            else 0.0
            for query in labelled
        ) / len(labelled)
        ndcg = math.fsum(_ndcg_at_k(query, k) for query in labelled) / len(labelled)

    return {
        "labelled_queries": len(labelled),
        "unlabelled_queries": unlabelled_count,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,
        "avg_query_latency_s": average_latency,
    }
