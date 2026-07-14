from __future__ import annotations

from importlib import import_module, util
import math

import pytest

from benchmarking.core.schemas import Chunk
from source.services.project_questions import ChunkRef, ProjectQuestion, RetrievalLabels


def _metrics_module():
    spec = util.find_spec("source.services.project_run_metrics")
    assert spec is not None, "project_run_metrics module must exist"
    return import_module("source.services.project_run_metrics")


def _chunk(chunk_id: int, text: str, source_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        pdf_name="policy.txt",
        paragraph=text,
        parent_id=source_id,
        metadata={"source_id": source_id},
    )


def test_summary_uses_labelled_denominator_and_real_rank_positions():
    metrics = _metrics_module()
    first = metrics.QueryMetricInput(
        query_id="q_000001",
        label_applies=True,
        relevant_ranks=(2,),
        relevant_corpus_count=2,
        retrieval_latency_s=0.1,
        rerank_latency_s=0.2,
    )
    missed = metrics.QueryMetricInput(
        query_id="q_000002",
        label_applies=True,
        relevant_ranks=(),
        relevant_corpus_count=1,
        retrieval_latency_s=0.2,
        rerank_latency_s=0.1,
    )

    result = metrics.summarize_query_metrics([first, missed], k=3)

    assert result["labelled_queries"] == 2
    assert result["unlabelled_queries"] == 0
    assert result["recall_at_k"] == 0.5
    assert result["mrr_at_k"] == 0.25
    assert math.isclose(result["ndcg_at_k"], 0.19342640361727081)
    assert math.isclose(result["avg_query_latency_s"], 0.3)


def test_evidence_only_summary_keeps_latency_without_quality_scores():
    metrics = _metrics_module()
    operational = metrics.QueryMetricInput(
        query_id="q_000001",
        label_applies=False,
        relevant_ranks=(),
        relevant_corpus_count=0,
        retrieval_latency_s=0.2,
        rerank_latency_s=0.3,
    )

    result = metrics.summarize_query_metrics([operational], k=5)

    assert result == {
        "labelled_queries": 0,
        "unlabelled_queries": 1,
        "recall_at_k": None,
        "mrr_at_k": None,
        "ndcg_at_k": None,
        "avg_query_latency_s": 0.5,
    }


def test_labelled_miss_is_zero_not_missing_and_invalid_k_fails_closed():
    metrics = _metrics_module()
    missed = metrics.QueryMetricInput(
        query_id="q_000001",
        label_applies=True,
        relevant_ranks=(),
        relevant_corpus_count=2,
        retrieval_latency_s=0.0,
        rerank_latency_s=0.0,
    )
    result = metrics.summarize_query_metrics([missed], k=5)
    assert result["recall_at_k"] == 0.0
    assert result["mrr_at_k"] == 0.0
    assert result["ndcg_at_k"] == 0.0

    for invalid_k in (0, -1, True):
        with pytest.raises(ValueError):
            metrics.summarize_query_metrics([missed], k=invalid_k)


def test_metric_inputs_reject_inconsistent_relevance_and_duplicate_queries():
    metrics = _metrics_module()
    with pytest.raises(ValueError, match="corpus"):
        metrics.QueryMetricInput(
            query_id="q_000001",
            label_applies=True,
            relevant_ranks=(1, 2),
            relevant_corpus_count=1,
            retrieval_latency_s=0.1,
            rerank_latency_s=0.1,
        )
    with pytest.raises(ValueError, match="query_id"):
        metrics.QueryMetricInput(
            query_id="",
            label_applies=False,
            relevant_ranks=(),
            relevant_corpus_count=0,
            retrieval_latency_s=0.1,
            rerank_latency_s=0.1,
        )

    row = metrics.QueryMetricInput(
        query_id="q_000001",
        label_applies=True,
        relevant_ranks=(1,),
        relevant_corpus_count=1,
        retrieval_latency_s=0.1,
        rerank_latency_s=0.1,
    )
    with pytest.raises(ValueError, match="duplicate query_id"):
        metrics.summarize_query_metrics([row, row], k=5)


def test_relevant_corpus_count_reuses_exact_project_relevance_rules():
    relevance = import_module("source.services.project_relevance")
    assert hasattr(relevance, "chunk_is_relevant")
    assert hasattr(relevance, "relevant_corpus_count")

    source_a = "source_" + "a" * 64
    source_b = "source_" + "b" * 64
    chunks = [
        _chunk(1, "refunds are available within thirty days", source_a),
        _chunk(2, "contact support for refund approval", source_a),
        _chunk(3, "unrelated warranty details", source_b),
    ]
    source_question = ProjectQuestion(
        question_id="q_000001",
        query="What is the refund rule?",
        labels=RetrievalLabels(source_ids=(source_a,)),
    )
    ref_question = ProjectQuestion(
        question_id="q_000002",
        query="Which chunk has approval instructions?",
        labels=RetrievalLabels(
            chunk_refs=(ChunkRef(chunker="fixed_tok1200_ov150", chunk_id="2"),)
        ),
    )

    assert relevance.relevant_corpus_count(
        source_question, chunks, "Heading_sections_l2"
    ) == 2
    assert relevance.relevant_corpus_count(
        ref_question, chunks, "fixed_tok1200_ov150"
    ) == 1
    assert relevance.relevant_corpus_count(
        ref_question, chunks, "Heading_sections_l2"
    ) == 0
    assert relevance.chunk_is_relevant(
        source_question, chunks[0], "Heading_sections_l2"
    )
