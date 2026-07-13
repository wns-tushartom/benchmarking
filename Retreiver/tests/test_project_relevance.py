from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmarking.core.schemas import Chunk, SearchHit
from source.services.project_questions import (
    ChunkRef,
    ProjectQuestion,
    RetrievalLabels,
    load_normalized_questions,
    parse_question_bytes,
    parse_typed_question,
)
from source.services import project_relevance
from source.services.project_relevance import (
    ProjectRelevanceError,
    hit_is_relevant,
    label_applies,
    load_pinned_stopwords,
    metric_denominator,
    wns_context_tokens_v1,
)


def _hit(
    text: str,
    *,
    source_id: str = "source_" + "0" * 64,
    chunk_id: int = 1,
) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=chunk_id,
            pdf_name="policy.txt",
            paragraph=text,
            metadata={"source_id": source_id},
        ),
        score=0.9,
    )


def _question(labels: RetrievalLabels, answer: str | None = None) -> ProjectQuestion:
    return ProjectQuestion(
        question_id="q_000001",
        query="What is the refund policy?",
        labels=labels,
        reference_answer=answer,
        source_row=2,
    )


def test_answer_only_questions_never_enter_retrieval_denominator():
    questions = parse_question_bytes(
        "questions.csv",
        b"question,ground_truth\nrefund policy,ask support\n",
    )

    assert questions[0].reference_answer == "ask support"
    assert not label_applies(questions[0], "fixed_tok1200_ov150")
    assert metric_denominator(questions, "fixed_tok1200_ov150") == 0


def test_source_and_chunk_labels_match_exactly_and_chunk_refs_are_chunker_specific():
    source_id = "source_" + "a" * 64
    question = _question(
        RetrievalLabels(
            source_ids=(source_id,),
            chunk_refs=(
                ChunkRef(chunker="fixed_tok1200_ov150", chunk_id="42"),
            ),
        )
    )

    assert label_applies(question, "fixed_tok1200_ov150")
    assert label_applies(question, "Heading_sections_l2")  # source_id applies to every chunker
    assert hit_is_relevant(question, _hit("unrelated", source_id=source_id), "Heading_sections_l2")
    assert hit_is_relevant(question, _hit("unrelated", chunk_id=42), "fixed_tok1200_ov150")
    assert not hit_is_relevant(question, _hit("unrelated", chunk_id=42), "Heading_sections_l2")
    assert not hit_is_relevant(question, _hit("unrelated", chunk_id=7), "fixed_tok1200_ov150")


def test_every_relevance_scoring_operation_verifies_stopword_pin(
    monkeypatch,
    tmp_path: Path,
):
    source_id = "source_" + "a" * 64
    question = _question(RetrievalLabels(source_ids=(source_id,)))
    matching = _hit("unrelated", source_id=source_id)

    def fail_closed(*_args, **_kwargs):
        raise ProjectRelevanceError("stopword hash mismatch")

    monkeypatch.setattr(project_relevance, "load_pinned_stopwords", fail_closed)
    with pytest.raises(ProjectRelevanceError, match="hash"):
        parse_question_bytes("questions.txt", b"Refund policy?\n")
    with pytest.raises(ProjectRelevanceError, match="hash"):
        parse_typed_question("Refund policy?")
    with pytest.raises(ProjectRelevanceError, match="hash"):
        load_normalized_questions(tmp_path / "questions.jsonl")
    with pytest.raises(ProjectRelevanceError, match="hash"):
        hit_is_relevant(question, matching, "fixed_tok1200_ov150")
    with pytest.raises(ProjectRelevanceError, match="hash"):
        metric_denominator([question], "fixed_tok1200_ov150")


def test_chunk_only_labels_change_denominator_by_combination():
    question = _question(
        RetrievalLabels(
            chunk_refs=(
                ChunkRef(chunker="fixed_tok1200_ov150", chunk_id="42"),
            )
        )
    )

    assert metric_denominator([question], "fixed_tok1200_ov150") == 1
    assert metric_denominator([question], "Heading_sections_l2") == 0


def test_context_relevance_uses_normalization_substring_or_eighty_percent_set_recall():
    substring_question = _question(
        RetrievalLabels(reference_contexts=("Refunds   are ACCEPTED within thirty days",))
    )
    assert hit_is_relevant(
        substring_question,
        _hit("Policy: refunds are accepted within thirty days for eligible bookings."),
        "entity_heuristic_w4",
    )

    recall_question = _question(
        RetrievalLabels(reference_contexts=("refund cancellation booking support policy",))
    )
    assert hit_is_relevant(
        recall_question,
        _hit("Booking cancellation support follows the refund rules."),
        "entity_heuristic_w4",
    )
    assert not hit_is_relevant(
        recall_question,
        _hit("Booking support is available."),
        "entity_heuristic_w4",
    )


def test_context_with_fewer_than_three_normalized_tokens_is_not_applicable():
    question = _question(RetrievalLabels(reference_contexts=("the refund",)))
    assert not label_applies(question, "entity_heuristic_w4")
    assert metric_denominator([question], "entity_heuristic_w4") == 0


def test_tokenizer_is_nfkc_casefold_ascii_set_based_and_stopword_pinned():
    tokens = wns_context_tokens_v1("ＴＨＥ Refund refund POLICY and 30 Days 政策")
    assert tokens == frozenset({"refund", "policy", "30", "days"})

    stopwords, digest = load_pinned_stopwords()
    assert "the" in stopwords
    assert len(digest) == 64


def test_stopword_pin_is_reverified_after_prior_success(tmp_path: Path):
    stopwords = tmp_path / "stopwords.txt"
    pin = tmp_path / "stopwords.sha256"
    original = b"the\nand\n"
    stopwords.write_bytes(original)
    pin.write_text(hashlib.sha256(original).hexdigest() + "\n", encoding="utf-8")
    assert load_pinned_stopwords(stopwords, pin)[0] == frozenset({"the", "and"})

    stopwords.write_text("the\nand\nchanged\n", encoding="utf-8")
    with pytest.raises(ProjectRelevanceError, match="hash"):
        load_pinned_stopwords(stopwords, pin)


def test_stopword_loader_rejects_symlinked_parent_directory(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    content = b"the\nand\n"
    (real / "stopwords.txt").write_bytes(content)
    (real / "stopwords.sha256").write_text(
        hashlib.sha256(content).hexdigest() + "\n",
        encoding="utf-8",
    )
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ProjectRelevanceError, match="symlink"):
        load_pinned_stopwords(
            alias / "stopwords.txt",
            alias / "stopwords.sha256",
        )


def test_stopword_pin_mismatch_fails_closed(tmp_path: Path):
    stopwords = tmp_path / "stopwords.txt"
    pin = tmp_path / "stopwords.sha256"
    stopwords.write_text("the\nand\n", encoding="utf-8")
    pin.write_text(hashlib.sha256(b"different").hexdigest() + "\n", encoding="utf-8")

    with pytest.raises(ProjectRelevanceError, match="hash"):
        load_pinned_stopwords(stopwords, pin)
