from __future__ import annotations

from source.services.project_chunking import (
    CHUNKER_SPECS,
    OFFICIAL_PROJECT_CHUNKERS,
    chunk_project_documents,
    entity_heuristic_chunking,
    fixed_token_chunks,
    heading_sections_chunking,
)
from source.services.project_documents import ProjectDocument


SAMPLE_POLICY = """PROJECT_ALPHA_SENTINEL REFUNDS:
PROJECT_ALPHA_SENTINEL Refund requests are accepted within thirty days. Customers must provide the booking reference.

PROJECT_ALPHA_SENTINEL BAGGAGE:
PROJECT_ALPHA_SENTINEL Delayed baggage claims require a receipt. Support reviews each claim before payment.
"""


def test_project_chunkers_execute_all_five_real_methods_with_provenance():
    documents = [
        ProjectDocument(
            source_name="policy.txt",
            text=SAMPLE_POLICY,
            metadata={"page_number": "1", "parser_method": "utf8_text"},
        )
    ]

    assert tuple(OFFICIAL_PROJECT_CHUNKERS) == (
        "entity_heuristic_w6",
        "entity_heuristic_w5",
        "entity_heuristic_w4",
        "Heading_sections_l2",
        "fixed_tok1200_ov150",
    )
    outputs = {
        method: chunk_project_documents(documents, method)
        for method in OFFICIAL_PROJECT_CHUNKERS
    }

    assert all(outputs.values())
    assert [chunk.paragraph for chunk in outputs["fixed_tok1200_ov150"]] != [
        chunk.paragraph for chunk in outputs["Heading_sections_l2"]
    ]
    for method, chunks in outputs.items():
        assert len({chunk.id for chunk in chunks}) == len(chunks)
        assert all("PROJECT_ALPHA_SENTINEL" in chunk.paragraph for chunk in chunks)
        assert all(chunk.pdf_name == "policy.txt" for chunk in chunks)
        assert all(chunk.metadata["chunker_id"] == method for chunk in chunks)
        assert all(chunk.metadata["algorithm_version"] for chunk in chunks)
        assert all(len(chunk.metadata["input_sha256"]) == 64 for chunk in chunks)
        assert all(len(chunk.metadata["output_sha256"]) == 64 for chunk in chunks)
        assert all(chunk.metadata["source_id"] == documents[0].source_id for chunk in chunks)


def test_project_chunk_ids_and_provenance_are_canonical_across_document_order():
    alpha = ProjectDocument(
        "a.txt",
        SAMPLE_POLICY,
        {"page_number": "1", "parser_method": "utf8_text_v1"},
    )
    beta = ProjectDocument(
        "b.txt",
        SAMPLE_POLICY.replace("REFUNDS", "RETURNS"),
        {"page_number": "1", "parser_method": "utf8_text_v1"},
    )

    forward = chunk_project_documents([alpha, beta], "entity_heuristic_w4")
    reversed_input = chunk_project_documents([beta, alpha], "entity_heuristic_w4")

    assert [
        (chunk.id, chunk.pdf_name, chunk.paragraph, chunk.parent_id, chunk.metadata)
        for chunk in forward
    ] == [
        (chunk.id, chunk.pdf_name, chunk.paragraph, chunk.parent_id, chunk.metadata)
        for chunk in reversed_input
    ]


def test_shared_chunkers_match_legacy_workbook_algorithms():
    from scripts.apply_candidate_chunking_methods import fixed_token_chunks as legacy_fixed
    from scripts.apply_chunking_methods import (
        entity_heuristic_chunking as legacy_entity,
        heading_sections_chunking as legacy_heading,
    )

    assert legacy_entity is entity_heuristic_chunking
    assert legacy_heading is heading_sections_chunking
    assert legacy_fixed is fixed_token_chunks


def test_project_chunker_parameters_are_exact_and_unknown_methods_fail_closed():
    assert CHUNKER_SPECS["entity_heuristic_w4"].params == {"window_size": 4}
    assert CHUNKER_SPECS["entity_heuristic_w5"].params == {"window_size": 5}
    assert CHUNKER_SPECS["entity_heuristic_w6"].params == {"window_size": 6}
    assert CHUNKER_SPECS["Heading_sections_l2"].params == {"level": 2}
    assert CHUNKER_SPECS["fixed_tok1200_ov150"].params == {
        "chunk_tokens": 1200,
        "overlap_tokens": 150,
    }

    document = ProjectDocument(
        source_name="policy.txt",
        text=SAMPLE_POLICY,
        metadata={"page_number": "1", "parser_method": "utf8_text"},
    )
    try:
        chunk_project_documents([document], "semantic_split")
    except ValueError as exc:
        assert "semantic_split" not in str(exc) or "Unknown" in str(exc)
    else:
        raise AssertionError("non-official project chunker was accepted")
