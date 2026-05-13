import pytest

from src.custom_data import (
    documents_from_upload,
    document_payload,
    documents_from_payload,
    evaluation_queries_from_upload,
)


def test_documents_from_csv_upload_accepts_minimal_schema():
    content = b"doc_id,title,category,text\nrefund,Refunds,billing,Refunds are available for eligible plans.\n"

    documents = documents_from_upload("docs.csv", content)

    assert documents[0].doc_id == "refund"
    assert documents[0].text.startswith("Refunds")


def test_documents_from_jsonl_upload_rejects_duplicate_ids():
    content = b'{"doc_id":"a","text":"first"}\n{"doc_id":"a","text":"second"}\n'

    with pytest.raises(ValueError, match="Duplicate doc_id"):
        documents_from_upload("docs.jsonl", content)


def test_document_payload_round_trip():
    documents = documents_from_upload("docs.csv", b"doc_id,text\na,Alpha text\n")

    assert documents_from_payload(document_payload(documents)) == documents


def test_evaluation_queries_from_csv_upload_splits_relevant_ids():
    content = b"query_id,query,relevant_doc_ids\nq1,how to refund,refund;billing\n"

    queries = evaluation_queries_from_upload("eval.csv", content)

    assert queries[0].query == "how to refund"
    assert queries[0].relevant_doc_ids == ("refund", "billing")
