from src.chunking import chunk_document, chunk_text
from src.data_loader import Document


def test_chunk_text_uses_overlap():
    text = " ".join(f"word{i}" for i in range(12))
    chunks = chunk_text(text, chunk_size_words=5, overlap_words=2)

    assert chunks == [
        "word0 word1 word2 word3 word4",
        "word3 word4 word5 word6 word7",
        "word6 word7 word8 word9 word10",
        "word9 word10 word11",
    ]


def test_chunk_document_preserves_metadata():
    document = Document(
        doc_id="doc-a",
        title="Example",
        category="support",
        source="synthetic",
        text="alpha beta gamma delta epsilon zeta",
    )

    chunks = chunk_document(document, chunk_size_words=4, overlap_words=1)

    assert chunks[0].chunk_id == "doc-a::chunk_0"
    assert chunks[0].title == "Example"
    assert chunks[0].doc_id == "doc-a"
    assert len(chunks) == 2

