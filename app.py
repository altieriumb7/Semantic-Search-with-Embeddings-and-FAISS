from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.chunking import chunk_documents
from src.config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K, EMBEDDINGS_PATH, EVALUATION_REPORT_PATH
from src.custom_data import (
    document_payload,
    documents_from_payload,
    documents_from_upload,
    evaluation_queries_from_upload,
)
from src.data_loader import load_evaluation_queries
from src.evaluate_retrieval import evaluate_searcher, report_is_current
from src.embeddings import SentenceTransformerEmbedder
from src.indexing import build_faiss_index, load_chunks
from src.rag import extractive_answer
from src.retrieval import SemanticSearcher, TfidfSearcher
from src.runtime_checks import missing_search_artifacts, search_artifacts_ready
from src.tfidf import build_tfidf


EXAMPLE_QUERIES = [
    "I forgot my password and need a reset link",
    "export search results to csv or excel with many records",
    "rotate a developer API key without downtime",
    "phone notifications are not appearing on mobile app",
]


def artifacts_exist() -> bool:
    return search_artifacts_ready(DEFAULT_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_searchers(model_name: str):
    chunks = load_chunks()
    semantic = SemanticSearcher(chunks=chunks, model_name=model_name)
    tfidf = TfidfSearcher(chunks=chunks)
    return semantic, tfidf


@st.cache_resource(show_spinner=False)
def build_visitor_searchers(documents_payload: tuple[tuple[str, str, str, str, str], ...], model_name: str):
    documents = documents_from_payload(documents_payload)
    chunks = chunk_documents(documents)
    embedder = SentenceTransformerEmbedder(model_name=model_name, local_files_only=True)
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    semantic = SemanticSearcher(chunks=chunks, faiss_index=build_faiss_index(embeddings), embedder=embedder)
    vectorizer, matrix = build_tfidf(chunks)
    tfidf = TfidfSearcher(chunks=chunks, vectorizer=vectorizer, matrix=matrix)
    return semantic, tfidf, chunks


@st.cache_data(show_spinner=False)
def load_or_compute_metrics(model_name: str, top_k: int) -> dict:
    if EVALUATION_REPORT_PATH.exists():
        report = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
        if report_is_current(report, model_name, top_k):
            return report["metrics"]
    queries = load_evaluation_queries()
    semantic, tfidf = load_searchers(model_name)
    _, semantic_summary = evaluate_searcher(semantic, queries, top_k)
    _, tfidf_summary = evaluate_searcher(tfidf, queries, top_k)
    return {"semantic": semantic_summary, "tfidf": tfidf_summary}


@st.cache_data(show_spinner=False)
def embedding_projection() -> pd.DataFrame:
    chunks = load_chunks()
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(embeddings) < 2:
        return pd.DataFrame()
    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    return pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "title": [chunk.title for chunk in chunks],
            "category": [chunk.category for chunk in chunks],
            "doc_id": [chunk.doc_id for chunk in chunks],
        }
    )


def highlight_query_terms(text: str, query: str) -> str:
    escaped = html.escape(text)
    terms = sorted({term.lower() for term in re.findall(r"[A-Za-z0-9]{4,}", query)}, key=len, reverse=True)
    for term in terms:
        pattern = re.compile(rf"({re.escape(term)})", flags=re.IGNORECASE)
        escaped = pattern.sub(r"<mark>\1</mark>", escaped)
    return escaped


def render_results(title: str, results, query: str) -> None:
    st.subheader(title)
    for result in results:
        with st.container(border=True):
            st.caption(f"Rank {result.rank} | score {result.score:.4f} | {result.chunk.category}")
            st.markdown(f"**{html.escape(result.chunk.title)}**")
            st.code(result.chunk.doc_id, language=None)
            st.markdown(
                f"<p>{highlight_query_terms(result.chunk.text, query)}</p>",
                unsafe_allow_html=True,
            )


def metric_cards(metrics: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("Recall@k", f"{metrics['recall_at_k']:.3f}")
    cols[1].metric("Precision@k", f"{metrics['precision_at_k']:.3f}")
    cols[2].metric("MRR", f"{metrics['mrr']:.3f}")


def render_metric_summary(semantic_metrics: dict, tfidf_metrics: dict) -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Semantic search**")
        metric_cards(semantic_metrics)
    with col_b:
        st.markdown("**TF-IDF baseline**")
        metric_cards(tfidf_metrics)


def render_bundled_demo(top_k: int, selected_example: str, show_answer: bool, show_projection: bool) -> None:
    query = st.text_input("Search query", value=selected_example)
    if not query.strip():
        st.warning("Enter a search query.")
        st.stop()
    semantic, tfidf = load_searchers(DEFAULT_MODEL_NAME)

    semantic_results = semantic.search(query, top_k=top_k)
    tfidf_results = tfidf.search(query, top_k=top_k)

    metrics = load_or_compute_metrics(DEFAULT_MODEL_NAME, top_k)
    st.markdown("### Evaluation Summary")
    render_metric_summary(metrics["semantic"], metrics["tfidf"])

    st.markdown("### Side-by-Side Retrieval")
    left, right = st.columns(2)
    with left:
        render_results("Embedding retrieval", semantic_results, query)
    with right:
        render_results("Keyword retrieval", tfidf_results, query)

    if show_answer:
        st.markdown("### Optional Answer Draft")
        st.info(extractive_answer(query, semantic_results))

    if show_projection:
        st.markdown("### Document Embedding PCA")
        projection = embedding_projection()
        if projection.empty:
            st.info("At least two embeddings are required for PCA.")
        else:
            st.scatter_chart(projection, x="pc1", y="pc2", color="category", size=60)
            st.dataframe(projection[["doc_id", "title", "category"]], hide_index=True, use_container_width=True)


def render_visitor_demo(top_k: int) -> None:
    st.markdown("### Try Your Own Documents")
    st.info(
        "Uploads are processed in memory for this session only. They are not saved, committed, or used to retrain the hosted demo."
    )
    st.caption("Accepted formats: JSONL or CSV with columns `doc_id`, `title`, `category`, `text`. Only `text` is required.")

    sample = pd.DataFrame(
        [
            {
                "doc_id": "policy_1",
                "title": "Refund policy",
                "category": "billing",
                "text": "Annual subscriptions can be refunded within fourteen days when usage is minimal.",
            },
            {
                "doc_id": "security_1",
                "title": "API key rotation",
                "category": "security",
                "text": "Create a replacement key, update the application, verify traffic, then revoke the old key.",
            },
        ]
    )
    with st.expander("Example CSV schema"):
        st.dataframe(sample, hide_index=True, use_container_width=True)

    upload = st.file_uploader("Upload documents", type=["jsonl", "csv"], key="visitor_documents")
    if upload is None:
        return

    try:
        documents = documents_from_upload(upload.name, upload.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    payload = document_payload(documents)
    with st.spinner("Building a temporary in-memory search index..."):
        semantic, tfidf, chunks = build_visitor_searchers(payload, DEFAULT_MODEL_NAME)

    st.success(f"Loaded {len(documents)} documents and built {len(chunks)} chunks for this session.")
    st.dataframe(
        pd.DataFrame({"doc_id": doc.doc_id, "title": doc.title, "category": doc.category} for doc in documents),
        hide_index=True,
        use_container_width=True,
    )

    query = st.text_input("Search your uploaded documents", key="visitor_query")
    if query.strip():
        left, right = st.columns(2)
        with left:
            render_results("Embedding retrieval", semantic.search(query, top_k=top_k), query)
        with right:
            render_results("Keyword retrieval", tfidf.search(query, top_k=top_k), query)

    st.markdown("### Optional Self-Evaluation")
    st.caption(
        "Upload JSONL or CSV with `query` and `relevant_doc_ids`. Separate multiple relevant IDs with commas or semicolons."
    )
    eval_upload = st.file_uploader("Upload labeled evaluation queries", type=["jsonl", "csv"], key="visitor_eval")
    if eval_upload is None:
        return
    try:
        queries = evaluation_queries_from_upload(eval_upload.name, eval_upload.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    semantic_rows, semantic_summary = evaluate_searcher(semantic, queries, top_k)
    tfidf_rows, tfidf_summary = evaluate_searcher(tfidf, queries, top_k)
    render_metric_summary(semantic_summary, tfidf_summary)
    st.dataframe(
        pd.DataFrame(
            {
                "query_id": sem["query_id"],
                "query": sem["query"],
                "relevant": ", ".join(sem["relevant_doc_ids"]),
                "semantic_top_docs": ", ".join(sem["retrieved_doc_ids"][:3]),
                "tfidf_top_docs": ", ".join(tf["retrieved_doc_ids"][:3]),
            }
            for sem, tf in zip(semantic_rows, tfidf_rows)
        ),
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Semantic Search with FAISS", layout="wide")
    st.title("Semantic Search with FAISS")
    st.caption("Synthetic support documentation dataset. Retrieval metrics exclude optional answer drafting.")

    if not artifacts_exist():
        st.error("Search artifacts are missing.")
        missing = "\n".join(str(path) for path in missing_search_artifacts(DEFAULT_MODEL_NAME))
        st.code(missing, language="text")
        st.code("python -m src.build_index\npython -m src.evaluate_retrieval\nstreamlit run app.py", language="bash")
        st.stop()

    with st.sidebar:
        selected_example = st.selectbox("Example queries", EXAMPLE_QUERIES)
        top_k = st.slider("Top k", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        show_answer = st.toggle("Draft answer from semantic results", value=False)
        show_projection = st.toggle("Show embedding PCA", value=True)

    bundled_tab, visitor_tab = st.tabs(["Bundled Demo", "Try Your Own Data"])
    with bundled_tab:
        render_bundled_demo(top_k, selected_example, show_answer, show_projection)
    with visitor_tab:
        render_visitor_demo(top_k)


if __name__ == "__main__":
    main()

