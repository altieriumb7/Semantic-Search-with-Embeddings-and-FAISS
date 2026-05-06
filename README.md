# Semantic Search with Embeddings and FAISS

Portfolio-quality semantic search project that replaces keyword-only retrieval with dense vector similarity search. It includes a FAISS semantic index, TF-IDF baseline, reproducible evaluation, tests, and an interactive Streamlit demo.

## What This Project Does

- Loads a small synthetic support/product documentation dataset from `data/documents.jsonl`.
- Chunks documents into overlapping passages for retrieval.
- Generates SentenceTransformers embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Builds and caches a FAISS inner-product index over normalized embeddings.
- Builds and caches a TF-IDF keyword baseline.
- Evaluates both methods with Recall@k, Precision@k, MRR, Success@1, and nDCG@k.
- Provides a Streamlit UI for side-by-side search comparison.
- Keeps optional RAG-style answer drafting separate from retrieval metrics.

## Dataset

The included dataset is synthetic. It contains support-style articles for account access, billing, document search, integrations, security, and mobile troubleshooting. Evaluation labels are stored in `data/evaluation_queries.jsonl` as query-to-relevant-document mappings.

No private API keys are required.

## Project Structure

```text
data/
  documents.jsonl
  evaluation_queries.jsonl
indexes/
  .gitkeep
reports/
  .gitkeep
src/
  build_index.py
  chunking.py
  config.py
  data_loader.py
  embeddings.py
  evaluate_retrieval.py
  indexing.py
  metrics.py
  rag.py
  retrieval.py
  search.py
  tfidf.py
tests/
  test_chunking.py
  test_embeddings.py
  test_indexing.py
  test_metrics.py
  test_retrieval.py
app.py
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `faiss-cpu` is not available for your platform through pip, install FAISS with Conda and then install the remaining requirements:

```bash
conda install -c conda-forge faiss-cpu
pip install -r requirements.txt
```

## Build Indexes

```bash
python -m src.build_index
```

Generated artifacts are cached in `indexes/`:

- `chunks.jsonl`
- `embeddings.npy`
- `embedding_meta.json`
- `faiss.index`
- `tfidf.joblib`
- `model_cache/`

To force embedding and index regeneration:

```bash
python -m src.build_index --force-rebuild
```

## Search from the CLI

```bash
python -m src.search --query "I forgot my password and need a reset link"
python -m src.search --query "export search results to csv or excel with many records" --top-k 3
python -m src.search --query "rotate a developer API key without downtime" --method semantic
```

After `python -m src.build_index`, search loads the cached model locally by default. If you intentionally want the search command to download a missing model, add `--allow-download`.

## Evaluate Retrieval

```bash
python -m src.evaluate_retrieval
```

The evaluator writes `reports/evaluation_report.json` and prints summary metrics:

- Recall@k
- Precision@k
- MRR
- Success@1
- nDCG@k
- qualitative examples comparing semantic and TF-IDF top documents

This README does not claim a fixed percentage improvement. Run the evaluation script on the checked-in labels to compute the current results for your environment and any model changes.

### Current Results

The current local run uses `k=5` on the synthetic evaluation labels:

| Method | Recall@5 | Precision@5 | MRR |
| --- | ---: | ---: | ---: |
| Semantic FAISS | 1.000 | 0.229 | 1.000 |
| TF-IDF baseline | 1.000 | 0.229 | 1.000 |

On this intentionally small and direct dataset, the two methods tie on aggregate metrics. The qualitative examples in `reports/evaluation_report.json` are still useful for inspecting rank order and seeing where each method places neighboring documents.

## Streamlit Demo

```bash
streamlit run app.py
```

The demo includes:

- search query input
- at least three preloaded example queries
- side-by-side semantic and TF-IDF results
- similarity scores
- highlighted retrieved chunks
- Recall@k, Precision@k, and MRR summary
- optional extractive answer draft from retrieved context
- optional PCA visualization of document embeddings

## Tests

```bash
pytest -q
make test
```

The tests cover chunking, embedding cache shape, FAISS indexing, semantic retrieval, TF-IDF retrieval, and metric computation. The FAISS indexing test is skipped automatically if FAISS is not installed.

## Notes and Limitations

- The dataset is synthetic, so results demonstrate retrieval mechanics rather than production performance.
- Retrieval evaluation is computed at document level while the retrievers rank chunks.
- The optional answer draft in `src/rag.py` is extractive and local; it is not included in retrieval metrics.
- The default embedding model may need to download on first use, then it is cached locally under `indexes/model_cache/`.
- For production search, add a larger real dataset, stronger labeling, metadata filters, monitoring, and model/version tracking.
