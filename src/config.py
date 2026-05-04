from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
REPORTS_DIR = PROJECT_ROOT / "reports"

DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
EVALUATION_QUERIES_PATH = DATA_DIR / "evaluation_queries.jsonl"

CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
EMBEDDING_META_PATH = INDEX_DIR / "embedding_meta.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
TFIDF_PATH = INDEX_DIR / "tfidf.joblib"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE_WORDS = 80
DEFAULT_CHUNK_OVERLAP_WORDS = 20
DEFAULT_TOP_K = 5

