from pathlib import Path

# Project root: .../hybrid_rag_project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_DIRECTORY = PROJECT_ROOT / "data" / "pdfs"
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"

COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

DENSE_CANDIDATES = 10
SPARSE_CANDIDATES = 10
DEFAULT_TOP_K = 5

RRF_K = 60
