"""Central configuration — loads from .env and exposes typed settings."""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
ENTITIES_PATH = DATA_PROCESSED_DIR / "entities.jsonl"
DATA_DB_DIR = ROOT_DIR / os.getenv("DATA_DB_DIR", "data/db")
VECTOR_DB_PATH = ROOT_DIR / os.getenv("VECTOR_DB_PATH", "data/db/vectors.duckdb")
CACHE_DB_PATH = ROOT_DIR / os.getenv("CACHE_DB_PATH", "data/db/cache.duckdb")
GRAPH_PATH = ROOT_DIR / os.getenv("GRAPH_PATH", "data/processed/knowledge_graph.json")

# Ensure directories exist
for _dir in [DATA_PROCESSED_DIR, DATA_DB_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# LLM
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
PROMPTS_DIR = ROOT_DIR / "src/prompts"

# Entities
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1500"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512          # tokens per chunk
CHUNK_OVERLAP: int = 64        # overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_CHUNKS: int = 10         # initial vector search hits
GRAPH_HOP_DEPTH: int = 1       # graph traversal depth after vector search
RERANK_TOP_K: int = 5          # final chunks fed to LLM

# ── Semantic Cache ────────────────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
CACHE_MAX_ENTRIES: int = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))
