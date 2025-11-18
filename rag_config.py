# rag_config.py
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass(frozen=True)
class RAGConfig:
    persist_dir: Path = Path("rag_index")
    collection_name: str = "lico2_unified_rag"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_top_k: int = 6


CFG = RAGConfig()

# Simple constants for other modules
PERSIST_DIR = str(CFG.persist_dir)
COLLECTION_NAME = CFG.collection_name


@lru_cache(maxsize=1)
def get_embedding_model():
    """Shared embedding model for all RAG components."""
    return HuggingFaceEmbeddings(model_name=CFG.embedding_model)
