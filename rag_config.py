# rag_config.py
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings

# Where the JSON/JSONL files live
DATA_DIR = Path("rag_data")

# Where Chroma will store the vector DB
PERSIST_DIR = "rag_index"

def get_embedding_model():
    """
    Small sentence-transformer model, works fine on CPU/M1.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
