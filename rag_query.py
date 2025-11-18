# rag_query.py — clean version (Gemini-only, no tester)
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from google import genai
import os

from rag_config import PERSIST_DIR, get_embedding_model

# Initialize Gemini client (only if key exists)
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None


def get_vectorstore() -> Chroma:
    embeddings = get_embedding_model()
    return Chroma(
        collection_name="lico2_unified_rag",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def retrieve_docs(
    question: str,
    top_k: int = 6,
    mode: str = "all",
) -> List[Document]:

    vs = get_vectorstore()
    where = None

    if mode == "catalyst":
        where = {"chunk_type": {"$in": ["catalyst_card_with_ml", "catalyst_card_with_ml_no_proba"]}}
    elif mode == "mechanism":
        where = {"chunk_type": "mechanism_bilingual"}
    elif mode == "design":
        where = {"chunk_type": "design_rule_bilingual"}
    elif mode == "background":
        where = {"chunk_type": "background_bilingual"}

    docs = vs.similarity_search(
        question,
        k=top_k,
        filter=where,
    )
    return docs


def call_llm(question: str, docs: List[Document]) -> str:
    """Gemini RAG answer using retrieved docs"""

    if client is None:
        return "Gemini client not configured. Set GEMINI_API_KEY."

    context = "\n\n".join(
        [f"[DOC {i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    prompt = f"""
You are an expert assistant for Li–CO₂ and Li-based battery mechanisms and catalysts.
Answer based ONLY on the retrieved documents.

Question:
{question}

Retrieved context:
{context}

Give a clear, accurate, technical answer.
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )

    return response.text
