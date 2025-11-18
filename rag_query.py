# rag_query.py — unified RAG pipeline using Gemini + Chroma
from typing import List, Tuple
import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from google import genai

from rag_config import PERSIST_DIR, COLLECTION_NAME, get_embedding_model


# Initialise Gemini client (once)
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None


def get_vectorstore() -> Chroma:
    """Shared Chroma vector store."""
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def retrieve_docs(question: str, top_k: int = 6, mode: str = "all") -> List[Document]:
    """
    RAG retrieval step.
    mode controls which chunk_type to retrieve (RAG-skeleton-style filtering):
      - 'all'
      - 'catalyst'
      - 'mechanism'
      - 'design'
      - 'background'
    """
    vs = get_vectorstore()
    where = None

    if mode == "catalyst":
        where = {
            "chunk_type": {
                "$in": ["catalyst_card_with_ml", "catalyst_card_with_ml_no_proba"]
            }
        }
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
    """Gemini RAG answer using retrieved docs."""
    if client is None:
        return "Gemini client not configured. Set GEMINI_API_KEY."

    if not docs:
        return "No documents retrieved."

    # Build compact context (like your llm_answer_gemini in app.py)
    chunks = []
    for i, d in enumerate(docs[:8]):
        meta = d.metadata or {}
        ctype = meta.get("chunk_type", "unknown")
        text = (d.page_content or "").strip()
        if len(text) > 800:
            text = text[:800] + " ..."
        chunks.append(f"[DOC {i+1} – {ctype}]\n{text}")

    context = "\n\n".join(chunks)

    prompt = (
        "You are a Li–CO₂ battery and catalyst expert.\n"
        "Use ONLY the following context from our curated corpus to answer.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write a clear, technical answer (2–4 paragraphs), "
        "summarising catalysts, design rules, mechanisms, and any available "
        "overpotential / stability insights. If something is uncertain, say so."
    )

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"


def answer_with_rag(
    question: str, top_k: int = 6, mode: str = "all"
) -> Tuple[str, List[Document]]:
    """
    High-level RAG call:
      1. retrieve docs
      2. call Gemini with those docs
    """
    docs = retrieve_docs(question, top_k=top_k, mode=mode)
    answer = call_llm(question, docs)
    return answer, docs
