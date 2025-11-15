# rag_query.py
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI

from rag_config import PERSIST_DIR, get_embedding_model

# If you have an OpenAI API key in env, this will work.
# If you DON'T, you can comment out the call_llm function and
# just use the retrieved chunks.
client = OpenAI()


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
    """
    mode: 'all', 'catalyst', 'mechanism', 'design', 'background'
    """
    vs = get_vectorstore()
    where = None

    if mode == "catalyst":
        where = {"chunk_type": {"$in": ["catalyst_card_with_ml", "catalyst_card_with_ml_no_proba"]}}
    elif mode == "mechanism":
        where = {"chunk_type": "mechanism"}
    elif mode == "design":
        where = {"chunk_type": "design_rule"}
    elif mode == "background":
        where = {"chunk_type": "background"}

    docs = vs.similarity_search(
        question,
        k=top_k,
        filter=where,
    )
    return docs


def call_llm(question: str, docs: List[Document]) -> str:
    """
    Combines user question + retrieved docs into a single answer using OpenAI.
    Requires OPENAI_API_KEY in your environment.
    """
    context = "\n\n".join(
        [f"[DOC {i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant for Li-CO2 and related Li-based batteries. "
                "Answer based ONLY on the provided documents. "
                "Explain clearly and briefly for a materials scientist / battery engineer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Context documents:\n{context}\n\n"
                "Now answer the question using the context."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content
