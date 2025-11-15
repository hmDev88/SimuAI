# pages/1_LiCO2_RAG_Assistant.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag_config import PERSIST_DIR, get_embedding_model

# NOTE: we don't call st.set_page_config here because app.py already does it

st.title("🔋 Li–CO₂ RAG Assistant (local, no OpenAI)")

st.markdown(
    """
This assistant searches over your curated **Li–CO₂ knowledge base**:

- Catalyst cards (with ML labels)  
- Mechanisms  
- Design rules  
- Background  

It retrieves the most relevant text chunks and shows them to you.  
Everything runs **locally**: no OpenAI / external API calls.
"""
)


@st.cache_resource
def get_vectorstore() -> Chroma:
    embeddings = get_embedding_model()
    return Chroma(
        collection_name="lico2_unified_rag",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def retrieve_docs(question: str, top_k: int = 6, mode: str = "all") -> list[Document]:
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


mode = st.selectbox(
    "Focus mode:",
    [
        "All knowledge",
        "Catalysts only",
        "Mechanisms",
        "Design rules",
        "Background",
    ],
)

mode_map = {
    "All knowledge": "all",
    "Catalysts only": "catalyst",
    "Mechanisms": "mechanism",
    "Design rules": "design",
    "Background": "background",
}
mode_key = mode_map[mode]

question = st.text_area(
    "Your question:",
    "Which MOF/COF-based Li–CO₂ catalysts show high discharge capacity and low overpotential, "
    "and what are the main design principles?",
    height=130,
)

top_k = st.slider("Number of chunks to retrieve", 3, 15, 6)

if st.button("🔍 Retrieve"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching your Li–CO₂ knowledge base..."):
            docs = retrieve_docs(question, top_k=top_k, mode=mode_key)

        if not docs:
            st.info("No documents retrieved. Try a different question.")
        else:
            st.subheader("Retrieved context")
            for i, d in enumerate(docs, start=1):
                meta = d.metadata or {}
                st.markdown(f"### Doc {i} — *{meta.get('chunk_type', 'unknown')}*")

                # Optional: show some useful metadata if present
                name = meta.get("name") or meta.get("catalyst_name")
                perf = meta.get("performance_label_text") or meta.get("performance_label")
                cap = meta.get("capacity_mAh_g") or meta.get("discharge_capacity_mAh_g")
                over = meta.get("overpotential_V") or meta.get("measured_overpotential_v")

                info_line_parts = []
                if name:
                    info_line_parts.append(f"**Catalyst:** {name}")
                if cap is not None:
                    info_line_parts.append(f"capacity ≈ {cap}")
                if over is not None:
                    info_line_parts.append(f"overpotential ≈ {over} V")
                if perf:
                    info_line_parts.append(f"performance: {perf}")

                if info_line_parts:
                    st.markdown(" · ".join(info_line_parts))

                text = d.page_content
                st.write(text if len(text) < 1200 else text[:1200] + " ...")
                st.markdown("---")
