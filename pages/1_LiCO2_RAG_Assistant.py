# pages/1_LiCO2_RAG_Assistant.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag_config import PERSIST_DIR, get_embedding_model

# NOTE: we don't call st.set_page_config here because app.py already does it

st.title("🔋 Li–CO₂ RAG Assistant")

st.markdown(
    """
This assistant searches over your curated **Li–CO₂ knowledge base**:

- Catalyst cards (with ML labels)  
- Mechanisms  
- Design rules  
- Background  

It retrieves the most relevant text chunks and shows them to you. 
"""
)

def build_extractive_answer(question: str, docs: list[Document]) -> str:
    """
    Build a human-readable answer using only the retrieved docs.
    No LLM, just simple heuristics + metadata.
    """
    catalysts = []
    design_rules = []
    mechanisms = []
    background = []

    for d in docs:
        meta = d.metadata or {}
        ctype = meta.get("chunk_type", "")
        text = (d.page_content or "").strip()

        # Catalyst cards
        if "catalyst" in ctype:
            name = meta.get("name") or meta.get("catalyst_name")
            cap = meta.get("capacity_mAh_g") or meta.get("discharge_capacity_mAh_g")
            over = meta.get("overpotential_V") or meta.get("measured_overpotential_v")
            perf = meta.get("performance_label_text") or meta.get("performance_label")

            desc_parts = []
            if name:
                desc_parts.append(f"**{name}**")
            if cap is not None:
                desc_parts.append(f"capacity ≈ {cap} mAh g⁻¹")
            if over is not None:
                desc_parts.append(f"overpotential ≈ {over} V")
            if perf:
                desc_parts.append(f"performance: {perf}")

            if desc_parts:
                catalysts.append(" – " + ", ".join(desc_parts))

        # Design rules: take the first sentence
        elif ctype == "design_rule":
            sent = text.split(". ")[0]
            if sent:
                design_rules.append(" – " + sent.strip())

        # Mechanisms: also first sentence
        elif ctype == "mechanism":
            sent = text.split(". ")[0]
            if sent:
                mechanisms.append(" – " + sent.strip())

        # Background: keep a couple of sentences
        elif ctype == "background":
            sent = text.split(". ")[0]
            if sent:
                background.append(" – " + sent.strip())

    lines = []
    lines.append(f"### 🧾 Answer (no LLM, built from {len(docs)} retrieved chunks)\n")

    if catalysts:
        lines.append("**Promising catalysts mentioned:**")
        for c in catalysts[:5]:
            lines.append(c)
        lines.append("")  # blank line

    if design_rules:
        lines.append("**Key design principles found:**")
        for r in design_rules[:5]:
            lines.append(r)
        lines.append("")

    if mechanisms:
        lines.append("**Mechanistic insights:**")
        for m in mechanisms[:5]:
            lines.append(m)
        lines.append("")

    if background and not (catalysts or design_rules or mechanisms):
        # fallback if the question was more general
        lines.append("**Background information:**")
        for b in background[:5]:
            lines.append(b)
        lines.append("")

    if len(lines) <= 1:
        lines.append("I couldn’t extract a clear answer from the retrieved chunks. "
                     "Try rephrasing your question or being more specific.")

    return "\n".join(lines)


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
                    # --- Build an extractive answer from the retrieved docs ---
            st.subheader("🧠 Synthesized answer (local, no LLM)")
            answer_md = build_extractive_answer(question, docs)
            st.markdown(answer_md)

