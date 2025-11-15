# pages/2_Hybrid_Catalyst_Agent.py

import streamlit as st
import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag_config import PERSIST_DIR, get_embedding_model

CSV_PATH = "Catalyst Database.csv"

# We don't call set_page_config here because app.py already does that

st.title("🧠 Hybrid Catalyst Agent (CSV + RAG)")

st.markdown(
    """
This agent combines:

1. **Numeric filters** on your `Catalyst Database.csv` (capacity, overpotential, barriers, ...).
2. **Semantic search** over your curated Li–CO₂ RAG index:
   - catalyst cards (with ML labels),
   - mechanisms,
   - design rules,
   - background.
"""
)

# ---------------------------
# Load CSV
# ---------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

try:
    df = load_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{CSV_PATH}` in the repository root.")
    st.stop()

st.caption(f"Dataset loaded: {CSV_PATH}  —  {df.shape[0]} rows, {df.shape[1]} columns.")


# ---------------------------
# Utility: auto-detect key cols
# ---------------------------
lower_cols = {c.lower(): c for c in df.columns}

def find_col(substrings):
    for lc, orig in lower_cols.items():
        if any(s in lc for s in substrings):
            return orig
    return None

cap_col     = find_col(["capacity", "mah"])
over_col    = find_col(["overpot", "overpotential"])
delta_col   = find_col(["adsorption", "deltae"])
barrier_col = find_col(["barrier"])
cat_name_col= find_col(["catalyst", "name"])

st.write("**Detected columns:**")
st.write({
    "capacity": cap_col,
    "overpotential": over_col,
    "adsorption energy": delta_col,
    "reaction barrier": barrier_col,
    "catalyst name": cat_name_col,
})

# ---------------------------
# Build simple numeric constraints from text
# ---------------------------
def apply_numeric_agent(query: str, df_in: pd.DataFrame):
    """Parse phrases like 'high capacity', 'low overpotential', 'low barrier' and filter df."""
    q = query.lower()
    d = df_in.copy()
    expl = []

    # high capacity
    if cap_col and cap_col in d.columns and ("high capacity" in q or "large capacity" in q):
        thr = d[cap_col].quantile(0.75)
        d = d[d[cap_col] >= thr]
        expl.append(f"{cap_col} ≥ {thr:.0f}")

    # low capacity (if user ever writes this)
    if cap_col and cap_col in d.columns and ("low capacity" in q or "small capacity" in q):
        thr = d[cap_col].quantile(0.25)
        d = d[d[cap_col] <= thr]
        expl.append(f"{cap_col} ≤ {thr:.0f}")

    # low overpotential
    if over_col and over_col in d.columns and ("low overpotential" in q or "small overpotential" in q):
        thr = d[over_col].quantile(0.25)
        d = d[d[over_col] <= thr]
        expl.append(f"{over_col} ≤ {thr:.2f}")

    # favourable adsorption
    if delta_col and delta_col in d.columns and (
        "low adsorption" in q
        or "strong adsorption" in q
        or "favourable adsorption" in q
        or "favorable adsorption" in q
    ):
        thr = d[delta_col].quantile(0.25)
        d = d[d[delta_col] <= thr]
        expl.append(f"{delta_col} ≤ {thr:.2f}")

    # low barrier
    if barrier_col and barrier_col in d.columns and ("low barrier" in q or "small barrier" in q):
        thr = d[barrier_col].quantile(0.25)
        d = d[d[barrier_col] <= thr]
        expl.append(f"{barrier_col} ≤ {thr:.2f}")

    # rank by capacity if available
    if cap_col and cap_col in d.columns and pd.api.types.is_numeric_dtype(d[cap_col]):
        d = d.sort_values(cap_col, ascending=False)

    return d, expl

# ---------------------------
# RAG: load vectorstore
# ---------------------------
@st.cache_resource
def get_vectorstore() -> Chroma:
    embeddings = get_embedding_model()
    return Chroma(
        collection_name="lico2_unified_rag",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def retrieve_rag_for_catalyst(vs: Chroma, catalyst_name: str, user_query: str, k: int = 3) -> list[Document]:
    """
    Retrieve RAG docs (catalyst cards + mechanisms + design rules) for a given catalyst.
    """
    # We search across catalyst+mechanism+design chunks with the catalyst name + user query
    search_text = f"{user_query}\nFocus on catalyst: {catalyst_name}"
    docs = vs.similarity_search(
        search_text,
        k=k,
        filter={"chunk_type": {"$in": [
            "catalyst_card_with_ml",
            "catalyst_card_with_ml_no_proba",
            "mechanism",
            "design_rule",
        ]}},
    )
    return docs

# ---------------------------
# UI: Hybrid agent
# ---------------------------
st.markdown("### 🔬 Ask the hybrid agent")

user_q = st.text_area(
    "Describe what you want (constraints + hints):",
    "High capacity and low overpotential catalysts with low reaction barrier. Prefer Cu- or Co-based MOFs/COFs.",
    height=120,
)

top_n = st.slider("Top N candidates (from CSV) to inspect", 3, 15, 5)

if st.button("Run hybrid agent"):
    if not user_q.strip():
        st.warning("Please type a query first.")
    else:
        # 1) numeric filtering on CSV
        filtered, expl = apply_numeric_agent(user_q, df)
        st.subheader("Step 1 – Numeric filtering on CSV")

        if expl:
            st.write("Applied numeric filters: " + " ; ".join(expl))
        else:
            st.write("No specific numeric keywords detected; showing top rows only.")

        if filtered.empty:
            st.error("No rows satisfy the numeric constraints. Try relaxing your query.")
            st.stop()

        st.write(f"{len(filtered)} rows after filtering. Showing top {top_n}:")
        top_candidates = filtered.head(top_n)
        st.dataframe(top_candidates, use_container_width=True)

        # 2) Use catalyst names (if any) to query RAG
        st.subheader("Step 2 – RAG evidence for top candidates")

        if not cat_name_col or cat_name_col not in top_candidates.columns:
            st.info("No explicit 'catalyst name' column detected. RAG search will be skipped.")
        else:
            vs = get_vectorstore()
            for idx, row in top_candidates.iterrows():
                cat_name = str(row[cat_name_col])
                st.markdown(f"#### 📌 Catalyst: **{cat_name}**")

                docs = retrieve_rag_for_catalyst(vs, cat_name, user_q, k=3)
                if not docs:
                    st.write("_No RAG documents found for this catalyst._")
                    continue

                for i, d in enumerate(docs, start=1):
                    meta = d.metadata or {}
                    ctype = meta.get("chunk_type", "unknown")
                    st.markdown(f"**Doc {i} — {ctype}**")
                    text = d.page_content
                    st.write(text if len(text) < 800 else text[:800] + " ...")
                    st.markdown("---")
