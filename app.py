import os
from dotenv import load_dotenv
load_dotenv() 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nbformat
from collections import Counter
from sklearn.decomposition import PCA
from langchain_core.documents import Document


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# RAG pipeline
from rag_config import get_embedding_model
from rag_query import retrieve_docs, call_llm, answer_with_rag

# ------------------------------------------------
# Paths & config
# ------------------------------------------------
NOTEBOOK_PATH = "Catalyst.ipynb"
CSV_PATH = "Catalyst Database.csv"


# ------------------------------------------------
# Helpers: notebook loading & data
# ------------------------------------------------
@st.cache_resource
def load_nb_namespace(nb_path: str):
    """Executes code cells from a notebook and returns its namespace."""
    ns = {}
    if not os.path.exists(nb_path):
        ns.setdefault("__errors__", []).append(f"Notebook not found at: {nb_path}")
        return ns
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "\n".join(
                [
                    line
                    for line in cell.source.splitlines()
                    if not line.strip().startswith(("%", "!"))
                ]
            )
            try:
                exec(compile(src, nb_path, "exec"), ns)
            except Exception as e:
                ns.setdefault("__errors__", []).append(repr(e))
    return ns


@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)







def build_extractive_answer(question: str, docs):
    """Local, LLM-free extractive answer from retrieved docs."""
    if not docs:
        return "No documents retrieved."

    catalysts, rules, mech, background, generic = [], [], [], [], []

    for d in docs:
        meta = d.metadata or {}
        ctype = (meta.get("chunk_type") or "").lower()
        text = (d.page_content or "").strip()
        parts = text.split(". ")
        short = ". ".join(parts[:2]).strip()

        if "catalyst" in ctype:
            catalysts.append("– " + short)
        elif "design" in ctype or "rule" in ctype:
            rules.append("– " + short)
        elif "mechanism" in ctype:
            mech.append("– " + short)
        elif "background" in ctype:
            background.append("– " + short)
        else:
            generic.append("– " + short)

    out = [f"### 🧠 Local Extractive Answer (from {len(docs)} chunks)\n"]

    if catalysts:
        out.append("**Catalyst insights:**")
        out += catalysts[:5]
        out.append("")

    if rules:
        out.append("**Design rules:**")
        out += rules[:5]
        out.append("")

    if mech:
        out.append("**Mechanistic insights:**")
        out += mech[:5]
        out.append("")

    if background and not (catalysts or rules or mech):
        out.append("**Background info:**")
        out += background[:5]
        out.append("")

    if not any([catalysts, rules, mech, background]):
        out.append(
            "No clear structured information was extracted, but see retrieved context above."
        )

    return "\n".join(out)



def summarize_chunk_types(docs):
    """Count how many retrieved chunks of each type we have."""
    counts = Counter()
    for d in docs:
        meta = d.metadata or {}
        ctype = (meta.get("chunk_type") or "unknown").lower()
        counts[ctype] += 1
    return counts


def compute_similarity_scores(question: str, docs):
    """Compute true cosine similarity between query and each doc chunk."""
    if not docs:
        return np.array([])

    emb = get_embedding_model()
    # Query embedding
    q_vec = np.array(emb.embed_query(question))
    # Doc embeddings
    doc_texts = [d.page_content or "" for d in docs]
    d_vecs = np.array(emb.embed_documents(doc_texts))

    # Cosine similarity
    q_norm = np.linalg.norm(q_vec) + 1e-8
    d_norms = np.linalg.norm(d_vecs, axis=1) + 1e-8
    sims = (d_vecs @ q_vec) / (d_norms * q_norm)
    return sims  # shape (k,)


def project_embeddings(question: str, docs):
    """Project query + doc embeddings into 2D with PCA for visualization."""
    if not docs:
        return None, None

    emb = get_embedding_model()
    texts = [question] + [d.page_content or "" for d in docs]
    vecs = np.array(emb.embed_documents(texts))  # (1 + k, dim)

    if vecs.shape[0] < 2:
        return None, None

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)  # (1 + k, 2)
    q_coord = coords[0]
    doc_coords = coords[1:]
    return q_coord, doc_coords


def recommend_mof_candidates(anode: str, cathode: str, df: pd.DataFrame, top_k: int = 5):
    """
    Given anode + cathode chemical names and the catalyst dataframe,
    return top-k MOF-like catalysts ranked by embedding similarity
    to the query.
    """
    if "Catalyst" not in df.columns:
        return pd.DataFrame()  # fail gracefully

    # 1) Filter to MOF-like catalysts (very simple heuristic)
    cat_series = df["Catalyst"].fillna("").astype(str)

    mof_mask = cat_series.str.contains(
        r"mof|zif|uio-|hkust", case=False, regex=True
    )
    mof_df = df[mof_mask].copy()

    if mof_df.empty:
        return pd.DataFrame()  # no MOFs found

    # 2) Build a text representation for each MOF
    texts = mof_df["Catalyst"].fillna("").astype(str).tolist()

    # 3) Embed query + candidates
    emb = get_embedding_model()

    query_text = (
        f"Best MOF catalyst for a Li-CO₂ battery "
        f"with cathode {cathode} and anode {anode}. "
        f"Prioritise stability, low overpotential and good CO₂ reduction activity."
    )
    q_vec = np.array(emb.embed_query(query_text))

    cand_vecs = np.array(emb.embed_documents(texts))

    # 4) Cosine similarity
    q_norm = np.linalg.norm(q_vec) + 1e-8
    c_norms = np.linalg.norm(cand_vecs, axis=1) + 1e-8
    sims = (cand_vecs @ q_vec) / (c_norms * q_norm)  # shape (n_candidates,)

    # 5) Sort and keep top-k
    order = np.argsort(-sims)  # descending
    top_idx = order[:top_k]
    top_sims = sims[top_idx]

    result = mof_df.iloc[top_idx].copy()
    result["similarity"] = top_sims

    return result

# ------------------------------------------------
# Streamlit page setup
# ------------------------------------------------
st.set_page_config(
    page_title="SimuAI – Li–CO₂ Catalyst Assistant", layout="wide"
)
st.title("SimuAI – Li–CO₂ Catalyst Assistant")

# load notebook namespace & CSV
ns = load_nb_namespace(NOTEBOOK_PATH)
nb_errors = ns.get("__errors__", [])

if not os.path.exists(CSV_PATH):
    st.error(f"Missing data file: {CSV_PATH}")
    st.stop()

df = load_data(CSV_PATH)

tab1, tab2 = st.tabs(
    ["⚙️ Catalyst Explorer & Trainer", "🧪 RAG QA (Local / Gemini)"]
)

# ------------------------------------------------
# TAB 1: Explorer & Trainer
# ------------------------------------------------
with tab1:
    st.caption(
        "Filter catalysts, parse names, train models, and download classified results."
    )

    # Sidebar: Filters
    with st.sidebar:
        st.header("Filters")
        txt_col = st.selectbox(
            "Search column", df.select_dtypes("object").columns
        )
        query = st.text_input("Contains text")
        num_cols = df.select_dtypes("number").columns
        selected_num = st.multiselect(
            "Numeric filters", num_cols, default=list(num_cols)
        )
        ranges = {}
        for c in selected_num:
            cmin, cmax = float(df[c].min()), float(df[c].max())
            ranges[c] = st.slider(c, cmin, cmax, (cmin, cmax))
        show_cols = st.multiselect(
            "Columns to show", list(df.columns), default=list(df.columns)
        )

    mask = pd.Series(True, index=df.index)
    if query:
        mask &= df[txt_col].fillna("").str.contains(query, case=False)
    for c, (lo, hi) in ranges.items():
        mask &= df[c].between(lo, hi)
    filtered = df.loc[mask, show_cols].copy()

    st.subheader("📄 Filtered Table")
    st.write(f"{len(filtered)}/{len(df)} rows")
    st.dataframe(filtered, use_container_width=True)

    # Parsing
    st.subheader("🔎 Catalyst Parser")

    normalize_text = ns.get("normalize_text")
    detect_type = ns.get("detect_type")
    extract_form = ns.get("extract_form")
    extract_metals = ns.get("extract_metals")
    infer_structure = ns.get("infer_structure")

    row_list = filtered.index.tolist()
    row = (
        st.selectbox("Select a row", options=row_list)
        if len(row_list) > 0
        else None
    )
    cat_input = ""
    if row is not None and "Catalyst" in filtered.columns:
        cat_input = filtered.loc[row, "Catalyst"]
    cat_input = st.text_input("Catalyst name", value=cat_input)

    parsed = {}
    if cat_input:
        try:
            parsed["normalized"] = (
                normalize_text(cat_input) if normalize_text else cat_input
            )
            parsed["type"] = (
                detect_type(parsed["normalized"]) if detect_type else None
            )
            parsed["form"] = (
                extract_form(parsed["normalized"]) if extract_form else None
            )
            parsed["metals"] = (
                extract_metals(parsed["normalized"])
                if extract_metals
                else None
            )
            parsed["structure"] = (
                infer_structure(parsed["normalized"], "")
                if infer_structure
                else None
            )
        except Exception as e:
            parsed["error"] = repr(e)
    st.json(parsed if parsed else {"info": "Enter or select a catalyst name"})

    if nb_errors:
        with st.expander("Notebook load warnings"):
            for e in nb_errors:
                st.code(e)

    # Model Trainer
    st.subheader("⚙️ Train Machine Learning Model")

    st.markdown(
        """
Train an **XGBoost** classifier and see Accuracy, Macro-F1, Confusion Matrix, and Classification Report.
The app automatically encodes text columns and handles missing data.
"""
    )

    cols = list(df.columns)
    if len(cols) < 2:
        st.error("Dataset must have at least 2 columns.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            x_cols = st.multiselect(
                "Feature columns (X)", cols[:-1], default=cols[:-1]
            )
        with c2:
            y_col = st.selectbox("Target column (y)", cols, index=len(cols) - 1)

        test_size = st.slider("Test size (%)", 10, 50, 20) / 100.0
        max_depth = st.slider("max_depth", 2, 12, 6)
        n_estimators = st.slider(
            "n_estimators", 50, 500, 200, step=50
        )
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)

        if st.button("🚀 Train Model"):
            if len(x_cols) == 0 or y_col not in df.columns:
                st.warning("Please select valid features and target.")
            else:
                X_raw = df[x_cols].copy()
                y_raw = df[y_col].copy()

                # Clean X
                for c in X_raw.columns:
                    if pd.api.types.is_bool_dtype(X_raw[c]):
                        X_raw[c] = X_raw[c].astype(int)
                X = pd.get_dummies(X_raw, dummy_na=True)
                X = X.fillna(0)

                # Clean y
                y = y_raw.copy()
                label_mapping = None
                if (
                    y.dtype == "object"
                    or pd.api.types.is_categorical_dtype(y)
                ):
                    y = y.astype("category")
                    label_mapping = {
                        k: v for v, k in enumerate(y.cat.categories)
                    }
                    y = y.cat.codes

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y if len(np.unique(y)) > 1 else None,
                )

                # Train model
                model = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluate
                acc = accuracy_score(y_test, y_pred)
                f1m = f1_score(y_test, y_pred, average="macro")

                cA, cB = st.columns(2)
                with cA:
                    st.metric("Accuracy", f"{acc*100:.2f}%")
                with cB:
                    st.metric("Macro F1", f"{f1m*100:.2f}%")

                cm = confusion_matrix(y_test, y_pred)
                st.markdown("### Confusion Matrix")
                fig, ax = plt.subplots()
                ax.imshow(cm)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                        )
                st.pyplot(fig)

                st.markdown("### Classification Report")
                if label_mapping:
                    inv_map = {v: k for k, v in label_mapping.items()}
                    y_test_named = pd.Series(y_test).map(inv_map).astype(str)
                    y_pred_named = pd.Series(y_pred).map(inv_map).astype(str)
                    report = classification_report(
                        y_test_named, y_pred_named, digits=3
                    )
                else:
                    report = classification_report(
                        y_test, y_pred, digits=3
                    )
                st.code(report, language="text")

                # Download test predictions
                out = (
                    pd.DataFrame(
                        {
                            "y_true": y_test
                            if label_mapping is None
                            else pd.Series(y_test).map(inv_map),
                            "y_pred": y_pred
                            if label_mapping is None
                            else pd.Series(y_pred).map(inv_map),
                        }
                    )
                    .reset_index(drop=True)
                )
                st.download_button(
                    "Download Test Predictions (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="model_predictions.csv",
                    mime="text/csv",
                )

                # Predict full dataset and download
                y_all_pred = model.predict(X)
                if label_mapping:
                    inv_map = {v: k for k, v in label_mapping.items()}
                    y_all_pred = pd.Series(y_all_pred).map(inv_map)

                classified_df = df.copy()
                classified_df["Predicted_Label"] = y_all_pred

                st.markdown("### 📁 Full Classified Dataset")
                st.dataframe(
                    classified_df.head(), use_container_width=True
                )

                st.download_button(
                    "⬇️ Download Full Classified Data (CSV)",
                    data=classified_df.to_csv(index=False).encode("utf-8"),
                    file_name="full_classified_dataset.csv",
                    mime="text/csv",
                )

# ------------------------------------------------
# TAB 2: RAG QA (Local / Gemini) + MOF Recommender
# ------------------------------------------------
with tab2:
    st.header("🧪 Li–CO₂ RAG Question Answering")

    # LLM provider selection
    provider = st.selectbox(
        "Answer mode",
        ["Local only (no LLM)", "Gemini"],
        index=1,
    )

    question = st.text_area(
        "Ask about Li–CO₂ catalysts, mechanisms, design rules:",
        height=100,
    )

    # Retrieval config
    top_k = st.slider("Number of retrieved documents", 1, 12, 5)

    # Optional: filter by chunk type (RAG-skeleton style)
    retrieval_filter = st.selectbox(
        "Filter retrieved chunks by type",
        [
            "All",
            "Catalyst cards only",
            "Mechanisms only",
            "Design rules only",
            "Background only",
        ],
        index=0,
    )

    mode_map = {
        "All": "all",
        "Catalyst cards only": "catalyst",
        "Mechanisms only": "mechanism",
        "Design rules only": "design",
        "Background only": "background",
    }
    rag_mode = mode_map[retrieval_filter]

    if st.button("🔍 Retrieve & Answer", key="rag_answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # === RAG retrieval ===
            with st.spinner("Retrieving from RAG index..."):
                docs = retrieve_docs(question, top_k=top_k, mode=rag_mode)

            # === Show retrieved context ===
            st.subheader("📚 Retrieved context")
            for i, d in enumerate(docs, 1):
                meta = d.metadata or {}
                st.markdown(
                    f"**Doc {i} — {meta.get('chunk_type', 'unknown')}**"
                )
                text = d.page_content or ""
                st.write(text if len(text) < 1000 else text[:1000] + " …")
                st.markdown("---")

            # === Graph A: distribution of chunk types ===
            st.subheader("📊 Retrieved chunk types")
            counts = summarize_chunk_types(docs)

            if counts:
                types = list(counts.keys())
                values = [counts[t] for t in types]

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(types, values)
                ax.set_xlabel("Chunk type")
                ax.set_ylabel("Count in top-k")
                ax.set_title("Distribution of retrieved chunk types")
                ax.tick_params(axis="x", rotation=30, labelrotation=30)
                st.pyplot(fig)
            else:
                st.write("No chunks retrieved to summarize.")

            # === Graph B: true cosine similarity heatmap ===
            if docs:
                st.subheader("🎯 Relative relevance of retrieved docs")

                sims = compute_similarity_scores(question, docs)  # shape (k,)
                if sims.size > 0:
                    # Normalize to 0–1 for nicer visualization
                    sims_min, sims_max = float(sims.min()), float(sims.max())
                    if sims_max > sims_min:
                        sims_norm = (sims - sims_min) / (sims_max - sims_min)
                    else:
                        sims_norm = np.ones_like(sims)

                    fig2, ax2 = plt.subplots(figsize=(5, 1.5))
                    im = ax2.imshow([sims_norm], aspect="auto")
                    ax2.set_yticks([])
                    ax2.set_xticks(range(len(docs)))
                    ax2.set_xticklabels([f"D{i+1}" for i in range(len(docs))])
                    ax2.set_xlabel("Document (ranked by similarity)")
                    cbar = fig2.colorbar(im, ax=ax2)
                    cbar.set_label("Normalized cosine similarity")
                    st.pyplot(fig2)
                else:
                    st.write("Could not compute similarity scores.")

            # === Graph C: semantic scatter plot (query vs docs) ===
            if docs:
                st.subheader("🗺️ Semantic map of query and retrieved chunks")
                q_coord, doc_coords = project_embeddings(question, docs)

                if q_coord is not None and doc_coords is not None:
                    fig3, ax3 = plt.subplots(figsize=(5, 4))
                    # Plot docs
                    ax3.scatter(doc_coords[:, 0], doc_coords[:, 1], marker="o")
                    for i, (x, y) in enumerate(doc_coords):
                        ax3.text(x, y, f"D{i+1}", fontsize=8, ha="center", va="bottom")
                    # Plot query
                    ax3.scatter(q_coord[0], q_coord[1], marker="*", s=120)
                    ax3.text(
                        q_coord[0],
                        q_coord[1],
                        "Query",
                        fontsize=9,
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                    )
                    ax3.set_xlabel("PCA dim 1")
                    ax3.set_ylabel("PCA dim 2")
                    ax3.set_title("2D projection of query + retrieved chunks")
                    st.pyplot(fig3)
                else:
                    st.write("Not enough data to build semantic map.")

            # === Explanation of the graphs ===
            with st.expander("ℹ️ How to read these graphs"):
                st.markdown(
                    """
- **Chunk types bar chart**: shows how many of the retrieved chunks come from each knowledge category (design rules, mechanisms, catalyst cards, background, etc.).  
- **Relevance heatmap**: darker cells correspond to chunks with **higher cosine similarity** to the query, i.e. they are more relevant in embedding space.  
- **Semantic map**: shows the query and the retrieved chunks in a 2D projection of the embedding space.  
  Chunks closer to the "Query" point are more semantically similar to the question.
"""
                )

            # === Answer generation ===
            if provider == "Local only (no LLM)":
                st.subheader("🧠 Local Extractive Answer")
