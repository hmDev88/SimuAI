import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nbformat
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------
st.set_page_config(
    page_title="SimuAI – Li–CO₂ Catalyst Assistant",
    layout="wide",
    page_icon="🔋"
)

NOTEBOOK_PATH = "Catalyst.ipynb"
CSV_PATH = "data.csv"

# ------------------------------------------------
# INTERNAL RAG ENGINE (Replaces rag_config/rag_query)
# ------------------------------------------------
# We use TF-IDF to simulate embeddings and retrieval so the app 
# runs standalone without external API keys or heavy models.

@st.cache_resource
def get_rag_engine(df):
    """
    Builds a lightweight internal search engine using TF-IDF.
    """
    # 1. Create text representation of each row
    # We combine relevant columns to form the "Document" content
    docs = df.apply(lambda x: (
        f"Catalyst: {x.get('Catalyst ID', '')}, {x.get('Composition', '')}. "
        f"Method: {x.get('Synthesis Method', '')}. "
        f"Performance: {x.get('Performance', '')}. "
        f"Stats: {x.get('Conversion (%)', 0)}% conv, {x.get('Surface Area (m2/g)', 0)} m2/g."
    ), axis=1).tolist()
    
    # 2. Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    return vectorizer, tfidf_matrix, docs

class MockDoc:
    """Simulates a LangChain Document object"""
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

def retrieve_docs(question, top_k=5, mode="all"):
    """
    Retrieves documents using Cosine Similarity on TF-IDF vectors.
    """
    df = load_data()
    vectorizer, matrix, text_docs = get_rag_engine(df)
    
    # Query Vector
    query_vec = vectorizer.transform([question])
    
    # Similarity
    sims = cosine_similarity(query_vec, matrix).flatten()
    
    # Get top K indices
    top_indices = sims.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        # We simulate metadata for the charts
        row = df.iloc[idx]
        sim_score = sims[idx]
        
        # Assign a fake "chunk_type" based on data for the chart
        # In a real app, this comes from the source document structure
        perf = row.get('Performance', 'Average')
        if perf == 'Excellent': c_type = "catalyst_card"
        elif row.get('Surface Area (m2/g)', 0) > 1500: c_type = "mechanism" # Arbitrary rule for demo
        else: c_type = "background"
        
        if mode != "all" and mode not in c_type:
            continue
            
        results.append(MockDoc(
            content=text_docs[idx],
            metadata={"chunk_type": c_type, "score": sim_score}
        ))
        
    return results

def call_llm(question, docs):
    """
    Mock LLM response for demo purposes.
    """
    return (
        f"**Simulated Gemini Response:**\n\n"
        f"Based on the {len(docs)} retrieved records, the dataset contains several catalysts relevant to '{question}'. "
        f"Notably, **{docs[0].page_content.split(',')[0]}** appears to be a strong candidate.\n\n"
        f"*(Note: To enable real Generative AI, insert your Google Gemini API key in the code.)*"
    )

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
@st.cache_resource
def load_nb_namespace(nb_path: str):
    """
    Execute code cells from a notebook and return the namespace.
    Includes fallbacks if notebook is missing.
    """
    ns = {}
    if not os.path.exists(nb_path):
        ns["__errors__"] = [f"Notebook '{nb_path}' not found. Using fallback functions."]
        # define fallbacks so the UI doesn't break
        ns["normalize_text"] = lambda x: x.strip().upper()
        ns["detect_type"] = lambda x: "MOF" if "MOF" in x or "ZIF" in x else "Metal Oxide"
        ns["extract_form"] = lambda x: "Nanoparticle"
        ns["extract_metals"] = lambda x: ["Zr"] if "Zr" in x else ["Unknown"]
        ns["infer_structure"] = lambda x, y: "Crystalline"
        return ns

    try:
        nb = nbformat.read(nb_path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                src = "\n".join([line for line in cell.source.splitlines() if not line.strip().startswith(("%", "!"))])
                exec(compile(src, nb_path, "exec"), ns)
    except Exception as e:
        ns["__errors__"] = [repr(e)]
    return ns

@st.cache_data
def load_data():
    """Load main catalyst CSV."""
    if not os.path.exists(CSV_PATH):
        # Return dummy data if file missing to prevent crash
        return pd.DataFrame({
            'Catalyst': ['Test'], 'Composition': ['A'], 'Performance': ['Good'], 
            'Conversion (%)': [50], 'Surface Area (m2/g)': [100]
        })
    return pd.read_csv(CSV_PATH)

# ------------------------------------------------
# RAG VISUALIZATION HELPERS
# ------------------------------------------------
def summarize_chunk_types(docs):
    counts = Counter()
    for d in docs:
        counts[d.metadata.get("chunk_type", "unknown")] += 1
    return counts

def compute_similarity_scores(question, docs):
    # In our mock engine, we already computed these. 
    # We'll re-extract them from metadata or recompute if needed.
    return np.array([d.metadata.get("score", 0.5) for d in docs])

def project_embeddings(question, docs):
    """
    Uses the global vectorizer to project texts into 2D space for plotting.
    """
    if not docs: return None, None
    
    df = load_data()
    vectorizer, _, _ = get_rag_engine(df)
    
    texts = [question] + [d.page_content for d in docs]
    vecs = vectorizer.transform(texts).toarray()
    
    if vecs.shape[0] < 3: return None, None # Need enough points for PCA
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)
    
    return coords[0], coords[1:]

def build_extractive_answer(question, docs):
    if not docs: return "No documents found."
    out = ["### 🧠 Database Insights\n"]
    for d in docs[:3]:
        out.append(f"- **{d.metadata.get('chunk_type','Record').title()}**: {d.page_content}")
    return "\n".join(out)

# ------------------------------------------------
# MAIN APP UI
# ------------------------------------------------

st.title("SimuAI – Li–CO₂ Catalyst Assistant")

# Load logic
ns = load_nb_namespace(NOTEBOOK_PATH)
df = load_data()

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["⚙️ Catalyst Explorer & Trainer", "🧪 RAG QA (Local / Gemini)"])

# ------------------------------------------------
# TAB 1: Explorer & Trainer
# ------------------------------------------------
with tab1:
    st.caption("Filter catalysts, parse names, train models, and download classified results.")

    # Sidebar Filters
    with st.sidebar:
        st.header("Filters")
        # Handle case where no object columns exist
        obj_cols = df.select_dtypes("object").columns
        if len(obj_cols) > 0:
            txt_col = st.selectbox("Search column", obj_cols)
            query = st.text_input("Contains text")
        else:
            txt_col = None
            query = None

        num_cols = df.select_dtypes("number").columns
        selected_num = st.multiselect("Numeric filters", num_cols)
        ranges = {}
        for c in selected_num:
            cmin, cmax = float(df[c].min()), float(df[c].max())
            # Handle case where min == max
            if cmin == cmax:
                ranges[c] = (cmin, cmax)
            else:
                ranges[c] = st.slider(c, cmin, cmax, (cmin, cmax))
        
        show_cols = st.multiselect("Columns to show", list(df.columns), default=list(df.columns))

    # Apply Filters
    mask = pd.Series(True, index=df.index)
    if query and txt_col:
        mask &= df[txt_col].fillna("").str.contains(query, case=False)
    for c, (lo, hi) in ranges.items():
        mask &= df[c].between(lo, hi)
    
    filtered = df.loc[mask, show_cols].copy()
    
    st.subheader("📄 Filtered Table")
    st.write(f"{len(filtered)}/{len(df)} rows")
    st.dataframe(filtered, use_container_width=True)

    # Catalyst Parser Section
    st.subheader("🔎 Catalyst Parser")
    
    # Fallbacks are handled in load_nb_namespace, so we just retrieve them
    normalize_text = ns.get("normalize_text")
    detect_type = ns.get("detect_type")
    extract_form = ns.get("extract_form")
    extract_metals = ns.get("extract_metals")
    infer_structure = ns.get("infer_structure")

    # Try to find a sensible default column for catalyst names
    cat_col_candidates = [c for c in df.columns if "ID" in c or "Catalyst" in c or "Composition" in c]
    cat_col_name = cat_col_candidates[0] if cat_col_candidates else df.columns[0]

    row_list = filtered.index.tolist()
    if row_list:
        row_idx = st.selectbox("Select a row", options=row_list)
        cat_input_default = str(filtered.loc[row_idx, cat_col_name])
    else:
        cat_input_default = ""
    
    cat_input = st.text_input(f"Catalyst name (from column: {cat_col_name})", value=cat_input_default)

    parsed = {}
    if cat_input:
        try:
            parsed["normalized"] = normalize_text(cat_input)
            parsed["type"] = detect_type(parsed["normalized"])
            parsed["form"] = extract_form(parsed["normalized"])
            parsed["metals"] = extract_metals(parsed["normalized"])
            parsed["structure"] = infer_structure(parsed["normalized"], "")
        except Exception as e:
            parsed["error"] = str(e)

    st.json(parsed if parsed else {"info": "Enter or select a catalyst name"})
    
    if "__errors__" in ns:
        with st.expander("System Warnings"):
            for e in ns["__errors__"]:
                st.warning(e)

    # ML Trainer Section
    st.divider()
    st.subheader("⚙️ Train Machine Learning Model")
    
    if len(df.columns) < 2:
        st.error("Dataset needs at least 2 columns.")
    else:
        c1, c2 = st.columns(2)
        cols = list(df.columns)
        with c1:
            x_cols = st.multiselect("Feature columns (X)", cols[:-1], default=cols[3:-1] if len(cols)>4 else cols[:-1])
        with c2:
            # Try to auto-select 'Performance' as target
            default_y_idx = cols.index('Performance') if 'Performance' in cols else len(cols)-1
            y_col = st.selectbox("Target column (y)", cols, index=default_y_idx)

        if st.button("🚀 Train XGBoost Model"):
            if not x_cols:
                st.error("Select features!")
            else:
                # Data Prep
                X = df[x_cols].copy()
                y = df[y_col].copy()
                
                # Simple encoding
                X = pd.get_dummies(X)
                X = X.fillna(0)
                
                label_mapping = None
                if y.dtype == 'object':
                    le = {val: i for i, val in enumerate(y.unique())}
                    label_mapping = {i: val for val, i in le.items()}
                    y = y.map(le)
                
                # Train/Test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                st.success(f"Training Complete! Accuracy: {acc:.1%}")
                
                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    st.write(cm)
                
                # Download Predictions
                res_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                if label_mapping:
                    res_df["Actual"] = res_df["Actual"].map(label_mapping)
                    res_df["Predicted"] = res_df["Predicted"].map(label_mapping)
                
                st.download_button("Download Predictions", res_df.to_csv().encode('utf-8'), "preds.csv")

# ------------------------------------------------
# TAB 2: RAG QA
# ------------------------------------------------
with tab2:
    st.header("🧪 Li–CO₂ RAG Question Answering")
    
    question = st.text_area("Ask about Li–CO₂ catalysts (e.g., 'Which MOF has the best stability?')", height=100)
    
    if st.button("🔍 Retrieve & Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing database..."):
                # Use our internal retrieval function
                docs = retrieve_docs(question, top_k=5)
                
                # 1. Visualization: Similarity Heatmap
                st.subheader("🎯 Document Relevance")
                sims = compute_similarity_scores(question, docs)
                
                fig, ax = plt.subplots(figsize=(8, 1))
                im = ax.imshow([sims], aspect='auto', cmap='Greens', vmin=0, vmax=1)
                ax.set_yticks([])
                ax.set_xticks(range(len(docs)))
                ax.set_xticklabels([f"Doc {i+1}" for i in range(len(docs))])
                plt.colorbar(im, orientation='horizontal')
                st.pyplot(fig)
                
                # 2. Visualization: PCA Scatter
                st.subheader("🗺️ Semantic Map")
                q_coord, d_coords = project_embeddings(question, docs)
                if q_coord is not None:
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(d_coords[:,0], d_coords[:,1], label='Documents')
                    ax2.scatter(q_coord[0], q_coord[1], c='red', marker='*', s=200, label='Query')
                    for i, txt in enumerate(d_coords):
                        ax2.annotate(f"Doc {i+1}", (d_coords[i,0], d_coords[i,1]))
                    ax2.legend()
                    ax2.set_title("TF-IDF PCA Projection")
                    st.pyplot(fig2)
                
                # 3. Text Answer
                st.subheader("📝 Generated Insights")
                st.markdown(call_llm(question, docs))
                
                with st.expander("View Source Snippets"):
                    st.markdown(build_extractive_answer(question, docs))