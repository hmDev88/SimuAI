import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, io, nbformat, json, types
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# ------------------------------------------------
# Paths
# ------------------------------------------------
NOTEBOOK_PATH = "Catalyst.ipynb"
CSV_PATH = "Catalyst Database.csv"

# ------------------------------------------------
# Helpers
# ------------------------------------------------
@st.cache_resource
def load_nb_namespace(nb_path: str):
    """Executes code cells from a notebook and returns its namespace (used for parsing)."""
    ns = {}
    if not os.path.exists(nb_path):
        ns.setdefault("__errors__", []).append(f"Notebook not found at: {nb_path}")
        return ns
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "\n".join(
                [line for line in cell.source.splitlines() if not line.strip().startswith(("%", "!"))]
            )
            try:
                exec(compile(src, nb_path, "exec"), ns)
            except Exception as e:
                ns.setdefault("__errors__", []).append(repr(e))
    return ns


@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)


# ------------------------------------------------
# Page setup
# ------------------------------------------------
st.set_page_config(page_title="Catalyst Explorer & Trainer", layout="wide")
st.title("🔬 Catalyst Explorer & Trainer")
st.caption("Filter catalysts, parse names, train models, and evaluate accuracy.")

# ------------------------------------------------
# Load notebook functions (for parsing)
# ------------------------------------------------
ns = load_nb_namespace(NOTEBOOK_PATH)
nb_errors = ns.get("__errors__", [])

# ------------------------------------------------
# Load data
# ------------------------------------------------
if not os.path.exists(CSV_PATH):
    st.error(f"Missing data file: {CSV_PATH}")
    st.stop()

df = load_data(CSV_PATH)

# ------------------------------------------------
# Sidebar: Filters
# ------------------------------------------------
with st.sidebar:
    st.header("Filters")
    txt_col = st.selectbox("Search column", df.select_dtypes("object").columns)
    query = st.text_input("Contains text")
    num_cols = df.select_dtypes("number").columns
    selected_num = st.multiselect("Numeric filters", num_cols, default=list(num_cols))
    ranges = {}
    for c in selected_num:
        cmin, cmax = float(df[c].min()), float(df[c].max())
        ranges[c] = st.slider(c, cmin, cmax, (cmin, cmax))
    show_cols = st.multiselect("Columns to show", list(df.columns), default=list(df.columns))

mask = pd.Series(True, index=df.index)
if query:
    mask &= df[txt_col].fillna("").str.contains(query, case=False)
for c, (lo, hi) in ranges.items():
    mask &= df[c].between(lo, hi)
filtered = df.loc[mask, show_cols].copy()

st.subheader("📄 Filtered Table")
st.write(f"{len(filtered)}/{len(df)} rows")
st.dataframe(filtered, use_container_width=True)

# ------------------------------------------------
# Parsing
# ------------------------------------------------
st.subheader("🔎 Catalyst Parser")

normalize_text = ns.get("normalize_text")
detect_type = ns.get("detect_type")
extract_form = ns.get("extract_form")
extract_metals = ns.get("extract_metals")
infer_structure = ns.get("infer_structure")

row_list = filtered.index.tolist()
row = st.selectbox("Select a row", options=row_list) if len(row_list) > 0 else None
cat_input = ""
if row is not None and "Catalyst" in filtered.columns:
    cat_input = filtered.loc[row, "Catalyst"]
cat_input = st.text_input("Catalyst name", value=cat_input)

parsed = {}
if cat_input:
    try:
        parsed["normalized"] = normalize_text(cat_input) if normalize_text else cat_input
        parsed["type"] = detect_type(parsed["normalized"]) if detect_type else None
        parsed["form"] = extract_form(parsed["normalized"]) if extract_form else None
        parsed["metals"] = extract_metals(parsed["normalized"]) if extract_metals else None
        parsed["structure"] = (
            infer_structure(parsed["normalized"], "") if infer_structure else None
        )
    except Exception as e:
        parsed["error"] = repr(e)
st.json(parsed if parsed else {"info": "Enter or select a catalyst name"})

if nb_errors:
    with st.expander("Notebook load warnings"):
        for e in nb_errors:
            st.code(e)

# ------------------------------------------------
# Model Trainer
# ------------------------------------------------
st.subheader("⚙️ Train Machine Learning Model")

st.markdown(
    """
Train a simple **XGBoost classifier** on your dataset to predict catalyst performance.
Select the input features (X) and target column (y).  
"""
)

cols = list(df.columns)
if len(cols) < 2:
    st.error("Dataset must have at least 2 columns.")
else:
    c1, c2 = st.columns(2)
    with c1:
        x_cols = st.multiselect("Feature columns (X)", cols[:-1], default=cols[:-1])
    with c2:
        y_col = st.selectbox("Target column (y)", cols, index=len(cols) - 1)

    test_size = st.slider("Test size (%)", 10, 50, 20) / 100.0

    if st.button("🚀 Train Model"):
        if len(x_cols) == 0 or y_col not in df.columns:
            st.warning("Please select valid features and target.")
        else:
            X = df[x_cols]
            y = df[y_col]

            # Handle missing values
            X = X.fillna(0)
            if y.dtype == "object":
                y = y.astype("category").cat.codes

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")

            st.success(f"✅ Training complete! Accuracy: **{acc*100:.2f}%**, Macro F1: **{f1m*100:.2f}%**")

            cm = confusion_matrix(y_test, y_pred)
            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            st.pyplot(fig)

            st.markdown("### Classification Report")
            report = classification_report(y_test, y_pred, digits=3)
            st.code(report, language="text")

            # Allow download of predictions
            res_df = pd.DataFrame(
                {"y_true": y_test, "y_pred": y_pred}
            ).reset_index(drop=True)
            st.download_button(
                "Download predictions CSV",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="model_predictions.csv",
                mime="text/csv",
            )
