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
    """Executes code cells from a notebook and returns its namespace."""
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
st.caption("Filter catalysts, parse names, train models, and download classified results.")

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
# Model Trainer (robust encoding + full classified output)
# ------------------------------------------------
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
        x_cols = st.multiselect("Feature columns (X)", cols[:-1], default=cols[:-1])
    with c2:
        y_col = st.selectbox("Target column (y)", cols, index=len(cols) - 1)

    test_size = st.slider("Test size (%)", 10, 50, 20) / 100.0
    max_depth = st.slider("max_depth", 2, 12, 6)
    n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
    learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)

    if st.button("🚀 Train Model"):
        if len(x_cols) == 0 or y_col not in df.columns:
            st.warning("Please select valid features and target.")
        else:
            X_raw = df[x_cols].copy()
            y_raw = df[y_col].copy()

            # --- Clean X ---
            for c in X_raw.columns:
                if pd.api.types.is_bool_dtype(X_raw[c]):
                    X_raw[c] = X_raw[c].astype(int)
            X = pd.get_dummies(X_raw, dummy_na=True)
            X = X.fillna(0)

            # --- Clean y ---
            y = y_raw.copy()
            label_mapping = None
            if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
                y = y.astype("category")
                label_mapping = {k: v for v, k in enumerate(y.cat.categories)}
                y = y.cat.codes

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
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

            # --- Evaluate ---
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
            im = ax.imshow(cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            st.pyplot(fig)

            st.markdown("### Classification Report")
            if label_mapping:
                inv_map = {v: k for k, v in label_mapping.items()}
                y_test_named = pd.Series(y_test).map(inv_map).astype(str)
                y_pred_named = pd.Series(y_pred).map(inv_map).astype(str)
                report = classification_report(y_test_named, y_pred_named, digits=3)
            else:
                report = classification_report(y_test, y_pred, digits=3)
            st.code(report, language="text")

            # --- Download test predictions ---
            out = pd.DataFrame({
                "y_true": y_test if label_mapping is None else pd.Series(y_test).map(inv_map),
                "y_pred": y_pred if label_mapping is None else pd.Series(y_pred).map(inv_map),
            }).reset_index(drop=True)
            st.download_button(
                "Download Test Predictions (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="model_predictions.csv",
                mime="text/csv",
            )

            # --- Predict full dataset and download full classified CSV ---
            y_all_pred = model.predict(X)
            if label_mapping:
                inv_map = {v: k for k, v in label_mapping.items()}
                y_all_pred = pd.Series(y_all_pred).map(inv_map)

            classified_df = df.copy()
            classified_df["Predicted_Label"] = y_all_pred

            st.markdown("### 📁 Full Classified Dataset")
            st.dataframe(classified_df.head(), use_container_width=True)

            st.download_button(
                "⬇️ Download Full Classified Data (CSV)",
                data=classified_df.to_csv(index=False).encode("utf-8"),
                file_name="full_classified_dataset.csv",
                mime="text/csv",
            )

# ============================================================
# Extra tools: Explore & filter + simple recommendation agent
# ============================================================

st.markdown("---")
st.header("🔍 Explore & filter catalysts")

# work on a copy of the original dataframe
df_view = df.copy()

# --- auto-detect key columns by name ---
lower_cols = {c.lower(): c for c in df_view.columns}

def find_col(substrings):
    for lc, orig in lower_cols.items():
        if any(s in lc for s in substrings):
            return orig
    return None

cap_col    = find_col(["capacity", "mah"])         # e.g. Discharge_capacity_mAh_g
over_col   = find_col(["overpot", "overpotential"])
delta_col  = find_col(["adsorption", "deltae"])
barrier_col= find_col(["barrier"])

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

# Capacity slider
if cap_col is not None and pd.api.types.is_numeric_dtype(df_view[cap_col]):
    with col_f1:
        cap_min = float(df_view[cap_col].min())
        cap_max = float(df_view[cap_col].max())
        min_cap = st.slider(f"Min {cap_col}", cap_min, cap_max, cap_min)
else:
    min_cap = None

# Overpotential slider
if over_col is not None and pd.api.types.is_numeric_dtype(df_view[over_col]):
    with col_f2:
        over_min = float(df_view[over_col].min())
        over_max = float(df_view[over_col].max())
        max_over = st.slider(f"Max {over_col}", over_min, over_max, over_max)
else:
    max_over = None

# ΔE adsorption slider
if delta_col is not None and pd.api.types.is_numeric_dtype(df_view[delta_col]):
    with col_f3:
        dmin = float(df_view[delta_col].min())
        dmax = float(df_view[delta_col].max())
        max_delta = st.slider(f"Max {delta_col}", dmin, dmax, dmax)
else:
    max_delta = None

# Reaction barrier slider
if barrier_col is not None and pd.api.types.is_numeric_dtype(df_view[barrier_col]):
    with col_f4:
        bmin = float(df_view[barrier_col].min())
        bmax = float(df_view[barrier_col].max())
        max_barrier = st.slider(f"Max {barrier_col}", bmin, bmax, bmax)
else:
    max_barrier = None

# Apply filters
filt = df_view.copy()
if min_cap is not None and cap_col in filt.columns:
    filt = filt[filt[cap_col] >= min_cap]
if max_over is not None and over_col in filt.columns:
    filt = filt[filt[over_col] <= max_over]
if max_delta is not None and delta_col in filt.columns:
    filt = filt[filt[delta_col] <= max_delta]
if max_barrier is not None and barrier_col in filt.columns:
    filt = filt[filt[barrier_col] <= max_barrier]

st.caption(f"{len(filt)} / {len(df_view)} rows match the current filters.")
st.dataframe(filt.head(30), use_container_width=True)

# ------------------------------------------------------------
# Agent-style recommendations (rule-based, no API)
# ------------------------------------------------------------
st.markdown("## 🤖 Agent-style recommendations (rule based, no API)")

def simple_agent(query: str, df_in: pd.DataFrame):
    q = query.lower()
    d = df_in.copy()
    expl = []

    # high capacity
    if cap_col and cap_col in d.columns and ("high capacity" in q or "large capacity" in q):
        thr = d[cap_col].quantile(0.75)
        d = d[d[cap_col] >= thr]
        expl.append(f"{cap_col} ≥ {thr:.0f}")

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
    ):
        thr = d[delta_col].quantile(0.25)
        d = d[d[delta_col] <= thr]
        expl.append(f"{delta_col} ≤ {thr:.2f}")

    # low barrier
    if barrier_col and barrier_col in d.columns and ("low barrier" in q or "small barrier" in q):
        thr = d[barrier_col].quantile(0.25)
        d = d[d[barrier_col] <= thr]
        expl.append(f"{barrier_col} ≤ {thr:.2f}")

    # final ranking: by capacity if we have it, otherwise leave as-is
    if cap_col and cap_col in d.columns and pd.api.types.is_numeric_dtype(d[cap_col]):
        d = d.sort_values(cap_col, ascending=False)

    return d, expl

user_q = st.text_area(
    "Describe what you want the system to find:",
    "High capacity and low overpotential catalysts with low reaction barrier.",
)

if st.button("Run recommendation agent"):
    recs, expl = simple_agent(user_q, df_view)
    if expl:
        st.write("Applied filters: " + "; ".join(expl))
    else:
        st.write("No specific keywords detected, showing top rows.")
    st.dataframe(recs.head(15), use_container_width=True)
