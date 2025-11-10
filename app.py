import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, nbformat, types, json, os

NOTEBOOK_PATH = "Catalyst.ipynb"
CSV_PATH = "Catalyst Database.csv"

# ----------------------------
# Data & Notebook loaders
# ----------------------------
@st.cache_resource
def load_nb_namespace(nb_path: str):
    """
    Execute code cells from a Jupyter notebook to obtain its functions in a namespace.
    Skips magics/shell lines. Collects any execution errors in __errors__.
    """
    ns = {}
    if not os.path.exists(nb_path):
        ns.setdefault("__errors__", []).append(f"Notebook not found at path: {nb_path}")
        return ns

    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = cell.source
            clean_lines = []
            for line in src.splitlines():
                # Skip IPython magics or shell commands
                if line.strip().startswith("%") or line.strip().startswith("!"):
                    continue
                clean_lines.append(line)
            code = "\n".join(clean_lines)
            try:
                exec(compile(code, nb_path, "exec"), ns, ns)
            except Exception as e:
                ns.setdefault("__errors__", []).append(repr(e))
    return ns

@st.cache_data
def load_data(csv_path: str):
    return pd.read_csv(csv_path)

# ----------------------------
# Page config & Title
# ----------------------------
st.set_page_config(page_title="Catalyst Explorer", layout="wide")
st.title("🔬 Catalyst Explorer")
st.caption("Browse, filter, parse, and export catalyst entries from your dataset.")

# ----------------------------
# Load notebook functions
# ----------------------------
ns = load_nb_namespace(NOTEBOOK_PATH)
nb_errors = ns.get("__errors__", [])

# ----------------------------
# Load CSV (with convenience reload)
# ----------------------------
top_buttons = st.columns([1, 1, 8])
with top_buttons[0]:
    if st.button("🔄 Reload CSV"):
        load_data.clear()
        st.rerun()
with top_buttons[1]:
    st.write("")  # spacer

if not os.path.exists(CSV_PATH):
    st.error(f"Could not find CSV at: {CSV_PATH}")
    st.stop()

df = load_data(CSV_PATH)

# ----------------------------
# Sidebar Filters
# ----------------------------
with st.sidebar:
    st.header("Filters")
    text_cols = ["Catalyst", "Paper_title", "Authors", "Journal", "DOI", "Measurement_conditions"]
    existing_text_cols = [c for c in text_cols if c in df.columns]
    if not existing_text_cols:
        existing_text_cols = [c for c in df.columns if df[c].dtype == "object"]

    txt_col = st.selectbox("Search in column", options=existing_text_cols, index=0)
    query = st.text_input("Contains text", "")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    selected_num = st.multiselect("Numeric filters", num_cols, default=num_cols)

    ranges = {}
    for c in selected_num:
        # Handle all-nan columns safely
        series = df[c].dropna()
        if len(series) == 0:
            continue
        cmin = float(np.nanmin(series.values))
        cmax = float(np.nanmax(series.values))
        if cmin == cmax:
            # Single-valued column: show a disabled-like slider range
            val = st.slider(c, min_value=cmin, max_value=cmax, value=(cmin, cmax))
        else:
            val = st.slider(c, min_value=cmin, max_value=cmax, value=(cmin, cmax))
        ranges[c] = val

    show_cols = st.multiselect("Columns to display", list(df.columns), default=list(df.columns))

# ----------------------------
# Apply Filters
# ----------------------------
mask = pd.Series([True] * len(df), index=df.index)

if query and txt_col in df.columns:
    mask &= df[txt_col].fillna("").str.contains(query, case=False, na=False)

for c, (lo, hi) in ranges.items():
    if c in df.columns:
        mask &= df[c].between(lo, hi)

filtered = df.loc[mask, show_cols].copy()

st.subheader("📄 Table")
st.write(f"{len(filtered)}/{len(df)} rows")
st.dataframe(filtered, use_container_width=True)

# ----------------------------
# Details & Parsing (with synced input)
# ----------------------------
st.subheader("🔎 Details & Parsing")

idx_options = filtered.index.tolist()
if len(idx_options) == 0:
    st.info("No rows to select. Adjust your filters to see items here.")
    row = None
else:
    # Keep text input in sync with selected row
    def sync_name_from_row():
        r = st.session_state.get("row_select", None)
        if r is not None and "Catalyst" in df.columns:
            val = df.loc[r, "Catalyst"]
            st.session_state["cat_text_input"] = str(val) if pd.notna(val) else ""

    row = st.selectbox(
        "Select a row to parse its catalyst name",
        options=idx_options,
        key="row_select",
        on_change=sync_name_from_row
    )

# Initialize text field on first load
if "cat_text_input" not in st.session_state:
    if row is not None and "Catalyst" in df.columns:
        st.session_state["cat_text_input"] = str(df.loc[row, "Catalyst"]) if pd.notna(df.loc[row, "Catalyst"]) else ""
    else:
        st.session_state["cat_text_input"] = ""

st.markdown("**Catalyst name**")
cat_input = st.text_input("Name", key="cat_text_input")

tab1, tab2, tab3 = st.tabs(["Parser", "Plot", "Raw JSON Preview"])

with tab1:
    normalize_text = ns.get("normalize_text")
    detect_type = ns.get("detect_type")
    extract_form = ns.get("extract_form")
    extract_metals = ns.get("extract_metals")
    infer_structure = ns.get("infer_structure")

    parsed = {}
    if cat_input:
        try:
            parsed["normalized"] = normalize_text(cat_input) if normalize_text else None
        except Exception as e:
            parsed["normalized_error"] = repr(e)

        try:
            base = parsed.get("normalized", cat_input)
            parsed["type"] = detect_type(base) if detect_type else None
        except Exception as e:
            parsed["type_error"] = repr(e)

        try:
            base = parsed.get("normalized", cat_input)
            parsed["form"] = extract_form(base) if extract_form else None
        except Exception as e:
            parsed["form_error"] = repr(e)

        try:
            base = parsed.get("normalized", cat_input)
            parsed["metals"] = extract_metals(base) if extract_metals else None
        except Exception as e:
            parsed["metals_error"] = repr(e)

        try:
            base = parsed.get("normalized", cat_input)
            parsed["structure"] = infer_structure(base, current_struct="") if infer_structure else None
        except Exception as e:
            parsed["structure_error"] = repr(e)

    st.json(parsed if parsed else {"info": "Enter a catalyst name to parse."})

    if nb_errors:
        with st.expander("Notebook load warnings"):
            for err in nb_errors:
                st.code(err)

with tab2:
    st.markdown("**Quick plot**")
    if len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]) >= 1:
        num_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        x_col = st.selectbox("X", options=num_cols_all, index=0)
        y_col = st.selectbox("Y (optional)", options=[None] + num_cols_all, index=0)

        fig, ax = plt.subplots()
        if y_col is None:
            ax.hist(filtered[x_col].dropna().values, bins=30)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {x_col}")
        else:
            ax.scatter(filtered[x_col].values, filtered[y_col].values)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig)
    else:
        st.info("No numeric columns available to plot.")

with tab3:
    st.json({
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "example_rows": df.head(3).to_dict(orient="records")
    })

# ----------------------------
# Download (filtered view)
# ----------------------------
st.download_button(
    "Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_catalyst_data.csv",
    mime="text/csv"
)
