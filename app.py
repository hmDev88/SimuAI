
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, nbformat, types, json

NOTEBOOK_PATH = "Catalyst.ipynb"
CSV_PATH = "Catalyst Database.csv"

@st.cache_resource
def load_nb_namespace(nb_path: str):
    """
    Execute code cells from a Jupyter notebook to obtain its functions in a namespace.
    Only executes code cells; ignores markdown.
    """
    ns = {}
    if not os.path.exists(nb_path):
        return ns
    nb = nbformat.read(nb_path, as_version=4)
    # Execute code cells, but be careful to skip magics or shell lines
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = cell.source
            # naive strip of magic commands
            clean_lines = []
            for line in src.splitlines():
                if line.strip().startswith("%") or line.strip().startswith("!"):
                    continue
                clean_lines.append(line)
            code = "\n".join(clean_lines)
            try:
                exec(compile(code, nb_path, "exec"), ns, ns)
            except Exception as e:
                # continue execution, but record the error for debugging
                ns.setdefault("__errors__", []).append(repr(e))
    return ns

@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

st.set_page_config(page_title="Catalyst Explorer", layout="wide")

st.title("🔬 Catalyst Explorer")
st.caption("Browse, filter, and parse catalyst entries from the uploaded database.")

# Load notebook functions (if available)
ns = load_nb_namespace(NOTEBOOK_PATH)
nb_errors = ns.get("__errors__", [])

# Load data
if not os.path.exists(CSV_PATH):
    st.error(f"Could not find CSV at: {CSV_PATH}")
    st.stop()

df = load_data(CSV_PATH)

with st.sidebar:
    st.header("Filters")
    # Text search fields
    txt_col = st.selectbox("Search in column", ["Catalyst","Paper_title","Authors","Journal","DOI","Measurement_conditions"], index=0)
    query = st.text_input("Contains text", "")
    # Numeric filters
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    selected_num = st.multiselect("Numeric filters", num_cols, default=num_cols)
    ranges = {}
    for c in selected_num:
        cmin = float(np.nanmin(df[c].values))
        cmax = float(np.nanmax(df[c].values))
        val = st.slider(c, min_value=cmin, max_value=cmax, value=(cmin, cmax))
        ranges[c] = val
    # Columns to show
    show_cols = st.multiselect("Columns to display", list(df.columns), default=list(df.columns))

# Apply filters
mask = pd.Series([True]*len(df))
if query:
    mask &= df[txt_col].fillna("").str.contains(query, case=False, na=False)

for c, (lo, hi) in ranges.items():
    mask &= df[c].between(lo, hi)

filtered = df.loc[mask, show_cols].copy()

st.subheader("📄 Table")
st.write(f"{len(filtered)}/{len(df)} rows")
st.dataframe(filtered, use_container_width=True)

# Detail pane
st.subheader("🔎 Details & Parsing")
row = st.selectbox("Select a row to parse its catalyst name", options=filtered.index.tolist())
if row is not None and len(filtered) > 0:
    cat_name = df.loc[row, "Catalyst"]
else:
    cat_name = ""

tab1, tab2, tab3 = st.tabs(["Parser", "Plot", "Raw JSON Preview"])

with tab1:
    st.markdown("**Catalyst name**")
    cat_input = st.text_input("Name", value=str(cat_name) if pd.notna(cat_name) else "", key="cat_text_input")

    # Wire notebook helpers if present
    normalize_text = ns.get("normalize_text")
    detect_type = ns.get("detect_type")
    extract_form = ns.get("extract_form")
    extract_metals = ns.get("extract_metals")
    canonicalize_basic = ns.get("canonicalize_basic")
    clean_metal = ns.get("clean_metal")
    infer_structure = ns.get("infer_structure")

    parsed = {}
    if cat_input:
        try:
            parsed["normalized"] = normalize_text(cat_input) if normalize_text else None
        except Exception as e:
            parsed["normalized_error"] = repr(e)
        try:
            parsed["type"] = detect_type(parsed.get("normalized", cat_input)) if detect_type else None
        except Exception as e:
            parsed["type_error"] = repr(e)
        try:
            parsed["form"] = extract_form(parsed.get("normalized", cat_input)) if extract_form else None
        except Exception as e:
            parsed["form_error"] = repr(e)
        try:
            parsed["metals"] = extract_metals(parsed.get("normalized", cat_input)) if extract_metals else None
        except Exception as e:
            parsed["metals_error"] = repr(e)
        try:
            parsed["structure"] = infer_structure(parsed.get("normalized", cat_input), current_struct="") if infer_structure else None
        except Exception as e:
            parsed["structure_error"] = repr(e)

    st.json(parsed if parsed else {"info": "Enter a catalyst name to parse."})

    if nb_errors:
        with st.expander("Notebook load warnings"):
            for err in nb_errors:
                st.code(err)

with tab2:
    st.markdown("**Quick plot**")
    if len(num_cols) >= 1:
        x_col = st.selectbox("X", options=num_cols, index=0)
        y_col = st.selectbox("Y (optional)", options=[None] + num_cols, index=0)
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

st.download_button(
    "Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_catalyst_data.csv",
    mime="text/csv"
)
