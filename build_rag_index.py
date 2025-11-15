# build_rag_index.py
import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_config import DATA_DIR, PERSIST_DIR, get_embedding_model

def clean_metadata(meta: dict) -> dict:
    """
    Ensure all metadata values are Chroma-friendly:
    only str, int, float, bool, or None.
    Lists/dicts -> JSON string; other types -> str().
    """
    clean = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (list, dict)):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)
    return clean



def load_json_docs(path: Path, chunk_type: str) -> List[Document]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for item in raw:
        text = item.get("text", "")
        meta = item.get("metadata", {}) or {}
        meta["chunk_type"] = meta.get("chunk_type", chunk_type)
        meta = clean_metadata(meta)
        docs.append(Document(page_content=text, metadata=meta))
    return docs



def load_jsonl_docs(path: Path, chunk_type: str) -> List[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            meta = obj.get("metadata", {}) or {}
            meta["chunk_type"] = meta.get("chunk_type", chunk_type)
            meta = clean_metadata(meta)
            docs.append(Document(page_content=text, metadata=meta))
    return docs



def main():
    all_docs: List[Document] = []

    # 1) Catalyst cards with ML labels
    cc_path = DATA_DIR / "RAG_Catalyst_Cards_with_ML.json"
    if cc_path.exists():
        all_docs.extend(load_json_docs(cc_path, "catalyst_card_with_ml"))

    # 2) Catalyst cards (no probabilities) - still useful as text
    no_proba_path = DATA_DIR / "RAG_Catalyst_Cards_with_ML_noProba.jsonl"
    if no_proba_path.exists():
        all_docs.extend(load_jsonl_docs(no_proba_path, "catalyst_card_with_ml_no_proba"))

    # 3) Mechanisms
    mech_path = DATA_DIR / "RAG_Mechanisms_LiCO2_50_bilingual.jsonl"
    if mech_path.exists():
        all_docs.extend(load_jsonl_docs(mech_path, "mechanism"))

    # 4) Design rules
    rules_path = DATA_DIR / "RAG_DesignRules_LiCO2_50_bilingual.jsonl"
    if rules_path.exists():
        all_docs.extend(load_jsonl_docs(rules_path, "design_rule"))

    # 5) Background
    bg_path = DATA_DIR / "RAG_Background_LiCO2_50_bilingual.jsonl"
    if bg_path.exists():
        all_docs.extend(load_jsonl_docs(bg_path, "background"))

    print(f"Loaded {len(all_docs)} raw documents")

    # Some docs are long; split them
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    split_docs = splitter.split_documents(all_docs)
    print(f"After splitting: {len(split_docs)} chunks")

    # Build Chroma DB
    embeddings = get_embedding_model()
    vs = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        collection_name="lico2_unified_rag",
        persist_directory=PERSIST_DIR,
    )
    vs.persist()
    print("✅ RAG index built and saved in:", PERSIST_DIR)


if __name__ == "__main__":
    main()
