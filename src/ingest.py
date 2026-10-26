"""Load documents, split into chunks, embed and store in a FAISS index."""
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings


def load_documents(data_dir: str):
    docs = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def build_index(data_dir="data", index_dir="index", chunk_size=800,
                chunk_overlap=100, embedding="hf"):
    docs = load_documents(data_dir)
    if not docs:
        raise SystemExit(f"No .pdf/.txt/.md files found in '{data_dir}'.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings(embedding))
    store.save_local(index_dir)
    return store, len(docs), len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--index", default="index")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=100)
    ap.add_argument("--embedding", default="hf", choices=["hf", "hash"])
    a = ap.parse_args()
    _, n_docs, n_chunks = build_index(a.data, a.index, a.chunk_size,
                                      a.chunk_overlap, a.embedding)
    print(f"Indexed {n_docs} pages/files into {n_chunks} chunks -> {a.index}/")
