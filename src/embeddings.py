"""Embedding backends.

"hash" : deterministic bag-of-words hashing embeddings. No model download;
         retrieval quality is lower than "hf".
"hf"   : sentence-transformers model (requires optional packages and download).
"""
import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HashEmbeddings(Embeddings):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings(kind: str = "hash") -> Embeddings:
    if kind == "hash":
        return HashEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=DEFAULT_HF_MODEL)
