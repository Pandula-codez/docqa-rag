from src.evaluate import hit_rate
from src.ingest import build_index
import json


def test_index_and_retrieval(tmp_path):
    store, n_docs, n_chunks = build_index("data", str(tmp_path / "idx"), 300, 50, "hash")
    assert n_docs >= 1 and n_chunks >= 2
    items = json.load(open("eval.json", encoding="utf-8"))
    assert hit_rate(store, items, 4) >= 0.8
