"""Evaluate retrieval and log experiments with MLflow.

eval file format (JSON list): [{"question": "...", "keyword": "text that must
appear in a retrieved chunk"}]. Metric: hit rate@k = share of questions whose
top-k chunks contain the expected keyword.
"""
import argparse
import json
import time

import mlflow

from .ingest import build_index
from .rag import retrieve


def hit_rate(store, eval_set, k):
    hits = 0
    for item in eval_set:
        docs = retrieve(store, item["question"], k)
        if any(item["keyword"].lower() in d.page_content.lower() for d in docs):
            hits += 1
    return hits / len(eval_set)


def run(data, eval_path, chunk_size, chunk_overlap, k, embedding, experiment):
    eval_set = json.load(open(eval_path, encoding="utf-8"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                           "k": k, "embedding": embedding})
        t0 = time.time()
        store, n_docs, n_chunks = build_index(data, f"index_{chunk_size}",
                                              chunk_size, chunk_overlap, embedding)
        score = hit_rate(store, eval_set, k)
        mlflow.log_metrics({"hit_rate_at_k": score, "num_chunks": n_chunks,
                            "seconds": round(time.time() - t0, 2)})
        print(f"chunk_size={chunk_size} k={k} hit_rate@k={score:.2f} chunks={n_chunks}")
    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--eval", default="eval.json")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[400, 800, 1200])
    ap.add_argument("--chunk-overlap", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--embedding", default="hash", choices=["hf", "hash"])
    ap.add_argument("--experiment", default="docqa-rag")
    a = ap.parse_args()
    for cs in a.chunk_sizes:
        run(a.data, a.eval, cs, a.chunk_overlap, a.k, a.embedding, a.experiment)
