"""Command-line chatbot over your indexed documents."""
import argparse

from src.rag import answer, load_store

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="index")
ap.add_argument("--embedding", default="hf", choices=["hf", "hash"])
ap.add_argument("--k", type=int, default=4)
a = ap.parse_args()

store = load_store(a.index, a.embedding)
print("DocQA ready. Ask a question (or 'exit').")
while True:
    try:
        q = input("\n> ").strip()
    except EOFError:
        break
    if q.lower() in {"exit", "quit", ""}:
        break
    text, _ = answer(store, q, a.k)
    print("\n" + text)
