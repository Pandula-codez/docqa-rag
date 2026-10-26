"""Retrieval + answer generation."""
import os

from langchain_community.vectorstores import FAISS

from .embeddings import get_embeddings

MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

PROMPT = """Answer the question using ONLY the context below. If the answer is not
in the context, say you don't know. Cite sources as [source, page].

Context:
{context}

Question: {question}"""


def load_store(index_dir="index", embedding="hf"):
    return FAISS.load_local(index_dir, get_embeddings(embedding),
                            allow_dangerous_deserialization=True)


def retrieve(store, question, k=4):
    return store.similarity_search(question, k=k)


def _fmt(doc):
    src = os.path.basename(doc.metadata.get("source", "?"))
    page = doc.metadata.get("page")
    tag = f"{src}, p.{page + 1}" if isinstance(page, int) else src
    return f"[{tag}]\n{doc.page_content}"


def answer(store, question, k=4):
    docs = retrieve(store, question, k)
    context = "\n\n".join(_fmt(d) for d in docs)
    if not os.getenv("ANTHROPIC_API_KEY"):
        return ("(No ANTHROPIC_API_KEY set: showing the retrieved passages only)\n\n"
                + context), docs
    import anthropic

    client = anthropic.Anthropic()
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=600,
            messages=[{"role": "user",
                       "content": PROMPT.format(context=context, question=question)}])
    except anthropic.APIError as exc:
        return (f"(Anthropic API request failed: {exc}. "
                "Showing the retrieved passages only.)\n\n" + context), docs
    return msg.content[0].text, docs
