"""Small web interface for the DocQA retrieval pipeline."""
import os
from io import BytesIO
from pathlib import Path
from threading import RLock

from flask import Flask, render_template_string, request
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.embeddings import get_embeddings
from src.ingest import build_index
from src.rag import answer, load_store

INDEX_DIR = Path(os.getenv("DOCQA_INDEX_DIR", "index"))
DATA_DIR = os.getenv("DOCQA_DATA_DIR", "data")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

if not (INDEX_DIR / "index.faiss").is_file():
    build_index(DATA_DIR, str(INDEX_DIR), embedding="hash")

store = load_store(str(INDEX_DIR), "hash")
store_lock = RLock()
embeddings = get_embeddings("hash")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f4f2ec">
  <title>DocQA | Ask your documents</title>
  <style>
    :root { color-scheme: light; --ink:#182722; --muted:#58645e; --paper:#f4f2ec; --surface:#fffefa; --line:#d8ddd5; --green:#1c684e; --lime:#c9e273; --coral:#d77a5a; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(ellipse at 88% 4%, #e5ebd4 0, transparent 33%), var(--paper); color:var(--ink); font-family: Georgia, 'Times New Roman', serif; min-height:100vh; }
    header { height:68px; display:flex; align-items:center; justify-content:space-between; padding:0 max(24px, calc((100vw - 1080px)/2)); border-bottom:1px solid var(--line); font-family: 'Segoe UI', sans-serif; }
    .brand { font-size:17px; font-weight:700; letter-spacing:0; display:flex; align-items:center; gap:10px; }
    .mark { display:grid; place-items:center; width:28px; height:28px; background:var(--green); color:white; border-radius:7px; font:700 14px 'Segoe UI',sans-serif; }
    .status { color:var(--muted); font-size:13px; display:flex; gap:8px; align-items:center; }
    .dot { width:8px; height:8px; background:#78a95d; border-radius:50%; }
    main { width:min(100% - 40px, 860px); margin:72px auto 80px; }
    .eyebrow { color:var(--green); font:700 12px 'Segoe UI',sans-serif; text-transform:uppercase; letter-spacing:1.2px; }
    h1 { font-size:clamp(38px, 6vw, 60px); font-weight:500; line-height:1.04; margin:14px 0 12px; }
    .sub { color:var(--muted); font:16px/1.6 'Segoe UI',sans-serif; max-width:560px; margin:0 0 35px; }
    .workspace { display:grid; grid-template-columns:1fr 265px; gap:22px; align-items:start; }
    .panel { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:24px; }
    label { display:block; font:700 13px 'Segoe UI',sans-serif; margin-bottom:10px; }
    textarea { width:100%; min-height:106px; resize:vertical; border:1px solid #b9c4bb; background:#fff; border-radius:5px; padding:13px; font:16px/1.5 'Segoe UI',sans-serif; color:var(--ink); }
    textarea:focus, input:focus { outline:2px solid #8db79e; outline-offset:1px; }
    button { border:0; background:var(--green); color:white; padding:12px 17px; border-radius:5px; font:700 14px 'Segoe UI',sans-serif; cursor:pointer; }
    button:hover { background:#15533e; }
    .ask { display:flex; justify-content:flex-end; margin-top:12px; }
    .side-title { font:700 13px 'Segoe UI',sans-serif; margin:0 0 10px; }
    .side-copy { color:var(--muted); font:13px/1.55 'Segoe UI',sans-serif; margin:0 0 15px; }
    input[type=file] { display:block; width:100%; font:12px 'Segoe UI',sans-serif; color:var(--muted); margin:0 0 12px; }
    .upload button { background:#e8eddf; color:var(--ink); }
    .upload button:hover { background:#dbe5cc; }
    .notice { margin:18px 0 0; padding:12px 14px; border-left:3px solid var(--coral); background:#fff4ed; font:13px/1.5 'Segoe UI',sans-serif; white-space:pre-wrap; overflow-wrap:anywhere; }
    .answer { margin-top:22px; }
    .answer h2 { font-size:22px; font-weight:500; margin:0 0 12px; }
    .answer-body { white-space:pre-wrap; overflow-wrap:anywhere; font:15px/1.7 'Segoe UI',sans-serif; color:#27342d; }
    .empty { color:var(--muted); font:14px/1.6 'Segoe UI',sans-serif; }
    @media (max-width:650px) { main { margin-top:46px; } .workspace { grid-template-columns:1fr; } header { padding:0 20px; } .status { font-size:12px; } .panel { padding:19px; } }
  </style>
</head>
<body>
  <header><div class="brand"><span class="mark">D</span> DocQA</div><div class="status"><span class="dot"></span> Ready to retrieve</div></header>
  <main>
    <div class="eyebrow">Document question answering</div>
    <h1>Ask what your<br>documents know.</h1>
    <p class="sub">Search your files and get answers grounded in their text, with source references.</p>
    <div class="workspace">
      <section class="panel">
        <form method="post" action="{{ url_for('ask') }}">
          <label for="question">Your question</label>
          <textarea id="question" name="question" placeholder="Ask a question about the indexed documents..." required>{{ question }}</textarea>
          <div class="ask"><button type="submit">Ask DocQA</button></div>
        </form>
        {% if message %}<div class="notice">{{ message }}</div>{% endif %}
        {% if answer %}<div class="answer"><h2>Response</h2><div class="answer-body">{{ answer }}</div></div>{% elif not message %}<p class="empty">Answers are based only on retrieved document passages. Without an Anthropic API key, matching passages are shown directly.</p>{% endif %}
      </section>
      <aside class="panel">
        <p class="side-title">Add documents</p>
        <p class="side-copy">Upload a PDF, text, or Markdown file to include it in this session's search.</p>
        <form class="upload" method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
          <input type="file" name="document" accept=".pdf,.txt,.md" required>
          <button type="submit">Add to search</button>
        </form>
      </aside>
    </div>
  </main>
</body>
</html>"""


def render_page(question="", result="", message="", status=200):
    return render_template_string(PAGE, question=question, answer=result,
                                  message=message), status


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return render_page()


@app.post("/ask")
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        return render_page(message="Enter a question to search.", status=400)
    with store_lock:
        result, _ = answer(store, question)
    return render_page(question=question, result=result)


@app.post("/upload")
def upload():
    uploaded = request.files.get("document")
    if uploaded is None or not uploaded.filename:
        return render_page(message="Choose a PDF, TXT, or MD file to upload.", status=400)
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        return render_page(message="Only PDF, TXT, and MD files are supported.", status=400)

    filename = Path(uploaded.filename).name
    contents = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        return render_page(message="The upload exceeds the 10 MB limit.", status=413)
    try:
        if suffix == ".pdf":
            pages = PdfReader(BytesIO(contents)).pages
            documents = [Document(page_content=page.extract_text() or "",
                                  metadata={"source": filename, "page": page_number})
                         for page_number, page in enumerate(pages)]
        else:
            documents = [Document(page_content=contents.decode("utf-8"),
                                  metadata={"source": filename})]
    except Exception:
        return render_page(message="Could not read that file. Check that it is a valid PDF or UTF-8 text.", status=400)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return render_page(message="That file did not contain searchable text.", status=400)
    with store_lock:
        additions = FAISS.from_documents(chunks, embeddings)
        store.merge_from(additions)
    return render_page(message=f"Added {filename} to this session's search index.")


@app.errorhandler(413)
def upload_too_large(_error):
    return render_page(message="The upload exceeds the 10 MB limit.", status=413)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))