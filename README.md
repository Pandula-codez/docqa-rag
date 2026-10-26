# DocQA: PDF Question-Answering Chatbot (RAG)

Ask questions about your own PDFs and get answers grounded in the source text, with citations.
Built with **LangChain**, **FAISS** vector search, a Claude LLM and **MLflow** experiment tracking.

## How it works
1. **Ingest:** PDFs / text files are loaded, split into overlapping chunks and embedded with `all-MiniLM-L6-v2`.
2. **Index:** embeddings are stored in a FAISS index on disk.
3. **Retrieve + generate:** the top-k chunks for a question are passed to the LLM, which answers only from that context and cites `[file, page]`.
4. **Evaluate:** `src/evaluate.py` measures retrieval hit rate@k on a small question set and logs every run (chunk size, overlap, k, embedding, metrics) to MLflow so configurations can be compared.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put your PDFs in data/ (a sample notes file is included)
python -m src.ingest --data data --index index

export ANTHROPIC_API_KEY=your_key   # Windows: set ANTHROPIC_API_KEY=your_key
python app.py
```
Without an API key the app still works and shows the retrieved passages.

## Experiment tracking
```bash
python -m src.evaluate --chunk-sizes 400 800 1200 --k 4
mlflow ui          # open http://localhost:5000 and compare runs
```
Edit `eval.json` with questions about your own documents and a keyword that must appear in the right chunk.

## Tests
```bash
pytest -q          # offline test using lightweight hash embeddings
```

## Project structure
```
src/embeddings.py   embedding backends (HuggingFace, offline hash)
src/ingest.py       load, chunk, embed, save FAISS index
src/rag.py          retrieval and answer generation
src/evaluate.py     hit-rate evaluation + MLflow logging
app.py              command-line chatbot
tests/              pipeline test
```

## Results
_Add your MLflow results here after running on your own documents (best chunk size, hit rate@k)._
