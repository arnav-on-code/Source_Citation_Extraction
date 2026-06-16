# 📚 Citation RAG — Source-Grounded QA with Provenance Tracking

A production-ready Retrieval-Augmented Generation system that **cites every answer** using LangChain's structured output approach. Every factual claim in an answer is backed by a verbatim quote, document title, author, and estimated page number.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend  (:8080)                               │
│                                                         │
│  1. ChromaDB retrieval (HuggingFace embeddings)         │
│     → top-k chunks WITH similarity scores              │
│                                                         │
│  2. Format context: [Source 1] ... [Source N]           │
│     (metadata injected: title, author, page, filename)  │
│                                                         │
│  3. LLM call with .with_structured_output(CitedAnswer)  │
│     → forces Pydantic schema validation                 │
│                                                         │
│  4. Return CitedAnswer + RetrievedChunks (provenance)   │
└─────────────────────────────────────────────────────────┘
     │
     ▼
Streamlit UI (:8501)  or  CLI  or  REST API
```

## Data Sources (Free / Public Domain)

All documents are from [Project Gutenberg](https://www.gutenberg.org/) — completely free.

| File | Title | Author | ~Pages | URL |
|------|-------|--------|--------|-----|
| `federalist_papers.txt` | The Federalist Papers | Hamilton, Madison, Jay | ~200 | [link](https://www.gutenberg.org/files/1404/1404-0.txt) |
| `origin_of_species.txt` | On the Origin of Species | Charles Darwin | ~500 | [link](https://www.gutenberg.org/files/1228/1228-0.txt) |
| `art_of_war.txt` | The Art of War | Sun Tzu | ~100 | [link](https://www.gutenberg.org/files/132/132-0.txt) |
| `meditations.txt` | Meditations | Marcus Aurelius | ~150 | [link](https://www.gutenberg.org/files/2680/2680-0.txt) |
| `the_republic.txt` | The Republic | Plato | ~400 | [link](https://www.gutenberg.org/files/1497/1497-0.txt) |

**Total: ~1,350 pages** of public domain text.

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- An API key for **Anthropic** (recommended) or **OpenAI**

### 2. Clone & configure

```bash
git clone <repo>
cd citation-rag
cp .env.example .env
# Edit .env — set LLM_PROVIDER and your API key
```

### 3. Download the data

```bash
bash data/download_data.sh
```

This downloads ~6 MB of public domain texts into `data/raw/`.

### 4. Start services

```bash
# Start ChromaDB + Backend + Frontend
docker compose up -d

# Run ingestion (one-time, splits and embeds all documents)
docker compose --profile ingest run ingest
```

### 5. Open the UI

- **Streamlit UI**: http://localhost:8501
- **API docs (Swagger)**: http://localhost:8080/docs
- **Health check**: http://localhost:8080/health

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| `chromadb` | 8000 | Vector store (persisted in Docker volume) |
| `backend` | 8080 | FastAPI — query endpoint |
| `frontend` | 8501 | Streamlit UI |
| `ingest` | — | One-shot ingestion job (profile: `ingest`) |

---

## API Reference

### `POST /query`

```json
{
  "question": "What does Sun Tzu say about deception?",
  "top_k": 5
}
```

**Response:**

```json
{
  "question": "What does Sun Tzu say about deception?",
  "cited_answer": {
    "answer": "Sun Tzu considers deception the foundation of all warfare [1]...",
    "citations": [
      {
        "source_id": 1,
        "quote": "All warfare is based on deception.",
        "relevance": "Directly states the core principle being asked about."
      }
    ],
    "confidence": "high"
  },
  "retrieved_chunks": [
    {
      "chunk_id": 1,
      "content": "...",
      "similarity_score": 0.8921,
      "metadata": {
        "filename": "art_of_war.txt",
        "title": "The Art of War",
        "author": "Sun Tzu",
        "chunk_index": 34,
        "page_estimate": 14,
        "total_chunks": 87,
        "source_url": "https://www.gutenberg.org/files/132/132-0.txt"
      }
    }
  ],
  "model_used": "claude-3-5-haiku-20241022",
  "collection_name": "citation_rag"
}
```

---

## Project Structure

```
citation-rag/
├── docker-compose.yml
├── requirements.txt
├── .env.example
│
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
├── app/
│   ├── config.py                  # env-based settings
│   ├── schemas/
│   │   └── __init__.py            # CitedAnswer, Citation, QueryResponse (Pydantic)
│   ├── ingestion/
│   │   └── ingest.py              # document loading, chunking, embedding
│   ├── chains/
│   │   └── citation_chain.py      # RAG chain with structured output
│   └── api/
│       └── main.py                # FastAPI routes
│
├── frontend/
│   └── app.py                     # Streamlit UI
│
├── data/
│   ├── download_data.sh           # downloads 5 public domain books
│   └── raw/                       # .txt files go here
│
├── scripts/
│   └── query_cli.py               # CLI for testing without UI
│
└── tests/
    └── test_schemas.py            # pytest tests for Pydantic schemas
```

---

## Key Design Decisions

### Structured Output (not prompting)

Following the [LangChain citation guide](https://python.langchain.com/docs/how_to/qa_citations/), citations are extracted via `llm.with_structured_output(CitedAnswer)` — not by parsing free-text. This means:

- The LLM **must** return a valid `CitedAnswer` or the call fails
- No regex parsing; Pydantic validates the schema
- `source_id` integers link citations back to retrieved chunks deterministically

### Metadata propagation

Following [LangChain sources guide](https://python.langchain.com/docs/how_to/qa_sources/), every chunk carries: `filename`, `title`, `author`, `chunk_index`, `page_estimate`, `total_chunks`, `source_url`. This metadata survives from ingestion → retrieval → citation card in the UI.

### Free local embeddings

Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace — **no embedding API key needed**. This keeps costs to just the LLM call.

---

## Local Development (without Docker)

```bash
# Install deps
pip install -r requirements.txt

# Start ChromaDB locally (Docker still needed for this)
docker run -p 8000:8000 chromadb/chroma

# Download data
bash data/download_data.sh

# Ingest
python -m app.ingestion.ingest

# Run backend
uvicorn app.api.main:app --reload --port 8080

# Run frontend (separate terminal)
streamlit run frontend/app.py

# CLI query
python scripts/query_cli.py "What did Darwin say about natural selection?"

# Tests
pytest tests/ -v
```

---

## Adding Your Own Documents

1. Place `.txt` files into `data/raw/`
2. Add an entry to `DOCUMENT_CATALOGUE` in `app/ingestion/ingest.py`
3. Re-run ingestion: `docker compose --profile ingest run ingest`

PDF support: install `pypdf` (already in requirements) and use `PyPDFLoader` instead of `TextLoader`.
