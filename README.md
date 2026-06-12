# DocMind — RAG Document Assistant

> **CODEDIVA Assignment #2.2 — AI Engineer**  
> A production-style RAG (Retrieval-Augmented Generation) system with tiered multi-LLM routing: free model first, escalate to paid only when needed.

---

## Architecture

```
PDF / TXT upload
      │
      ▼
  pdfplumber / txt parser
      │  clean_text() — strip noise, normalize whitespace
      ▼
  smart_chunk() — sentence-aware chunking (NLTK)
      │  ~300 words per chunk, 50-word overlap
      ▼
  sentence-transformers (all-MiniLM-L6-v2, runs locally)
      │  embed chunks → float32 vectors
      ▼
  ChromaDB (persistent vector store → ./chroma_db/)
      │  cosine similarity search, top-4 chunks
      ▼
  LLM Router
  ├─ provider=auto  →  Llama 3.1 8B via Groq (FREE, ~1s latency)
  │                    └─ low confidence? → escalation card → user picks
  ├─ provider=claude  →  Claude Sonnet 4 (Anthropic key required)
  └─ provider=openai  →  GPT-4o mini (OpenAI key required)
      │
      ▼
  Answer + cited_sources (chunk text, page number, similarity score)
```

---

## Features

- **Persistent vector store** — ChromaDB stores embeddings to disk; documents survive server restarts
- **Sentence-aware chunking** — NLTK tokenizer never cuts mid-sentence
- **Compound query retrieval** — detects "and / also / as well as" and fans out to sub-queries, merges results
- **Cited sources** — every answer includes page numbers and similarity scores
- **Tiered LLM routing** — free Llama first, escalate to Claude/GPT-4o mini on low confidence
- **Clean web UI** — no build step, drag-and-drop upload, chat interface with source panels

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/<your-username>/rag-assignment.git
cd rag-assignment
```

### 2. Set up Python environment

> Requires **Python 3.9+** (tested on 3.11). Python 3.13 works too.

```bash
cd backend
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API key

```bash
cp .env.example .env
```

Open `.env` and add your **free** Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

Get one free (no credit card) at **[console.groq.com](https://console.groq.com)**:
→ Sign in with Google → **API Keys** → **Create API Key** → copy `gsk_...`

### 4. Run

```bash
uvicorn main:app --reload --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

> **Or use Make:**
> ```bash
> make install   # install dependencies
> make run       # start the server
> ```

---

## Usage

1. **Upload** a PDF or TXT file by dragging onto the sidebar drop zone
2. **Select model** — default is Auto (Llama 3.1 8B via Groq, free)
3. **Ask** any question about the document
4. If the free model is unsure, an **escalation card** appears with one-click retry via Claude or GPT-4o mini

### Example questions to try

- *"What are the main terms of this agreement?"*
- *"What are the parties involved and what are their obligations?"* (compound query)
- *"Summarize the risk factors"*

---

## API Reference

### `POST /api/ingest`

Upload a PDF or TXT file for indexing.

**Request:** `multipart/form-data` with a `file` field

**Response:**
```json
{
  "doc_id": "a1b2c3d4",
  "filename": "agreement.pdf",
  "chunks": 54
}
```

---

### `POST /api/query`

Query an indexed document. **This is the primary endpoint** — returns answer with cited sources.

**Request:**
```json
{
  "doc_id": "a1b2c3d4",
  "query": "What is the refund policy?",
  "provider": "auto"
}
```

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | ID returned by `/api/ingest` |
| `query` | string | Natural language question |
| `provider` | string | `"auto"` (free) \| `"claude"` \| `"openai"` |

**Response:**
```json
{
  "answer": "The refund policy allows returns within 30 days...\n\n---\nSources: Page 3, Page 5",
  "cited_sources": [
    "Page 3 (score: 87%): Refunds are processed within 30 days of purchase...",
    "Page 5 (score: 71%): Contact support@example.com for return requests..."
  ],
  "sources": [
    { "chunk": "...within 30 days of purchase...", "page": 3, "score": 0.87 },
    { "chunk": "...contact support for returns...", "page": 5, "score": 0.71 }
  ],
  "provider_used": "llama",
  "model_used": "llama-3.1-8b-instant (Groq)",
  "escalated": false,
  "escalation_reason": null
}
```

If `escalated: true`, the frontend shows options to retry with Claude or OpenAI. The `escalation_reason` field indicates why: `"low_confidence"` or `"low_retrieval_score"`.

---

### `GET /api/health`

```json
{ "status": "ok", "docs_loaded": 1 }
```

### `GET /api/docs-list`

Returns all currently indexed documents.

```json
{
  "documents": [
    { "doc_id": "a1b2c3d4", "filename": "agreement.pdf", "chunk_count": 54 }
  ]
}
```

### `DELETE /api/docs/{doc_id}`

Delete a document and its vector store collection.

---

## Test with curl

```bash
# 1. Upload a document
curl -s -X POST http://localhost:8000/api/ingest \
  -F "file=@backend/sample_docs/sample.txt"

# 2. Query it (replace doc_id with value from step 1)
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"a1b2c3d4","query":"What is this document about?","provider":"auto"}' \
  | python -m json.tool

# 3. Health check
curl http://localhost:8000/api/health
```

---

## Testing Paid Models (Claude / GPT-4o mini)

The free Llama model via Groq works out of the box with just `GROQ_API_KEY`.

To test Claude or GPT-4o mini, add your key to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...   # → select "Claude Sonnet" in UI
OPENAI_API_KEY=sk-...          # → select "GPT-4o mini" in UI
```

Both are **optional** — the system works fully without them. If a paid key is missing and that model is selected, the API returns a clear `400` error rather than silently failing.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| PDF parsing | pdfplumber |
| Text cleaning | Custom `clean_text()` + NLTK |
| Chunking | Sentence-aware `smart_chunk()` via NLTK |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Free LLM | Llama 3.1 8B via Groq API |
| Paid LLM (optional) | Claude Sonnet 4 / GPT-4o mini |
| Frontend | Vanilla HTML/CSS/JS (no build step) |

---

## Project Structure

```
rag_assignment/
├── README.md
├── .env.example               ← copy to .env, add GROQ_API_KEY
├── .gitignore
├── Makefile                   ← make install / make run / make test
├── frontend/
│   └── index.html             ← full web UI, no build step needed
└── backend/
    ├── main.py                ← FastAPI app + all endpoints
    ├── requirements.txt
    ├── rag/
    │   ├── ingest.py          ← PDF/TXT → pages
    │   ├── pipeline.py        ← clean → chunk → embed → store
    │   └── retriever.py       ← ChromaDB vector search + compound query
    └── llm/
        ├── base.py            ← abstract LLMProvider + confidence check
        ├── hf.py              ← Llama 3.1 8B via Groq (free default)
        ├── claude.py          ← Claude Sonnet 4 (Anthropic)
        └── openai_llm.py      ← GPT-4o mini (OpenAI)
```

---

## Escalation Logic

```
/query (provider="auto")
        │
        ▼
   Retrieve top-4 chunks (ChromaDB cosine similarity)
        │
   top_score < 0.15? ──YES──► escalated=true, reason="low_retrieval_score"
        │NO
        ▼
   Call Llama 3.1 8B (Groq)
        │
   confidence check:
   ├── "I could not find" in answer?
   ├── answer < 15 words?
   └── any low-confidence phrase detected?
        │
      PASS ──────────────► return answer + cited_sources + escalated=false
        │FAIL
        ▼
   return escalated=true, reason="low_confidence"
        │
   Frontend shows: "Try Claude" / "Try GPT-4o mini" buttons
        │
   User picks → re-call /query with provider="claude"|"openai"
```

---

## Requirements

- Python 3.9+
- Free Groq API key ([console.groq.com](https://console.groq.com))
- ~500 MB disk space (for sentence-transformers model download on first run)
- Internet connection (for Groq API calls and first-time model download)

---

## Known Limitations & Future Improvements

### Current Limitations
- Single-user only — no auth, all documents visible to anyone on the server
- No reranking — retrieval uses cosine similarity only, no cross-encoder reranking
- Context window limit — very long documents may lose information across chunks
- Free tier rate limits — Groq allows 30 RPM / 1,000 RPD on free tier

### What Could Be Improved
- **Reranking** — add a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM`) for better chunk selection
- **Hybrid search** — combine dense vectors with BM25 keyword search for better recall
- **Multi-document queries** — currently queries one document at a time
- **Cloud vector DB** — swap ChromaDB for Supabase pgvector or Pinecone for true cloud persistence
- **Streaming responses** — stream LLM tokens to frontend instead of waiting for full response
- **Evaluation metrics** — add RAGAS evaluation (faithfulness, answer relevancy, context precision)
