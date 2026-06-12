import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
print(f"GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}")

from rag import extract_text_from_pdf, extract_text_from_txt, VectorStore, get_embedding_model
from rag.pipeline import smart_chunk
from llm import HFProvider, ClaudeProvider, OpenAIProvider

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_stores: dict[str, VectorStore] = {}
_doc_meta: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for col_info in _chroma_client.list_collections():
        doc_id = col_info.name
        meta = getattr(col_info, "metadata", {}) or {}
        collection = _chroma_client.get_collection(doc_id)
        _stores[doc_id] = VectorStore(collection)
        _doc_meta[doc_id] = {
            "filename": meta.get("filename", "unknown"),
            "chunk_count": meta.get("chunk_count", 0),
        }
    yield


app = FastAPI(title="RAG Demo API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    doc_id: str
    query: str
    provider: Literal["auto", "claude", "openai"] = "auto"


class SourceItem(BaseModel):
    chunk: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str | None
    sources: list[SourceItem]
    cited_sources: list[str] = []
    provider_used: str
    model_used: str
    escalated: bool
    escalation_reason: str | None = None


# ── Citation helpers ──────────────────────────────────────────────────────────

def _cited_sources(sources: list[SourceItem]) -> list[str]:
    out = []
    for s in sources:
        preview = s.chunk[:80].strip()
        suffix = "..." if len(s.chunk) > 80 else ""
        out.append(f"Page {s.page} (score: {round(s.score * 100)}%): {preview}{suffix}")
    return out


def _with_refs(answer: str | None, sources: list[SourceItem]) -> str | None:
    if answer is None or not sources:
        return answer
    unique_pages = list(dict.fromkeys(s.page for s in sources))
    page_list = ", ".join(f"Page {p}" for p in unique_pages)
    return f"{answer}\n\n---\nSources: {page_list}"


# ── Ingest stream ─────────────────────────────────────────────────────────────

def _event(data: dict) -> str:
    return json.dumps(data) + "\n"


async def _ingest_stream(filename: str, content: bytes):
    try:
        yield _event({"status": "extracting", "progress": 10})

        if filename.lower().endswith(".pdf"):
            pages = await asyncio.to_thread(extract_text_from_pdf, content)
        else:
            pages = extract_text_from_txt(content)

        if not pages:
            yield _event({"status": "error", "message": "Could not extract text from the document."})
            return

        yield _event({"status": "chunking", "progress": 30})
        chunks = await asyncio.to_thread(smart_chunk, pages)

        if not chunks:
            yield _event({"status": "error", "message": "Document appears to be empty after chunking."})
            return

        yield _event({"status": "embedding", "progress": 60})

        model = await asyncio.to_thread(get_embedding_model)
        texts = [c["text"] for c in chunks]
        embeddings = await asyncio.to_thread(
            lambda: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        )

        yield _event({"status": "storing", "progress": 90})

        doc_id = str(uuid.uuid4())[:8]

        def _store():
            col = _chroma_client.create_collection(
                name=doc_id,
                metadata={
                    "hnsw:space": "cosine",
                    "filename": filename,
                    "chunk_count": len(chunks),
                },
            )
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "page": c["page"],
                    "chunk_index": c.get("chunk_index", i),
                    "word_count": len(c["text"].split()),
                }
                for i, c in enumerate(chunks)
            ]
            col.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)
            return col

        collection = await asyncio.to_thread(_store)
        _stores[doc_id] = VectorStore(collection)
        _doc_meta[doc_id] = {"filename": filename, "chunk_count": len(chunks)}

        yield _event({"status": "done", "doc_id": doc_id, "chunks": len(chunks), "progress": 100})

    except Exception as exc:
        yield _event({"status": "error", "message": str(exc)})


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF or TXT and stream JSON progress events while indexing."""
    filename = file.filename or "document"
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported.")
    content = await file.read()
    return StreamingResponse(_ingest_stream(filename, content), media_type="application/x-ndjson")


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Query an indexed document."""
    store = _stores.get(req.doc_id)
    if not store:
        raise HTTPException(404, f"Document '{req.doc_id}' not found. Please upload first.")

    chunks = store.search(req.query, top_k=4)
    if not chunks:
        raise HTTPException(500, "Retrieval returned no results.")

    top_score = chunks[0]["score"] if chunks else 0.0
    sources = [
        SourceItem(chunk=c["text"][:400], page=c["page"], score=round(c["score"], 4))
        for c in chunks
    ]

    if top_score < 0.25:
        return QueryResponse(
            answer=None,
            sources=sources,
            cited_sources=_cited_sources(sources),
            provider_used="none",
            model_used="none",
            escalated=True,
            escalation_reason="low_retrieval_score",
        )

    provider_name = req.provider

    if provider_name == "auto":
        print("[query] Using provider: auto")
        try:
            result = HFProvider().generate(req.query, chunks)
        except Exception as e:
            escalation_reason = f"gemma_error: {str(e)[:100]}"
            print(f"[query] Escalating — reason: {escalation_reason}")
            return QueryResponse(
                answer=None,
                sources=sources,
                cited_sources=_cited_sources(sources),
                provider_used="gemma",
                model_used="google/gemma-2-2b-it",
                escalated=True,
                escalation_reason=escalation_reason,
            )

        if not result.confident:
            escalation_reason = "low_confidence"
            print(f"[query] Escalating — reason: {escalation_reason}")
            return QueryResponse(
                answer=_with_refs(result.answer, sources),
                sources=sources,
                cited_sources=_cited_sources(sources),
                provider_used=result.provider,
                model_used=result.model,
                escalated=True,
                escalation_reason=escalation_reason,
            )

        return QueryResponse(
            answer=_with_refs(result.answer, sources),
            sources=sources,
            cited_sources=_cited_sources(sources),
            provider_used=result.provider,
            model_used=result.model,
            escalated=False,
        )

    elif provider_name == "claude":
        if not os.getenv("ANTHROPIC_API_KEY", ""):
            raise HTTPException(400, "ANTHROPIC_API_KEY is not configured.")
        result = ClaudeProvider().generate(req.query, chunks)

    elif provider_name == "openai":
        if not os.getenv("OPENAI_API_KEY", ""):
            raise HTTPException(400, "OPENAI_API_KEY is not configured.")
        result = OpenAIProvider().generate(req.query, chunks)

    else:
        raise HTTPException(400, f"Unknown provider: {provider_name}")

    return QueryResponse(
        answer=_with_refs(result.answer, sources),
        sources=sources,
        cited_sources=_cited_sources(sources),
        provider_used=result.provider,
        model_used=result.model,
        escalated=False,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "docs_loaded": len(_stores)}


@app.get("/api/docs-list")
def docs_list():
    return {"documents": [{"doc_id": k, **v} for k, v in _doc_meta.items()]}


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str):
    if doc_id not in _stores:
        raise HTTPException(404, f"Document '{doc_id}' not found.")
    _chroma_client.delete_collection(doc_id)
    del _stores[doc_id]
    del _doc_meta[doc_id]
    return {"deleted": doc_id}


# Serve frontend
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
