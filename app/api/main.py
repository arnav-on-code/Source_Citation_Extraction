from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import chromadb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.chains.citation_chain import CitationRAGChain
from app.schemas import QueryRequest, QueryResponse
from app.config import settings

_chain: Optional[CitationRAGChain] = None

def get_chain() -> CitationRAGChain:
    global _chain
    if _chain is None:
        _chain = CitationRAGChain()
    return _chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_chain()
        print("Citation RAG chain read...")
    except Exception as e:
        print(f"Warning: chain not ready at startup: {e}")
    yield

app = FastAPI(
    title="Citation RAG API",
    description=(
        "Retrieval-Augmented Generation with structured source citation and "
        "provenance tracking. Every answer includes verbatim quotes and "
        "document metadata for full auditability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health():
    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host, port = settings.chroma_port
        )
        client.heartbeat()
        chroma_status= "ok"
    except Exception as e:
        chroma_status =f"error: {e}"
    
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "chromadb": chroma_status,
        "collection": settings.collection_name,
    }

@app.get("/collection/info", tags=["System"])
def collection_info():
    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        col= client.get_or_create_collection(settings.collection_name)
        count =col.count()
        return {
            "collection_name": settings.collection_name,
            "total_chunks": count,
            "chroma_host": settings.chroma_host,
        }
    except Exception as e:
        raise HTTPException(status_code= 503, detail= str(e))
    
@app.post("/query", response_model = QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    
    try:
        chain= get_chain()
        return chain.invoke(question=request.qurestion, top_k=request.top_k)
    except ValueError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    except Exception as e:
        raise HTTPException(status_code= 500, detail= str(e))
    

@app.get("/documents", tags=["RAG"])
def list_documents():
    """List all unique source documents ingested into the collection."""
    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        col = client.get_or_create_collection(settings.collection_name)

        # Pull all metadata; deduplicate by filename
        results = col.get(include=["metadatas"], limit=10_000)
        seen: dict = {}
        for meta in (results.get("metadatas") or []):
            fn = meta.get("filename", "unknown")
            if fn not in seen:
                seen[fn] = {
                    "filename": fn,
                    "title": meta.get("title", fn),
                    "author": meta.get("author", ""),
                    "source_url": meta.get("source_url", ""),
                    "total_chunks": meta.get("total_chunks", "?"),
                }

        return {"documents": list(seen.values()), "total_sources": len(seen)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
