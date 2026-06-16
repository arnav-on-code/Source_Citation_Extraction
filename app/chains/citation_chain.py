from __future__ import annotations

from typing import List, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.config import settings
from app.schemas import CitedAnswer, RetrievedChunk, SourceMetadata


CITATION_SYSTEM_PROMPT = """\
You are a precise research assistant that answers questions using ONLY the
provided source documents. You must cite every factual claim.

Rules:
1. Answer only from the numbered [Source X] blocks below.
2. After each factual claim in your answer, append [X] where X is the
   source_id from the citation.
3. If the sources don't contain enough information, say so clearly.
4. Never fabricate facts not present in the sources.
5. Keep quotes in citations short (under 150 chars) and verbatim.

Sources:
{context}
"""

CITATION_HUMAN_PROMPT = "Question: {question}"


def _build_llm(provider: str = None):
    provider = provider or settings.llm_provider

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    return llm.with_structured_output(CitedAnswer)


def _get_vectorstore() -> Chroma:
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=embeddings,
    )


def _format_docs_with_ids(docs_and_scores: List[Tuple[Document, float]]) -> str:

    lines = []
    for i, (doc, _score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata
        lines.append(
            f"[Source {i}]\n"
            f"Title: {meta.get('title', 'Unknown')}\n"
            f"Author: {meta.get('author', 'Unknown')}\n"
            f"File: {meta.get('filename', '')}\n"
            f"~Page {meta.get('page_estimate', '?')}\n"
            f"---\n"
            f"{doc.page_content.strip()}\n"
        )
    return "\n\n".join(lines)


def _docs_to_retrieved_chunks(
    docs_and_scores: List[Tuple[Document, float]]
) -> List[RetrievedChunk]:
    chunks = []
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata
        chunks.append(
            RetrievedChunk(
                chunk_id=i,
                content=doc.page_content,
                similarity_score=round(score, 4),
                metadata=SourceMetadata(
                    filename=meta.get("filename", ""),
                    title=meta.get("title", ""),
                    author=meta.get("author", "Unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    page_estimate=meta.get("page_estimate"),
                    total_chunks=meta.get("total_chunks"),
                    source_url=meta.get("source_url", ""),
                ),
            )
        )
    return chunks


class CitationRAGChain:
    
    def __init__(self):
        self.vectorstore = _get_vectorstore()
        self.llm = _build_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CITATION_SYSTEM_PROMPT),
                ("human", CITATION_HUMAN_PROMPT),
            ]
        )
        self.model_name = (
            settings.anthropic_model
            if settings.llm_provider == "anthropic"
            else settings.openai_model
        )

    def invoke(self, question: str, top_k: int = 5):
        from app.schemas import QueryResponse

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question, k=top_k
        )

        if not docs_and_scores:
            raise ValueError("No documents found. Have you run the ingestion step?")

        context = _format_docs_with_ids(docs_and_scores)

        chain = self.prompt | self.llm

        cited_answer: CitedAnswer = chain.invoke(
            {"context": context, "question": question}
        )

        return QueryResponse(
            question=question,
            cited_answer=cited_answer,
            retrieved_chunks=_docs_to_retrieved_chunks(docs_and_scores),
            model_used=self.model_name,
            collection_name=settings.collection_name,
        )
