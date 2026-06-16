
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: int = Field(
        description=(
            "The integer ID of the source document chunk this citation refers to. "
            "Must match an id from the provided context."
        )
    )
    quote: str = Field(
        description=(
            "A verbatim short quote from the source document that directly supports "
            "the claim being made in the answer. Keep it under 150 characters."
        )
    )
    relevance: str = Field(
        description=(
            "One sentence explaining why this source supports the specific claim "
            "being cited."
        )
    )


class CitedAnswer(BaseModel):
    
    answer: str = Field(
        description=(
            "The full answer to the user's question. Write in clear prose. "
            "After each factual claim, append a bracketed citation marker like [1] "
            "that maps to the source_id in the citations list."
        )
    )
    citations: List[Citation] = Field(
        description="List of citations that back up specific claims in the answer.",
        default_factory=list,
    )
    confidence: str = Field(
        description=(
            "Overall confidence in the answer: 'high', 'medium', or 'low'. "
            "Base this on how directly the retrieved sources address the question."
        ),
        default="medium",
    )



class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceMetadata(BaseModel):

    filename: str
    title: str
    author: str
    chunk_index: int
    page_estimate: Optional[int] = None
    total_chunks: Optional[int] = None
    source_url: Optional[str] = None


class RetrievedChunk(BaseModel):

    chunk_id: int          # sequential ID used in Citation.source_id
    content: str
    metadata: SourceMetadata
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):

    question: str
    cited_answer: CitedAnswer
    retrieved_chunks: List[RetrievedChunk]
    model_used: str
    collection_name: str
