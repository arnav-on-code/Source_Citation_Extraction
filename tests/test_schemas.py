
import pytest
from pydantic import ValidationError

from app.schemas import (
    Citation,
    CitedAnswer,
    QueryRequest,
    SourceMetadata,
    RetrievedChunk,
    QueryResponse,
)


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestCitationSchema:
    def test_valid_citation(self):
        c = Citation(
            source_id=1,
            quote="All warfare is based on deception.",
            relevance="Directly supports the claim about deception in strategy.",
        )
        assert c.source_id == 1
        assert "deception" in c.quote

    def test_citation_requires_fields(self):
        with pytest.raises(ValidationError):
            Citation(source_id=1)  # missing quote and relevance


class TestCitedAnswerSchema:
    def test_valid_cited_answer(self):
        ca = CitedAnswer(
            answer="Sun Tzu argues that deception is fundamental [1].",
            citations=[
                Citation(
                    source_id=1,
                    quote="All warfare is based on deception.",
                    relevance="Core principle stated explicitly.",
                )
            ],
            confidence="high",
        )
        assert len(ca.citations) == 1
        assert ca.confidence == "high"

    def test_default_confidence(self):
        ca = CitedAnswer(answer="Some answer.", citations=[])
        assert ca.confidence == "medium"

    def test_empty_citations_allowed(self):
        ca = CitedAnswer(answer="No citations found.", citations=[])
        assert ca.citations == []


class TestQueryRequest:
    def test_valid_request(self):
        req = QueryRequest(question="What is natural selection?")
        assert req.top_k == 5  # default

    def test_too_short_question(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")

    def test_custom_top_k(self):
        req = QueryRequest(question="What is natural selection?", top_k=10)
        assert req.top_k == 10

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here", top_k=25)


class TestSourceMetadata:
    def test_valid_metadata(self):
        meta = SourceMetadata(
            filename="art_of_war.txt",
            title="The Art of War",
            author="Sun Tzu",
            chunk_index=42,
            page_estimate=17,
            total_chunks=120,
            source_url="https://www.gutenberg.org/files/132/132-0.txt",
        )
        assert meta.author == "Sun Tzu"

    def test_optional_fields(self):
        meta = SourceMetadata(
            filename="test.txt",
            title="Test",
            author="Unknown",
            chunk_index=0,
        )
        assert meta.page_estimate is None
        assert meta.source_url is None


class TestRetrievedChunk:
    def test_chunk_roundtrip(self):
        chunk = RetrievedChunk(
            chunk_id=3,
            content="The supreme art of war is to subdue the enemy without fighting.",
            metadata=SourceMetadata(
                filename="art_of_war.txt",
                title="The Art of War",
                author="Sun Tzu",
                chunk_index=10,
            ),
            similarity_score=0.9123,
        )
        assert chunk.chunk_id == 3
        assert chunk.similarity_score == pytest.approx(0.9123)
