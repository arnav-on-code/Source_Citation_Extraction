"""Central config — reads from environment / .env file."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Models
    anthropic_model: str = "claude-3-5-haiku-20241022"   # fast + cheap
    openai_model: str = "gpt-4o-mini"

    # ChromaDB
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("COLLECTION_NAME", "citation_rag")

    # Ingestion
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"


settings = Settings()
