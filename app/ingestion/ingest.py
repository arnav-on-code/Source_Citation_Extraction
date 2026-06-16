
import os
import glob
from typing import List, Dict

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import chromadb

from app.config import settings

# ── Document metadata catalogue ──────────────────────────────────────────────
DOCUMENT_CATALOGUE: Dict[str, Dict] = {
    "federalist_papers.txt": {
        "title": "The Federalist Papers",
        "author": "Alexander Hamilton, James Madison, John Jay",
        "source_url": "https://www.gutenberg.org/files/1404/1404-0.txt",
    },
    "origin_of_species.txt": {
        "title": "On the Origin of Species",
        "author": "Charles Darwin",
        "source_url": "https://www.gutenberg.org/files/1228/1228-0.txt",
    },
    "art_of_war.txt": {
        "title": "The Art of War",
        "author": "Sun Tzu",
        "source_url": "https://www.gutenberg.org/files/132/132-0.txt",
    },
    "meditations.txt": {
        "title": "Meditations",
        "author": "Marcus Aurelius",
        "source_url": "https://www.gutenberg.org/files/2680/2680-0.txt",
    },
    "the_republic.txt": {
        "title": "The Republic",
        "author": "Plato",
        "source_url": "https://www.gutenberg.org/files/1497/1497-0.txt",
    },
}


def _estimate_page(text: str, chunk_index: int, chunk_size: int) -> int:
    """Rough page estimate: 250 words ≈ 1 page."""
    words_per_page = 250
    words_per_chunk = chunk_size // 5  # avg 5 chars per word
    return max(1, (chunk_index * words_per_chunk) // words_per_page + 1)


def load_and_split(data_dir: str) -> List[Document]:
    """Load all .txt files and split into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )

    all_docs: List[Document] = []

    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}.\n"
            "Run:  bash data/download_data.sh"
        )

    for filepath in sorted(txt_files):
        filename = os.path.basename(filepath)
        catalogue_entry = DOCUMENT_CATALOGUE.get(filename, {})

        print(f"  Loading: {filename}")
        loader = TextLoader(filepath, encoding="utf-8", autodetect_encoding=True)
        raw_docs = loader.load()

        chunks = splitter.split_documents(raw_docs)
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "filename": filename,
                    "title": catalogue_entry.get("title", filename),
                    "author": catalogue_entry.get("author", "Unknown"),
                    "source_url": catalogue_entry.get("source_url", ""),
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "page_estimate": _estimate_page(chunk.page_content, i, settings.chunk_size),
                }
            )

        all_docs.extend(chunks)
        print(f"    → {total_chunks} chunks")

    print(f"\nTotal chunks across all documents: {len(all_docs)}")
    return all_docs


def get_embeddings():
    """Use free local HuggingFace embeddings (no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(docs: List[Document]) -> Chroma:
    """Embed documents and persist to ChromaDB."""
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    embeddings = get_embeddings()

    print(f"\nEmbedding {len(docs)} chunks → ChromaDB collection '{settings.collection_name}' ...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        collection_name=settings.collection_name,
    )
    print("Ingestion complete ✓")
    return vectorstore


def main():
    print("=" * 60)
    print("Citation RAG — Document Ingestion")
    print("=" * 60)

    data_dir = os.path.abspath(settings.data_dir)
    print(f"\nData directory: {data_dir}")

    docs = load_and_split(data_dir)
    build_vectorstore(docs)


if __name__ == "__main__":
    main()