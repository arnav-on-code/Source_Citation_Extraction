from app.chains.citation_chain import CitationRAGChain

import sys
import os
import argparse
import textwrap

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()



DIVIDER = "─" * 70


def format_answer(response) -> None:
    ca = response.cited_answer
    print(f"\n{DIVIDER}")
    print(f"  QUESTION: {response.question}")
    print(f"{DIVIDER}")
    print(f"\n📝  ANSWER  (confidence: {ca.confidence.upper()})\n")
    # Wrap at 80 chars
    for para in ca.answer.split("\n"):
        print(textwrap.fill(para, width=80) if para.strip() else "")

    if ca.citations:
        print(f"\n{DIVIDER}")
        print(f"  CITATIONS ({len(ca.citations)})")
        print(DIVIDER)
        for cit in ca.citations:
            # Find matching chunk
            chunk = next(
                (c for c in response.retrieved_chunks if c.chunk_id == cit.source_id),
                None,
            )
            meta = chunk.metadata if chunk else None
            print(f"\n  [{cit.source_id}] {meta.title if meta else 'Unknown'}")
            if meta:
                print(f"      Author:  {meta.author}")
                print(f"      File:    {meta.filename}  ~page {meta.page_estimate}")
                if meta.source_url:
                    print(f"      URL:     {meta.source_url}")
            print(f"      Quote:   \"{cit.quote}\"")
            print(f"      Why:     {cit.relevance}")

    print(f"\n{DIVIDER}")
    print(f"  PROVENANCE — Retrieved Chunks ({len(response.retrieved_chunks)})")
    print(DIVIDER)
    for chunk in response.retrieved_chunks:
        m = chunk.metadata
        print(
            f"  #{chunk.chunk_id:02d}  score={chunk.similarity_score:.4f}  "
            f"{m.title[:35]:<35}  p.{m.page_estimate or '?':<4}  "
            f"chunk {m.chunk_index+1}/{m.total_chunks or '?'}"
        )

    print(f"\n  Model: {response.model_used}  |  Collection: {response.collection_name}")
    print(f"{DIVIDER}\n")


def main():
    parser = argparse.ArgumentParser(description="Citation RAG CLI")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.question:
        # Interactive mode
        print("\n📚  Citation RAG — Interactive CLI")
        print("  Type your question and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                question = input("Question: ").strip()
                if not question:
                    continue
                chain = CitationRAGChain()
                response = chain.invoke(question=question, top_k=args.top_k)
                format_answer(response)
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        chain = CitationRAGChain()
        response = chain.invoke(question=args.question, top_k=args.top_k)
        if args.json:
            print(response.model_dump_json(indent=2))
        else:
            format_answer(response)


if __name__ == "__main__":
    main()