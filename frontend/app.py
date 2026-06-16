
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Citation RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.citation-card {
    background: #f8f9ff;
    border-left: 4px solid #4f46e5;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 8px 0;
}
.chunk-card {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 6px 0;
    font-size: 0.85rem;
}
.confidence-high { color: #16a34a; font-weight: bold; }
.confidence-medium { color: #ca8a04; font-weight: bold; }
.confidence-low { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("📚 Citation RAG")
    st.caption("Source-grounded answers with provenance tracking")

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=15, value=5)

    st.divider()
    st.subheader("📄 Ingested Documents")

    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        if r.status_code == 200:
            docs_data = r.json()
            st.caption(f"{docs_data['total_sources']} source(s) in collection")
            for doc in docs_data["documents"]:
                with st.expander(f"📖 {doc['title']}"):
                    st.write(f"**Author:** {doc['author']}")
                    st.write(f"**Chunks:** {doc['total_chunks']}")
                    if doc.get("source_url"):
                        st.write(f"[Original source]({doc['source_url']})")
        else:
            st.warning("Could not load document list.")
    except Exception:
        st.error("Backend unreachable. Is it running?")

    st.divider()
    st.subheader("🔗 API")
    st.markdown(f"[Swagger UI]({BACKEND_URL}/docs)")
    st.markdown(f"[Health]({BACKEND_URL}/health)")


st.title("🔍 Ask a Question")
st.caption("Every answer is grounded in retrieved source documents with verbatim citations.")

# Example questions
with st.expander("💡 Example questions"):
    examples = [
        "What does Sun Tzu say about knowing your enemy?",
        "How does Marcus Aurelius describe dealing with adversity?",
        "What is Plato's concept of the ideal state?",
        "How did Darwin explain natural selection?",
        "What does the Federalist Papers say about separation of powers?",
    ]
    for q in examples:
        if st.button(q, key=q):
            st.session_state["question_input"] = q

question = st.text_input(
    "Your question:",
    key="question_input",
    placeholder="e.g. What does Marcus Aurelius say about virtue?",
)

submit = st.button("🔎 Search & Cite", type="primary", use_container_width=True)

if submit and question.strip():
    with st.spinner("Retrieving sources and generating cited answer..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=60,
            )
            if resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
                st.stop()

            data = resp.json()
            ca = data["cited_answer"]
            chunks = data["retrieved_chunks"]

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure it's running.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    
    st.subheader("📝 Answer")

    conf = ca.get("confidence", "medium")
    conf_class = f"confidence-{conf}"
    st.markdown(
        f'Confidence: <span class="{conf_class}">{conf.upper()}</span> &nbsp;|&nbsp; '
        f'Model: `{data["model_used"]}` &nbsp;|&nbsp; '
        f'Sources retrieved: `{len(chunks)}`',
        unsafe_allow_html=True,
    )

    st.markdown(ca["answer"])

     
    citations = ca.get("citations", [])
    if citations:
        st.subheader(f"🔖 Citations ({len(citations)})")
        for cit in citations:
            src_id = cit["source_id"]
            # Match to retrieved chunk
            chunk = next((c for c in chunks if c["chunk_id"] == src_id), None)
            title = chunk["metadata"]["title"] if chunk else "Unknown"
            author = chunk["metadata"]["author"] if chunk else ""
            page = chunk["metadata"].get("page_estimate", "?") if chunk else "?"
            filename = chunk["metadata"]["filename"] if chunk else ""
            source_url = chunk["metadata"].get("source_url", "") if chunk else ""

            st.markdown(
                f"""<div class="citation-card">
                <strong>[{src_id}] {title}</strong><br>
                <em>{author}</em> — ~p.{page}<br>
                <blockquote style="border-left:2px solid #4f46e5; margin:8px 0; padding-left:10px; color:#374151;">
                    "{cit['quote']}"
                </blockquote>
                <small>💬 {cit['relevance']}</small><br>
                <small style="color:#6b7280;">📁 {filename}
                {"&nbsp;·&nbsp;<a href='" + source_url + "' target='_blank'>source</a>" if source_url else ""}
                </small>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No structured citations were returned for this answer.")

    with st.expander(f"🗂️ Retrieved Chunks — Full Provenance ({len(chunks)} chunks)"):
        for chunk in chunks:
            meta = chunk["metadata"]
            score = chunk.get("similarity_score", "N/A")
            st.markdown(
                f"""<div class="chunk-card">
                <strong>Chunk #{chunk['chunk_id']}</strong> &nbsp;
                <code>score={score}</code><br>
                <em>{meta['title']}</em> by {meta['author']} — ~p.{meta.get('page_estimate','?')}
                (chunk {meta['chunk_index']+1}/{meta.get('total_chunks','?')})<br>
                <small>📁 {meta['filename']}</small>
                </div>""",
                unsafe_allow_html=True,
            )
            st.text(chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""))
            st.divider()

elif submit and not question.strip():
    st.warning("Please enter a question.")
