import os

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()


def render_sources(sources: list[dict], retrieval_source: str) -> None:
    if not sources:
        return

    st.markdown(
        f'<div class="source-heading">Evidence trail · {retrieval_source}</div>',
        unsafe_allow_html=True,
    )
    for index, source in enumerate(sources, start=1):
        metadata = source["metadata"]
        label = source.get("citation_label", f"S{index}")
        title = metadata.get("title") or metadata.get("source") or "Untitled source"
        method = source.get("retrieval_method", "unknown")
        score = float(source.get("score", 0.0))
        with st.expander(f"[{label}] {title} · {method} · {score:.4f}"):
            st.caption(f"Source file: {metadata.get('source', 'unknown')}")
            if metadata.get("url"):
                st.markdown(f"[Open original source]({metadata['url']})")
            st.write(source.get("content", ""))


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "none"),
            )


def main() -> None:
    st.set_page_config(
        page_title="IELTS Writing Evidence Lab",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

        :root {
            --ink: #17211b;
            --forest: #1f4b3b;
            --paper: #f7f2e8;
            --amber: #d9902f;
        }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 85% 8%, rgba(217, 144, 47, .18), transparent 24rem),
                linear-gradient(135deg, #fbf8f0 0%, var(--paper) 55%, #edf2e9 100%);
            font-family: 'DM Sans', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'DM Serif Display', serif !important;
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #173c30 0%, #102b23 100%);
        }
        [data-testid="stSidebar"] * { color: #f8f1df; }
        [data-testid="stSidebar"] code {
            color: #fff7e8 !important;
            background: rgba(255, 255, 255, .12) !important;
        }
        [data-testid="stSidebar"] button {
            background: var(--amber);
            border: 0;
        }
        [data-testid="stSidebar"] button p {
            color: #17211b !important;
            font-weight: 700;
        }
        .hero-kicker {
            color: var(--forest);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .hero-copy {
            max-width: 780px;
            color: #516057;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .source-heading {
            color: var(--forest);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .08em;
            margin-top: 1rem;
            text-transform: uppercase;
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, .72);
            border: 1px solid rgba(31, 75, 59, .12);
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(23, 33, 27, .05);
            margin-bottom: .8rem;
            padding: .25rem .65rem;
        }
        [data-testid="stChatInput"] textarea { background: #fffdf8; }
        @media (max-width: 760px) {
            .block-container { padding: 1.5rem 1rem 6rem; }
            .hero-copy { font-size: .95rem; }
            [data-testid="stChatMessage"] { border-radius: 14px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("## IELTS Evidence Lab")
        st.caption("12 documents · official descriptors and public guidance")
        top_k = st.slider("Evidence chunks", min_value=3, max_value=10, value=5)
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip() or "provider default"
        st.markdown(f"**Generator:** `{provider}`")
        st.markdown(f"**Model:** `{model}`")
        st.markdown("**Retrieval:** dense + BM25 → RRF")
        st.markdown("**Fallback threshold:** `0.8689`")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="hero-kicker">Grounded IELTS Writing assistant</div>', unsafe_allow_html=True)
    st.title("Ask the rubric, not the rumor.")
    st.markdown(
        '<div class="hero-copy">Explore IELTS Writing band descriptors, assessment criteria, '
        'and preparation guidance. Every supported answer links back to retrieved evidence.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.info(
            "Try: What does Band 7 Task Response require? · How is Task 1 assessed? · "
            "What should candidates know about coherence and cohesion?"
        )

    for message in st.session_state.messages:
        render_message(message)

    query = st.chat_input("Ask about IELTS Writing criteria...")
    if not query:
        return

    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Tracing the evidence..."):
            result = generate_with_citation(query, top_k=top_k)
        st.markdown(result["answer"])
        render_sources(result["sources"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )


if __name__ == "__main__":
    main()
