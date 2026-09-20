"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_TOKENS = 900

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ các nguồn hiện có."
SYSTEM_PROMPT = """You are an IELTS Writing evidence assistant.
Answer only from the supplied context. Cite every factual claim with one or more
source labels such as [S1] or [S2]. Never invent a citation. If the context does
not contain enough evidence, reply exactly: Tôi không thể xác minh thông tin này
từ các nguồn hiện có."""

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
}


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + list(reversed(back))


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        label = chunk.get("citation_label", f"S{index}")
        url = metadata.get("url") or "local corpus"
        parts.append(
            f"[{label}]\n"
            f"Title: {metadata['title']}\n"
            f"Source: {metadata['source']}\n"
            f"URL: {url}\n"
            f"Content: {chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = LLM_PROVIDER
    model = LLM_MODEL or DEFAULT_MODELS.get(provider, "")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        response = OpenAI(api_key=api_key).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        from google import genai
        from google.genai import types

        response = genai.Client(api_key=api_key).models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        return response.text or ""

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        from anthropic import Anthropic

        response = Anthropic(api_key=api_key).messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}")


def _safe_refusal() -> dict:
    return {
        "answer": SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }


def generate_from_chunks(query: str, chunks: list[dict]) -> dict:
    """Generate one grounded answer from a supplied retrieval result list."""
    if not query.strip() or not chunks:
        return _safe_refusal()

    labelled_sources = [
        {**chunk, "citation_label": f"S{index}"}
        for index, chunk in enumerate(chunks, start=1)
    ]
    reordered = reorder_for_llm(labelled_sources)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query.strip()}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message).strip()
    except Exception:
        return _safe_refusal()

    valid_labels = {source["citation_label"] for source in labelled_sources}
    cited_labels = set(re.findall(r"\[(S\d+)\]", answer))
    if not answer or not cited_labels or not cited_labels.issubset(valid_labels):
        return _safe_refusal()

    retrieval_method = labelled_sources[0].get("retrieval_method")
    retrieval_source = "pageindex" if retrieval_method == "pageindex" else "hybrid"
    return {
        "answer": answer,
        "sources": labelled_sources,
        "retrieval_source": retrieval_source,
    }


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        return _safe_refusal()
    return generate_from_chunks(query, chunks)


if __name__ == "__main__":
    print(generate_with_citation("test query"))
