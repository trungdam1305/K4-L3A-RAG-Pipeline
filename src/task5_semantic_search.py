"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.

score trả về ở đây là cosine similarity gốc (không phải score RRF) — Task 9 dùng
đúng giá trị này để so với SCORE_THRESHOLD khi quyết định fallback.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0:
        return []

    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    seen: set[str] = set()

    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        # Chroma có thể trả cùng id nếu collection bị index trùng; contract yêu
        # cầu ID unique nên bỏ bản sau.
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": content,
                # cosine distance -> similarity; clamp để không ra score âm.
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": dict(metadata),
                "retrieval_method": "dense",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("What does Band 7 require for coherence?", top_k=3):
        print(f"{result['score']:.4f}  {result['id']}")
        print(f"        {result['content'][:120]}...")
