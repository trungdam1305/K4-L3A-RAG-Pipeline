"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5: corpus được lấy lại bằng
chunk_documents(load_documents()) của Task 4, nên id và metadata khớp hệt bản
đã index vào Chroma. Nhờ đó Task 7 gộp được hai bảng xếp hạng theo id.

BM25 bù đúng điểm yếu của dense search: từ khóa chính xác như "Band 7",
"lexical resource", "Task Achievement" hay mã tài liệu.

CORPUS để rỗng và nạp lười ở lần gọi đầu; test monkeypatch biến này nên
lexical_search() phải đọc nó tại thời điểm gọi, không bind sẵn làm default.
"""

import re


CORPUS: list[dict] = []

# Cache một phần tử: (corpus đã dùng, BM25 index, corpus đã tokenize) để
# Streamlit không phải tokenize lại toàn bộ corpus mỗi lần user gõ một query.
_index_cache: tuple[list[dict], object, list[list[str]]] | None = None
_corpus_cache: list[dict] | None = None


def tokenize(text: str) -> list[str]:
    """Tokenizer dùng chung cho corpus và query.

    Tách theo ký tự word thay vì .split() để "Band 7," và "Band 7" ra cùng token.
    """
    return re.findall(r"[\w']+", text.lower())


def load_corpus() -> list[dict]:
    """Nạp lại đúng bộ chunks của Task 4."""
    global _corpus_cache

    if _corpus_cache is None:
        from .task4_chunking_indexing import chunk_documents, load_documents

        _corpus_cache = chunk_documents(load_documents())
    return _corpus_cache


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(item["content"]) for item in corpus])


def _get_index(corpus: list[dict]) -> tuple[object, list[list[str]]]:
    """Trả (BM25 index, corpus đã tokenize), rebuild khi corpus là object khác."""
    global _index_cache

    if _index_cache is None or _index_cache[0] is not corpus:
        _index_cache = (corpus, build_bm25_index(corpus), [tokenize(item["content"]) for item in corpus])
    return _index_cache[1], _index_cache[2]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0:
        return []

    corpus = CORPUS if CORPUS else load_corpus()
    if not corpus:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    bm25, tokenized = _get_index(corpus)
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(corpus)), key=lambda index: scores[index], reverse=True)

    query_tokens = set(tokens)
    results: list[dict] = []
    for index in ranked[:top_k]:
        # Loại chunk không chia sẻ token nào với query: giữ lại chỉ làm loãng
        # bảng xếp hạng đưa sang RRF.
        #
        # Không dùng "score <= 0" làm điều kiện loại: công thức idf của
        # BM25Okapi là log(N-df+0.5) - log(df+0.5), nên trên corpus rất nhỏ
        # (N=2, df=1) idf ra đúng 0 và chunk khớp hoàn hảo vẫn bị score 0.
        if scores[index] <= 0 and not query_tokens.intersection(tokenized[index]):
            continue
        item = corpus[index]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": dict(item["metadata"]),
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("Band 7 lexical resource collocation", top_k=3):
        print(f"{result['score']:.4f}  {result['id']}")
        print(f"        {result['content'][:120]}...")
