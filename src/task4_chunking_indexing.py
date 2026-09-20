"""
Task 4 — Chunking, embedding và indexing.

Đọc Markdown trong data/standardized/, chunk bằng RecursiveCharacterTextSplitter,
embed bằng một provider duy nhất và upsert vào ChromaDB (cosine).

ID ổn định: document id là đường dẫn tương đối trong standardized/, chunk id là
"<doc_id>::chunk-<index>". Chạy lại pipeline chỉ upsert đè lên chính nó, không
sinh bản trùng.

Task 5 import embed_texts() và get_collection() từ đây để dense search dùng đúng
model và dimension đã index.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
MANIFEST_PATH = STANDARDIZED_DIR / "manifest.json"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# chunk_size 500 đủ để giữ một tiêu chí band descriptor trọn vẹn mà không kéo
# theo band khác; overlap 50 để câu bị cắt giữa hai chunk vẫn còn ngữ cảnh.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Đổi từ BAAI/bge-m3 (default của repo) sang e5-small: 470MB thay vì 2.3GB và
# embed nhanh hơn nhiều trên CPU, vẫn đa ngữ. Đổi model thì phải đổi cả
# EMBEDDING_DIM và re-index, nên hai hằng số này luôn đi cùng nhau.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = 384
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

COLLECTION_NAME = "rag_documents"
EMBED_BATCH_SIZE = 32

_model_cache: dict[str, object] = {}


def _sentence_transformer():
    """Load model một lần rồi cache: Streamlit rerun mỗi lần user gõ."""
    if EMBEDDING_MODEL not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[EMBEDDING_MODEL] = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache[EMBEDDING_MODEL]


def embed_texts(texts: list[str], prefix: str = "query: ") -> list[list[float]]:
    """Embed danh sách text theo EMBEDDING_PROVIDER trong .env.

    Họ model e5 được train với tiền tố "query: " và "passage: "; bỏ tiền tố làm
    giảm chất lượng retrieval rõ rệt. Default là "query: " để Task 5 gọi
    embed_texts([query]) là đúng ngay, còn embed_chunks() truyền "passage: ".
    Provider khác e5 thì tiền tố bị bỏ qua.
    """
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "sentence_transformers":
        if "e5" in EMBEDDING_MODEL.lower():
            texts = [f"{prefix}{text}" for text in texts]
        vectors = _sentence_transformer().encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        client = genai.Client()
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [list(item.values) for item in response.embeddings]

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown trong standardized/ và trả về danh sách Document.

    Metadata lấy từ manifest.json của Task 3 để giữ đúng title và URL nguồn.
    Không có manifest thì fallback về tên file, url = None.
    """
    import json

    manifest = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists()
        else {}
    )

    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        entry = manifest.get(doc_id, {})
        documents.append(
            {
                "id": doc_id,
                "content": content,
                "metadata": {
                    "source": entry.get("source") or path.name,
                    "title": entry.get("title") or path.stem.replace("-", " "),
                    "doc_type": entry.get("doc_type")
                    or ("legal" if "legal" in path.parts else "news"),
                    "url": entry.get("url"),
                },
            }
        )

    if not documents:
        raise RuntimeError(
            f"No Markdown found in {STANDARDIZED_DIR}. Run task1-task3 first."
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id ổn định và chunk_index liên tục."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        index = 0
        for text in splitter.split_text(document["content"]):
            text = text.strip()
            if not text:
                # Chunk rỗng vi phạm contract; bỏ qua và không tăng chunk_index
                # để index luôn liên tục từ 0.
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
            index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk, giữ nguyên các field còn lại."""
    vectors = embed_texts([chunk["content"] for chunk in chunks], prefix="passage: ")
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _chroma_metadata(metadata: dict) -> dict:
    """Chroma không nhận metadata value None, đổi thành chuỗi rỗng.

    Đọc lại vẫn hợp contract vì contract cho phép url là str hoặc None.
    """
    return {key: ("" if value is None else value) for key, value in metadata.items()}


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB theo batch."""
    collection = get_collection()

    for start in range(0, len(chunks), 128):
        batch = chunks[start : start + 128]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Chunked into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded with {EMBEDDING_MODEL} (dim={len(embedded_chunks[0]['embedding'])})")

    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks into collection {COLLECTION_NAME!r}")
    print(f"Collection now holds {get_collection().count()} chunks")


if __name__ == "__main__":
    run_pipeline()
