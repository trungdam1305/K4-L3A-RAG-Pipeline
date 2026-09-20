"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
ROOT = Path(__file__).parent.parent
LEGAL_DIR = ROOT / "data" / "landing" / "legal"
SOURCE_MANIFEST = LEGAL_DIR / "sources.json"
CACHE_PATH = ROOT / "pageindex_doc_ids.json"
PAGEINDEX_BASE_URL = os.getenv("PAGEINDEX_BASE_URL", "https://api.pageindex.ai").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120
POLL_TIMEOUT_SECONDS = 45
POLL_INTERVAL_SECONDS = 2


def _headers() -> dict[str, str]:
    return {"api_key": PAGEINDEX_API_KEY}


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _source_manifest() -> dict[str, dict]:
    if not SOURCE_MANIFEST.exists():
        return {}
    try:
        data = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")

    import requests

    cache = _load_cache()
    manifest = _source_manifest()
    changed = False

    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        cache_key = path.relative_to(ROOT).as_posix()
        if cache.get(cache_key, {}).get("doc_id"):
            continue

        with path.open("rb") as file_handle:
            response = requests.post(
                f"{PAGEINDEX_BASE_URL}/doc/",
                headers=_headers(),
                files={"file": (path.name, file_handle, "application/pdf")},
                data={"if_retrieval": "true"},
                timeout=(10, REQUEST_TIMEOUT_SECONDS),
            )
        response.raise_for_status()
        payload = response.json()
        doc_id = payload.get("doc_id") or payload.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"PageIndex did not return a document ID for {path.name}")

        source = manifest.get(path.name, {})
        cache[cache_key] = {
            "doc_id": doc_id,
            "source": path.name,
            "title": source.get("title") or path.stem.replace("-", " "),
            "url": source.get("url"),
        }
        changed = True

    if changed:
        _write_cache(cache)


def _retrieval_finished(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status", "")).lower()
    if status in {"completed", "complete", "success", "succeeded", "failed", "error"}:
        return True
    return any(key in payload for key in ("results", "nodes", "retrieved_nodes", "data"))


def _poll_retrieval(retrieval_id: str) -> dict[str, Any]:
    import requests

    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(
            f"{PAGEINDEX_BASE_URL}/retrieval/{retrieval_id}/",
            headers=_headers(),
            timeout=(10, REQUEST_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        payload = response.json()
        if _retrieval_finished(payload):
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    return {}


def _candidate_nodes(value: Any):
    if isinstance(value, dict):
        text = next(
            (
                value.get(key)
                for key in ("text", "content", "node_text", "passage", "summary")
                if isinstance(value.get(key), str) and value.get(key).strip()
            ),
            None,
        )
        if text:
            yield value, text.strip()
        for nested in value.values():
            yield from _candidate_nodes(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _candidate_nodes(item)


def _score(node: dict[str, Any], rank: int) -> float:
    for key in ("score", "relevance_score", "similarity", "confidence"):
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 1.0 / rank


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    query = query.strip()
    if not query or top_k <= 0 or not PAGEINDEX_API_KEY:
        return []

    import requests

    cache = _load_cache()
    if not cache:
        upload_documents()
        cache = _load_cache()

    results: list[dict] = []
    seen: set[str] = set()

    for document in cache.values():
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        try:
            response = requests.post(
                f"{PAGEINDEX_BASE_URL}/retrieval/",
                headers=_headers(),
                json={"doc_id": doc_id, "query": query, "thinking": False},
                timeout=(10, REQUEST_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            submission = response.json()
            retrieval_id = submission.get("retrieval_id") or submission.get("id")
            if not isinstance(retrieval_id, str) or not retrieval_id:
                continue
            payload = _poll_retrieval(retrieval_id)
        except (requests.RequestException, ValueError, RuntimeError):
            continue

        for rank, (node, content) in enumerate(_candidate_nodes(payload), start=1):
            node_id = node.get("node_id") or node.get("id") or rank
            item_id = f"pageindex:{doc_id}:{node_id}"
            if item_id in seen:
                continue
            seen.add(item_id)
            results.append(
                {
                    "id": item_id,
                    "content": content,
                    "score": _score(node, rank),
                    "metadata": {
                        "source": document.get("source") or doc_id,
                        "title": document.get("title") or document.get("source") or doc_id,
                        "doc_type": "legal",
                        "url": document.get("url"),
                        "chunk_index": rank - 1,
                    },
                    "retrieval_method": "pageindex",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    upload_documents()
