"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Chủ đề nhóm: IELTS Writing — tiêu chí chấm và band descriptors.

Nguồn: CDN chính thức của IELTS (assets.ctfassets.net/unrdeg6se4ke) và
Cambridge Assessment English. Đây là bản "public version" được IELTS phát hành
công khai cho thí sinh và giáo viên, có URL kiểm chứng được.

Chạy lại nhiều lần không tải trùng: file đã tồn tại và > 1 KB thì bỏ qua.
"""

import json
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "sources.json"

# Tên file không dấu, phản ánh đúng nội dung tài liệu.
SOURCES: dict[str, str] = {
    "ielts-writing-band-descriptors-task-1.pdf": (
        "https://assets.ctfassets.net/unrdeg6se4ke/19SJoSvnUYjrHgVhWvuMnC/"
        "42f1b0cb0d7709646a1392d8418646d0/writingbanddescriptorstask1and2.pdf"
    ),
    "ielts-writing-band-descriptors-task-2.pdf": (
        "https://assets.ctfassets.net/unrdeg6se4ke/4AqjlJ7Tp1wLiY1j6DzpOg/"
        "39561e03e8d48ddc7648b479b903c701/Writing-Band-descriptors-Task-2.pdf"
    ),
    "ielts-writing-band-descriptors-2023.pdf": (
        "https://assets.ctfassets.net/unrdeg6se4ke/WfjJuLMvOxuR3JqLvQmTm/"
        "62439e8dcb84a094455b7fb91e7af526/Writing_Band_Descriptors.pdf"
    ),
    "ielts-guide-for-teachers.pdf": (
        "https://assets.ctfassets.net/unrdeg6se4ke/1LgkF1SjQMYi0TFKNEOL1d/"
        "d064817cc940426868a4f146a7eceba8/IELTS_Guide_for_teachers.pdf"
    ),
    "ielts-speaking-band-descriptors.pdf": (
        "https://assets.cambridgeenglish.org/webinars/"
        "ielts-speaking-band-descriptors.pdf"
    ),
}

# Tiêu đề dùng lại ở Task 3/Task 4 làm metadata.title.
TITLES: dict[str, str] = {
    "ielts-writing-band-descriptors-task-1.pdf": "IELTS Writing Task 1 — Band Descriptors (public version)",
    "ielts-writing-band-descriptors-task-2.pdf": "IELTS Writing Task 2 — Band Descriptors (public version)",
    "ielts-writing-band-descriptors-2023.pdf": "IELTS Writing Band Descriptors (updated May 2023)",
    "ielts-guide-for-teachers.pdf": "IELTS Guide for Teachers — test format and assessment criteria",
    "ielts-speaking-band-descriptors.pdf": "IELTS Speaking Band Descriptors (public version)",
}

REQUEST_TIMEOUT = 60
MIN_SIZE_BYTES = 1024
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải các PDF chính sách về data/landing/legal/ và ghi manifest nguồn."""
    setup_directory()

    for filename, url in SOURCES.items():
        target = DATA_DIR / filename

        if target.exists() and target.stat().st_size > MIN_SIZE_BYTES:
            print(f"Skip (already downloaded): {filename}")
            continue

        response = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers=HEADERS, allow_redirects=True
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            raise RuntimeError(
                f"{filename}: source returned {content_type!r}, not a PDF. "
                "URL may have changed - re-check the public source."
            )
        if len(response.content) <= MIN_SIZE_BYTES:
            raise RuntimeError(f"{filename}: downloaded file too small ({len(response.content)} bytes)")

        target.write_bytes(response.content)
        print(f"Saved: {filename} ({len(response.content):,} bytes)")

    write_manifest()


def write_manifest() -> None:
    """Ghi filename -> {url, title} để Task 3 giữ được nguồn của từng PDF.

    Không có manifest này thì metadata.url của tài liệu legal sẽ là None và
    citation trong câu trả lời không dẫn về được nguồn gốc.
    """
    manifest = {
        filename: {"url": url, "title": TITLES[filename]}
        for filename, url in SOURCES.items()
        if (DATA_DIR / filename).exists()
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {MANIFEST_PATH.name} ({len(manifest)} documents)")


if __name__ == "__main__":
    download_documents()
