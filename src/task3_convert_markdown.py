"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

PDF -> Markdown bằng MarkItDown, JSON -> Markdown bằng header + content.

Ngoài file .md cho người đọc, task này ghi thêm data/standardized/manifest.json
map từng file .md sang {title, url, doc_type, source}. Task 4 đọc manifest thay
vì parse lại header trong Markdown: parse text dễ vỡ khi MarkItDown đổi format,
còn metadata.url là thứ citation phụ thuộc vào nên phải chắc chắn.

Chạy lại không tạo file trùng: output ghi theo stem của file gốc, file đã
convert đủ dài thì bỏ qua.
"""

import json
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}
MIN_MARKDOWN_CHARS = 200


def build_header(title: str, url: str | None, doc_type: str) -> str:
    """Header giữ nguồn ngay trong file Markdown cho người đọc và giảng viên."""
    return (
        f"# {title}\n\n"
        f"**Source:** {url or 'local file'}\n\n"
        f"**Doc type:** {doc_type}\n\n"
        "---\n\n"
    )


def convert_legal_docs() -> list[dict]:
    """Convert PDF/DOCX trong landing/legal vào standardized/legal."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources_path = legal_dir / "sources.json"
    sources = (
        json.loads(sources_path.read_text(encoding="utf-8"))
        if sources_path.exists()
        else {}
    )

    converter = MarkItDown()
    entries: list[dict] = []

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in LEGAL_EXTENSIONS:
            continue

        source_info = sources.get(path.name, {})
        title = source_info.get("title") or path.stem.replace("-", " ")
        url = source_info.get("url")
        output = output_dir / f"{path.stem}.md"

        entry = {
            "path": output.relative_to(OUTPUT_DIR).as_posix(),
            "title": title,
            "url": url,
            "doc_type": "legal",
            "source": path.name,
        }
        entries.append(entry)

        if output.exists() and len(output.read_text(encoding="utf-8").strip()) >= MIN_MARKDOWN_CHARS:
            print(f"Skip (already converted): {output.name}")
            continue

        body = converter.convert(str(path)).text_content.strip()
        if len(body) < MIN_MARKDOWN_CHARS:
            raise RuntimeError(
                f"{path.name}: MarkItDown returned only {len(body)} chars. "
                "The PDF may be a scan - pick a text-based source instead."
            )

        output.write_text(build_header(title, url, "legal") + body + "\n", encoding="utf-8")
        print(f"Converted: {output.name} ({len(body):,} chars)")

    return entries


def convert_news_articles() -> list[dict]:
    """Convert JSON trong landing/news vào standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        title = data["title"]
        url = data["url"]
        output = output_dir / f"{path.stem}.md"

        entries.append(
            {
                "path": output.relative_to(OUTPUT_DIR).as_posix(),
                "title": title,
                "url": url,
                "doc_type": "news",
                "source": path.name,
                "date_crawled": data["date_crawled"],
            }
        )

        header = build_header(title, url, "news") + f"**Crawled:** {data['date_crawled']}\n\n"
        body = data["content_markdown"].strip()
        if len(body) < MIN_MARKDOWN_CHARS:
            raise RuntimeError(f"{path.name}: content_markdown only {len(body)} chars")

        output.write_text(header + body + "\n", encoding="utf-8")
        print(f"Converted: {output.name} ({len(body):,} chars)")

    return entries


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing và ghi manifest metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = convert_legal_docs() + convert_news_articles()

    MANIFEST_PATH.write_text(
        json.dumps({entry["path"]: entry for entry in entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    legal = sum(1 for entry in entries if entry["doc_type"] == "legal")
    news = len(entries) - legal
    print(f"Saved Markdown to: {OUTPUT_DIR}")
    print(f"Manifest: {legal} legal + {news} news = {len(entries)} documents")


if __name__ == "__main__":
    convert_all()
