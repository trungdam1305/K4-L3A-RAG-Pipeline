"""
Task 2 — Crawl bài viết/thông báo về IELTS Writing.

Crawl bằng Crawl4AI (Chromium do Playwright cài). Mỗi bài lưu thành một JSON
trong data/landing/news/ với đủ url, title, date_crawled, content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium

Đặt tên file theo slug của URL thay vì số thứ tự: một URL lỗi thì các file khác
không bị đổi tên khi chạy lại, và không sinh file trùng.

takeielts.britishcouncil.org bị loại khỏi danh sách vì chặn request tự động
(connection reset) — đề bài yêu cầu không vượt WAF nên nhóm chọn nguồn khác.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://ielts.org/take-a-test/preparation-resources/writing-test-resources",
    "https://ieltsliz.com/ielts-writing-task-2/",
    "https://ieltsliz.com/ielts-writing-task-1/",
    "https://www.ieltsadvantage.com/2017/11/02/task-2-coherence-and-cohesion/",
    "https://www.ieltsadvantage.com/ielts-writing-task-2/",
    "https://resources.cathoven.com/ielts-writing-task-2/band-descriptors",
    "https://ieltstutors.org/writing-band-descriptors/",
]

# Bài quá ngắn thường là trang consent/redirect, không phải nội dung thật.
MIN_CONTENT_CHARS = 500


def slugify(url: str) -> str:
    """Đổi URL thành tên file ổn định, không dấu."""
    path = re.sub(r"^https?://(www\.)?", "", url).strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower()
    return slug[:80]


def extract_markdown(result) -> str:
    """Lấy markdown từ CrawlResult, chấp nhận cả str và MarkdownGenerationResult."""
    markdown = getattr(result, "markdown", "") or ""
    for attribute in ("fit_markdown", "raw_markdown"):
        value = getattr(markdown, attribute, None)
        if value:
            return str(value)
    return str(markdown)


def extract_title(result, url: str) -> str:
    """Lấy title, fallback về slug nếu trang không khai báo."""
    metadata = getattr(result, "metadata", None) or {}
    title = (metadata.get("title") or "").strip()
    return title or slugify(url)


async def crawl_article(url: str, crawler=None) -> dict:
    """Crawl một URL và trả về dict đúng 4 field bắt buộc."""
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    if crawler is None:
        async with AsyncWebCrawler() as owned_crawler:
            return await crawl_article(url, owned_crawler)

    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(page_timeout=60_000, word_count_threshold=10),
    )
    if not getattr(result, "success", True):
        raise RuntimeError(getattr(result, "error_message", "crawl failed"))

    content = extract_markdown(result).strip()
    if len(content) < MIN_CONTENT_CHARS:
        raise RuntimeError(f"content too short ({len(content)} chars)")

    return {
        "url": url,
        "title": extract_title(result, url),
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": content,
    }


async def crawl_all() -> None:
    """Crawl toàn bộ ARTICLE_URLS, dùng chung một browser instance."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved, failed = 0, 0

    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        for url in ARTICLE_URLS:
            output = DATA_DIR / f"{slugify(url)}.json"
            try:
                article = await crawl_article(url, crawler)
            except Exception as error:
                failed += 1
                print(f"FAILED: {url} - {error}")
                continue

            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved += 1
            print(f"Saved: {output.name} ({len(article['content_markdown']):,} chars)")

    print(f"Done: {saved} saved, {failed} failed")
    if saved < 5:
        raise SystemExit(
            f"Only {saved} articles saved, the lab requires at least 5. "
            "Add more public URLs to ARTICLE_URLS and re-run."
        )


if __name__ == "__main__":
    asyncio.run(crawl_all())
