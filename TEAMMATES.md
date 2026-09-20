# Thành viên nhóm — Lab Day 8: RAG Pipeline

| Thông tin | Giá trị |
|---|---|
| Lớp | K4 — L3A |
| Repository nhóm | `K4-L3A-RAG-Pipeline` |
| Nhánh mặc định | `main` |
| Nhánh tích hợp | `main` — chứa bản đã review của cả hai người |
| Nhánh của Trung | `trungdam` |
| Nhánh của Tuấn | `thaituan` — đề xuất, đặt đối xứng với `trungdam` |
| Số thành viên | 2 |
| Chủ đề corpus | **IELTS Writing** — tiêu chí chấm và band descriptors. Corpus: 5 tài liệu chính thức (IELTS CDN + Cambridge English) + 7 bài viết công khai = 12 document, 781 chunk |

## Phân công

| # | Họ và tên | Mã học viên | Vai trò | Nhánh làm việc |
|---|---|---|---|---|
| 1 | Đàm Quang Trung | 2A202602525 | Data & Retrieval Foundation + Evaluation | `trungdam` |
| 2 | Thái Hữu Tuấn | 2A202602465 | Fusion, Fallback, Generation & UI | `thaituan` |

Nhóm chỉ có 2 người nên không chia theo 4 role gợi ý của đề bài. Pipeline được cắt tại đúng một ranh giới: **schema `SearchResult` trong [src/contracts.py](src/contracts.py)**. Thành viên #1 sản xuất `SearchResult` (dense + BM25), thành viên #2 tiêu thụ và gộp chúng (RRF → fallback → generation → UI).

### Ranh giới sở hữu file

Hai người không cùng sửa bất kỳ file nào, nên không có merge conflict.

| Thành viên #1 — Trung | Thành viên #2 — Tuấn |
|---|---|
| `src/task1_collect_legal_docs.py` | `src/task7_reranking.py` |
| `src/task2_crawl_news.py` | `src/task8_pageindex_vectorless.py` |
| `src/task3_convert_markdown.py` | `src/task9_retrieval_pipeline.py` |
| `src/task4_chunking_indexing.py` | `src/task10_generation.py` |
| `src/task5_semantic_search.py` | `app.py` |
| `src/task6_lexical_search.py` | |
| `src/evaluate_ab.py` | |
| `data/**` | |
| `group_project/evaluation/**` | |

`src/contracts.py` và `tests/**` là **read-only** — đề bài đã cho sẵn, không ai sửa. Sửa test để làm test pass sẽ mất điểm.

### Cân đối theo rubric (tổng 90 điểm)

| Hạng mục | Điểm | Trung | Tuấn |
|---|---:|---:|---:|
| Dữ liệu có nguồn rõ ràng và chuẩn hóa | 10 | 10 | — |
| Chunking, embedding, vector database | 10 | 10 | — |
| Dense search, BM25 và RRF | 20 | 13 | 7 |
| Retrieval pipeline và fallback | 10 | — | 10 |
| Generation có citation và safe refusal | 15 | — | 15 |
| Chatbot end-to-end, hiển thị nguồn | 10 | — | 10 |
| Golden dataset, 4 metric, A/B, phân tích lỗi | 10 | 10 | — |
| README, khả năng chạy lại, individual report | 5 | 2.5 | 2.5 |
| **Tổng** | **90** | **45.5** | **44.5** |

---

## #1 — Đàm Quang Trung (2A202602525)

### Phase A — Data (task 1–3) — ✅ xong

- Tải **≥ 3** file chính sách (`.pdf` / `.doc` / `.docx`, mỗi file **> 1 KB**) vào `data/landing/legal/`.
- Crawl **≥ 5** bài viết thành `.json` vào `data/landing/news/`; mỗi file bắt buộc có 4 key `url`, `title`, `date_crawled`, `content_markdown` và không key nào rỗng.
- Chuẩn hóa sang `data/standardized/legal/*.md` (≥ 3 file) và `data/standardized/news/*.md` (≥ 5 file), **mỗi file ≥ 200 ký tự** sau khi strip.
- Ghi URL nguồn của từng tài liệu vào `metadata.url`; tài liệu không có URL thì để `None`, không để chuỗi rỗng.

**Xong khi:** `pytest tests/test_acceptance.py -q -k "corpus or standardized"` pass 3 test.

### Phase B — Index & Retrievers (task 4–6) — ✅ xong

- `load_documents()` — **không tham số**; `chunk_documents(documents)` — đúng 1 tham số.
- Chunk phải giữ `id` và toàn bộ `metadata` của document cha, cộng thêm `chunk_index` là **int ≥ 0**.
- `semantic_search(query, top_k)` → list `SearchResult` với `retrieval_method="dense"`, score là **cosine gốc** của Chroma.
- `lexical_search(query, top_k)` → list `SearchResult` với `retrieval_method="bm25"`, chạy BM25 trên **đúng cùng bộ chunk** đã index.
- Cả hai phải: ID unique, sort giảm dần theo `score`, không vượt `top_k`.
- Dense và BM25 **dùng chung một nguồn hàm embedding/tokenizer** — contract test có monkeypatch để kiểm tra điều này.

**Xong khi:** `pytest tests/test_contracts.py -q -k "chunk or semantic or lexical"` pass 3 test.

### Phase C — Evaluation — ⚠️ golden dataset + calibration xong, 4 metric chờ task 9/10

- `group_project/evaluation/golden_dataset.json`: ✅ 20 case, đã verify cả 20 `expected_context` tồn tại verbatim trong corpus. (File gốc của repo rỗng 0 byte, làm test crash `JSONDecodeError` chứ không phải assert fail.)
- 4 metric faithfulness / answer relevance / context recall / context precision: ⏳ **chặn bởi task 9 + task 10**. Harness đã sẵn ở `src/evaluate_ab.py --ragas`, tự báo thiếu gì khi chạy.
- A/B: **Config A dense-only** vs **Config B hybrid + RRF**, giữ nguyên golden dataset, generator, evaluator, prompt và `top_k`; chỉ đổi retrieval strategy.
- Điền `group_project/evaluation/RESULT.md`: **xóa sạch mọi chữ `TODO`** và giữ đủ 4 heading `Overall scores`, `A/B comparison`, `Worst performers`, `Recommendations`.
- Cung cấp cho Tuấn bộ query in-domain và out-of-domain để calibrate `SCORE_THRESHOLD`, rồi ghi con số đã chốt vào dòng "Fallback threshold and calibration".

**Xong khi:** `pytest tests/test_acceptance.py -q -k "golden or evaluation_report"` pass 2 test.

---

## #2 — Thái Hữu Tuấn (2A202602465)

### Phase A — Fusion & Fallback (task 7–9)

- `rerank_rrf(ranked_lists, top_k, k)` — đúng 3 tham số theo thứ tự này.
- RRF chỉ gộp **thứ hạng** (không gộp score gốc), dedupe theo `id`, và gắn `retrieval_method="hybrid"` cho mọi kết quả trả về.
- `pageindex_search(query, top_k)` → `retrieval_method="pageindex"`.
- `retrieve(query, top_k, score_threshold, use_reranking)` — đúng 4 tham số theo thứ tự này.
- `retrieve` **chỉ chạy RRF một lần**; quyết định fallback dựa trên **cosine score gốc của dense retrieval**, không dùng score RRF.
- Provider của `pageindex_search` lỗi thì `retrieve` phải không vỡ, vẫn trả về kết quả hybrid.
- Calibrate `SCORE_THRESHOLD` bằng bộ query Trung cung cấp; không hard-code một con số tùy ý.

**Xong khi:** `pytest tests/test_contracts.py -q -k "rrf or retrieve"` pass 4 test.

Phase này **không cần chờ dữ liệu thật**: contract test của task 7 và task 9 đều dùng `monkeypatch` để fake dense/BM25, nên có thể code và test song song ngay từ đầu buổi.

### Phase B — Generation & UI (task 10 + `app.py`)

- `generate_with_citation(query, top_k)` — đúng 2 tham số, trả `GenerationResult` gồm `answer`, `sources`, `retrieval_source ∈ {hybrid, pageindex, none}`.
- Hàm reorder chunk phải **non-mutating** (không sửa list đầu vào) và context truyền cho LLM phải chứa `title` / `source`.
- Mọi citation trong `answer` phải map được về một phần tử trong `sources` — giữ nguyên `id` xuyên suốt.
- Dispatch theo biến `LLM_PROVIDER` trong `.env`: `openai` | `gemini` | `anthropic`.
- Không đủ evidence → **safe refusal**, với `retrieval_source="none"` và `answer` không rỗng.
- `streamlit run app.py` hiển thị: answer, danh sách source, `retrieval_method` và score.

**Xong khi:** `pytest tests/test_contracts.py -q -k "reorder or generation_result"` pass 2 test, và `streamlit run app.py` trả lời end-to-end kèm citation.

---

## Việc làm chung

| Việc | Khi nào | Ai |
|---|---|---|
| Chốt chủ đề corpus và ghi vào file này | Trước khi thu thập dữ liệu | Cả hai |
| Đọc [docs/MODULE_CONTRACTS.md](docs/MODULE_CONTRACTS.md), thống nhất cách sinh `id` của chunk | 10 phút đầu, trên `main` | Cả hai |
| Chọn `LLM_PROVIDER` + `EMBEDDING_PROVIDER` và điền `.env` cục bộ | 10 phút đầu | Cả hai (mỗi người một `.env` riêng, không commit) |
| Calibrate `SCORE_THRESHOLD` | ✅ đã đo: **0.8689** | Trung đã cấp query + số đo, Tuấn wire vào `retrieve()` |
| Review PR của nhau | Mỗi PR | Người còn lại |
| Demo 3 case: query đúng domain / query ngoài domain / kết quả A/B | Cuối buổi | Cả hai |

## Lộ trình 3 giờ

| Mốc | Phút | Trung | Tuấn |
|---|---:|---|---|
| 0. Setup | 10 | Chốt chủ đề, `.env`, đọc contract | Chốt chủ đề, `.env`, đọc contract |
| 1. Data | 25 | task 1–3: thu thập + chuẩn hóa | task 7: `rerank_rrf` (test bằng monkeypatch) |
| 2. Index & search | 30 | task 4–6: chunk, index, dense, BM25 | task 8–9: pageindex + `retrieve` |
| 3. Fusion & fallback | 25 | Chuẩn bị query in-domain / out-of-domain | Calibrate threshold, chạy contract test |
| 4. Generation & UI | 30 | Viết golden dataset ≥ 15 câu | task 10 + `app.py` |
| 5. Evaluation | 30 | 4 metric, A/B, `RESULT.md` | Sửa lỗi từ phân tích của Trung |
| 6. Demo & handoff | 30 | Individual report, `pytest -q` | Individual report, demo |

## Quy ước Git

- Mỗi người làm trên nhánh của mình (`trungdam` / `thaituan`), mở PR vào `main`. Khi mở PR trên GitHub, **kiểm tra base repository là repo nhóm, không phải `VinUni-AI20k`** — repo này là fork nên GitHub mặc định chọn upstream.
- Commit message ghi rõ task: `feat(task5): dense search returns SearchResult`.
- **Mỗi người tự ghi lại commit mình phụ trách** ngay khi làm, để viết individual report ở mốc cuối.
- Không commit: `.env`, API key, `.venv/`, `data/_tmp_pdf/`, file cache. `.env.example` chỉ chứa tên biến.

## Checklist nộp bài

- [x] Chủ đề corpus đã ghi vào file này — IELTS Writing
- [x] `data/landing/legal/` 5 PDF chính thức, `data/landing/news/` 7 JSON đủ 4 metadata key
- [x] `data/standardized/{legal,news}/` 5 + 7 file, mỗi file ≥ 200 ký tự
- [x] `group_project/evaluation/golden_dataset.json` — 20 case, 100% grounded
- [ ] `group_project/evaluation/RESULT.md` không còn chữ `TODO` — **chặn bởi task 9/10**; phần retrieval + threshold đã điền số thật
- [ ] `streamlit run app.py` chạy end-to-end, hiển thị source + `retrieval_method` + score
- [ ] `pytest -q` xanh toàn bộ
- [x] `reports/2A202602525-trung.md` — individual report của Trung
- [ ] `reports/2A202602465-tuan.md` — individual report của Tuấn
- [ ] Repository không chứa `.env` hay API key

Template individual report ở [group_project/ịndividual/INDIVIDUAL_REPORT.md](group_project/ịndividual/INDIVIDUAL_REPORT.md) (tên thư mục có typo `ị` là của repo gốc, giữ nguyên). Template yêu cầu copy thành `reports/<student-id>-<short-name>.md`, **không** sửa trực tiếp file template. `reports/RESULT.md` và `reports/INDIVIDUAL_REPORT.md` là bản trùng của repo gốc — báo cáo đánh giá phải điền vào `group_project/evaluation/RESULT.md` vì test đọc đúng đường dẫn đó.
