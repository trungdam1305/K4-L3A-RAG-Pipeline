# RAG evaluation results

> **Trạng thái:** phần retrieval và threshold calibration đã đo xong bằng số thật.
> Bốn metric RAGAS và A/B dense-only vs hybrid + RRF **chưa chạy được** vì cần
> `task7_reranking.rerank_rrf`, `task9_retrieval_pipeline.retrieve` và
> `task10_generation.generate_with_citation` (owner: Thái Hữu Tuấn) cùng một
> evaluator LLM. Các ô còn `TODO` là đúng những ô đó — không điền số phỏng đoán.
>
> Harness: `python -m src.evaluate_ab --calibrate --retrieval --ragas`
> Số liệu thô: `group_project/evaluation/ab_results.json`

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Harness `src/evaluate_ab.py`; RAGAS 0.4.3 cho 4 metric (chưa chạy) |
| Evaluator model                    | TODO — chốt sau khi Tuấn chọn `LLM_PROVIDER` |
| Generator model                    | TODO — phụ thuộc `LLM_PROVIDER` trong `.env` (Task 10) |
| Embedding model                    | `intfloat/multilingual-e5-small`, dim 384, cosine, normalize=True |
| Corpus version/commit              | `team/dev`, 12 documents (5 legal + 7 news), 781 chunks, chunk_size 500 / overlap 50 |
| Golden dataset size                | 20 case (yêu cầu tối thiểu 15), 100% grounded — đã verify `expected_context` tồn tại verbatim trong corpus |
| `top_k`                            | 5 |
| Fallback threshold and calibration | **0.8689** — quét threshold trên 12 query in-domain + 10 out-of-domain, chọn điểm accuracy cao nhất (0.9545), 0 false accept / 1 false reject. Chi tiết ở mục *Threshold calibration* |

## Configurations

- **Config A — dense-only:** `semantic_search(query, top_k=5)` trên ChromaDB, score là cosine similarity gốc (`1 - distance`). Không BM25, không fusion.
- **Config B — hybrid + RRF:** `semantic_search` và `lexical_search` mỗi nhánh lấy `top_k * 4 = 20` kết quả, gộp bằng `rerank_rrf([dense, lexical], top_k=5, k=60)` đúng một lần.

Hai config dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

Bốn metric RAGAS cần generation của Task 10, chưa chạy được.

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     TODO |     TODO |      TODO |
| Answer relevance  |     TODO |     TODO |      TODO |
| Context recall    |     TODO |     TODO |      TODO |
| Context precision |     TODO |     TODO |      TODO |
| **Average**       |     TODO |     TODO |      TODO |

### Retrieval-level results (đã đo, không cần LLM)

Hai chỉ số deterministic chạy được ngay, dùng để bắt lỗi retrieval trước khi tốn token cho evaluator. **Đây không phải 4 metric bắt buộc ở trên.**

- `hit@5`: có ít nhất một chunk trong top-5 đến từ đúng document chứa câu trả lời.
- `lexical context recall`: tỉ lệ token (độ dài ≥ 3) của `expected_context` xuất hiện trong context đã retrieve.

| Config | hit@5 | mean lexical recall |
| ------ | ----: | ------------------: |
| A — dense-only | **0.95** (19/20) | **0.8910** |
| bm25-only (baseline phụ) | 0.85 (17/20) | 0.7817 |
| B — hybrid + RRF | TODO | TODO |

## A/B comparison

- Cấu hình tốt hơn: **TODO** — cần Config B để kết luận.
- Evidence đã có: trên cùng 20 golden case và cùng `top_k=5`, dense-only hơn BM25-only ở cả hai chỉ số retrieval (hit@5 0.95 so với 0.85; lexical recall 0.8910 so với 0.7817). Hai retriever sai ở những case khác nhau — g03 lọt top-5 của dense nhưng recall BM25 chỉ 0.3077 — nên còn dư địa cho RRF gộp lại. Đây là lý do có cơ sở để kỳ vọng Config B thắng, **chưa phải bằng chứng là nó thắng**.
- Trade-off về latency/cost: Config B gọi thêm một lần BM25 trên 781 chunk và fuse hai bảng 20 phần tử. BM25 chạy in-process, không tốn token, chi phí tăng thêm là CPU chứ không phải tiền API. Số đo latency thật: TODO.

## Threshold calibration

Đo bằng `python -m src.evaluate_ab --calibrate`, dùng **cosine score gốc của dense top-1** (không dùng score RRF), đúng theo module contract.

| Nhóm query | n | min | mean | max |
| ---------- | -: | ---: | ---: | ---: |
| in-domain | 12 | 0.8312 | 0.8918 | 0.9309 |
| out-of-domain | 10 | 0.7873 | 0.8220 | 0.8477 |

**Hai phân bố chồng nhau** (in-domain min 0.8312 < out-of-domain max 0.8477), nên không có threshold nào phân loại đúng 100%. Bảng quét quanh điểm tối ưu:

| threshold | false reject (in-domain bị từ chối oan) | false accept (out-of-domain bị trả lời) | accuracy |
| --------: | --: | --: | -----: |
| 0.8312 | 0 | 3 | 0.8636 |
| 0.8477 | 1 | 1 | 0.9091 |
| **0.8689** | **1** | **0** | **0.9545** |
| 0.8813 | 2 | 0 | 0.9091 |
| 0.8949 | 5 | 0 | 0.7727 |

Chọn **0.8689**: với chatbot trả lời theo tài liệu chính sách, false accept (bịa câu trả lời cho câu hỏi ngoài corpus) tai hại hơn false reject (từ chối oan rồi người dùng hỏi lại). Threshold này đẩy false accept về 0 và chỉ đánh đổi 1 trong 12 query in-domain.

Nguyên nhân hai phân bố chồng nhau: e5 nén cosine similarity vào dải hẹp và cao (0.78–0.93 cho mọi thứ, kể cả text không liên quan), nên khoảng cách giữa in-domain và out-of-domain rất mỏng. Đây là đặc tính của model, không phải lỗi corpus.

## Worst performers

Số dưới đây là từ retrieval eval (Config A, dense-only). Cột 4 metric cần RAGAS nên để TODO.

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | g07 — "What does Band 9 Coherence and Cohesion look like?" | A | TODO | TODO | 0.4286 (lexical) | TODO | retrieval | Đáp án nằm trong một **dòng của bảng Markdown** (`\| 9 \| Argument flows seamlessly...`) ở chunk-17. Chunk đó không vào được top-5 của cả dense lẫn BM25. Dense trả về chunk prose chỉ *nhắc đến* "Coherence and Cohesion" (chunk-19, score 0.8552) vì embedding của chunk toàn markup `\|`/`---` ít giống câu hỏi tự nhiên. BM25 cũng trượt vì "band", "coherence", "cohesion" xuất hiện ở rất nhiều chunk nên idf thấp, không có token nào phân biệt được dòng bảng. |
|   2 | g09 — "What does Band 8 Lexical Resource require?" | A | TODO | TODO | 0.5000 (lexical) | TODO | retrieval | Cùng nguyên nhân với g07: dòng Band 8 của bảng Lexical Resource thua các chunk prose về cùng chủ đề (chunk-19 score 0.8560, chunk-24 score 0.8528). |
|   3 | g02 — "What does Band 7 Task Response mean in practice?" | A | TODO | TODO | 0.6000 (lexical) | TODO | retrieval | Cũng là dòng bảng. Ba case tệ nhất đều là bảng band descriptor, không phải ba lỗi độc lập. |

Một case duy nhất miss hoàn toàn document đúng (hit@5 = 19/20) nên vấn đề chính không phải "tìm sai tài liệu" mà là **xếp hạng sai chunk trong đúng tài liệu**.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Ở Task 3, flatten bảng Markdown thành câu tự mô tả: mỗi dòng bảng thành `"Band 9 Coherence and Cohesion: argument flows seamlessly..."` thay vì `\| 9 \| ... \|`. | Cả 3 worst performer (g07, g09, g02) đều là dòng bảng; dense thua vì embedding bị markup lấn, BM25 thua vì thiếu token phân biệt. | Tăng lexical recall của 3 case này; kỳ vọng mean lexical recall Config A vượt 0.90. | Chạy lại `--retrieval` sau khi re-index, so mean lexical recall và recall riêng của g02/g07/g09. |
|        2 | Cân nhắc đổi embedding sang `BAAI/bge-m3` (default gốc của repo) nếu fallback còn sai nhiều. | in-domain và out-of-domain chồng nhau trên e5-small: khoảng cách chỉ 0.0165 giữa in-min và out-max, buộc phải chấp nhận 1 false reject. | Dải score rộng hơn thì threshold tách sạch hơn, false reject về 0. | Chạy lại `--calibrate` với model mới, so `separated` và accuracy tại threshold tốt nhất. |
|        3 | Giữ `SCORE_THRESHOLD=0.8689` trong `.env` và **không hard-code** vào `retrieve()`; đọc từ env để re-calibrate không phải sửa code. | Threshold phụ thuộc model + corpus; đổi một trong hai là con số cũ vô nghĩa. | Re-calibrate thành thao tác đổi env, không phải sửa code. | `pytest tests/test_contracts.py -k retrieve` vẫn pass khi truyền `score_threshold` khác nhau. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| TODO       | TODO     |         TODO |               TODO | TODO       |
