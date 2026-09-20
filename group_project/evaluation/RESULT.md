# RAG evaluation result

## Reproducibility

| Item | Value |
|---|---|
| Evaluation date | 2026-09-20 |
| Framework | Custom retrieval harness + RAGAS 0.4.3 integration |
| Generator/evaluator | OpenAI `gpt-4o-mini` |
| Embedding model | `intfloat/multilingual-e5-small` (384 dimensions) |
| Corpus version | 12 documents, 781 chunks, commit `b42aa7f` baseline |
| Golden dataset | 20 grounded questions |
| `top_k` | 5 |
| Fallback threshold | `0.8689` from 12 in-domain and 10 out-of-domain queries |

Commands:

```bash
python -m src.task4_chunking_indexing
python -m src.evaluate_ab --calibrate --retrieval
python -m src.evaluate_ab --ragas
```

The retrieval, calibration and RAGAS results below were measured locally on the
same 20-question golden dataset.

## Overall scores

| Metric | A: dense-only | B: hybrid + RRF | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.7950 | 0.8792 | +0.0842 |
| Answer relevance | 0.8622 | 0.8149 | -0.0473 |
| Context recall | 0.7500 | 0.8250 | +0.0750 |
| Context precision | 0.7961 | 0.8032 | +0.0071 |
| **RAGAS average** | **0.8008** | **0.8306** | **+0.0298** |
| Retrieval hit@5 | 0.9500 | 0.9000 | -0.0500 |
| Lexical context recall | 0.8910 | 0.8585 | -0.0325 |

Per-case RAGAS values are stored in
`group_project/evaluation/ab_results.json`.

## A/B comparison

- **Config A — dense-only:** e5 query embedding against the Chroma cosine index.
- **Config B — hybrid + RRF:** dense top-20 plus BM25 top-20, fused once with
  reciprocal rank fusion and truncated to top-5.
- **Better end-to-end configuration:** Config B. Its RAGAS average is `0.8306`,
  ahead of Config A by `0.0298`, mainly from faithfulness and context recall.
- **Retrieval-only result:** Config A still leads by `0.05` hit@5 and `0.0325`
  lexical recall. These proxy metrics do not measure how the generator uses all
  retrieved chunks, which explains why their ranking differs from RAGAS.
- **Latency:** after one-time model/index warm-up, Config A took 0.4575 seconds
  for 20 queries and Config B took 0.4841 seconds. BM25-only took 0.0283 seconds.
- **Interpretation:** BM25 occasionally displaces a relevant dense result, but
  the additional lexical evidence improves grounded generation overall. Hybrid
  gains `0.0842` faithfulness and `0.0750` context recall while losing `0.0473`
  answer relevance.

## Threshold calibration

Dense top-1 cosine scores:

| Query group | n | min | mean | max |
|---|---:|---:|---:|---:|
| In-domain | 12 | 0.8312 | 0.8918 | 0.9309 |
| Out-of-domain | 10 | 0.7873 | 0.8220 | 0.8477 |

The distributions overlap. Threshold `0.8689` gives accuracy `0.9545`, with
zero false accepts across 10 out-of-domain questions and one false reject across
12 in-domain questions. Avoiding unsupported answers is prioritized over the
single additional refusal.

## Worst performers

| # | Question | Config | Retrieval recall | Failure stage | Root cause |
|---:|---|---|---:|---|---|
| 1 | g20 — What does the official guide say about Writing task timing? | A/B | 0.0000 RAGAS average | Retrieval/generation | Both configurations miss usable evidence and the answer is non-grounded or refused. |
| 2 | g09 — What does Band 8 Lexical Resource require? | A | Faithfulness 0, relevance 0, recall 0 | Retrieval | The exact band-descriptor table row is not retrieved reliably. |
| 3 | g07 — What does Band 9 Coherence and Cohesion look like? | A/B | Context recall 0 | Retrieval | Markdown table syntax weakens ranking of the exact criterion row. |

The recurring issue is ranking exact band-descriptor table rows. The correct
document is usually present, but table markup and repeated criterion names make
the precise row difficult to rank in the top five.

## Recommendations

| Priority | Action | Evidence | Expected impact | Verification |
|---:|---|---|---|---|
| 1 | Flatten Markdown table rows into self-contained sentences before indexing. | g02 and g07 are exact table-row failures. | Improve dense and BM25 ranking of band-specific criteria. | Re-index and compare per-case recall. |
| 2 | Tune RRF candidate depth and `k`, or apply a lightweight cross-encoder after fusion. | Config B wins RAGAS but trails Config A by 0.0325 lexical recall. | Preserve grounded generation gains while recovering exact retrieval hits. | Sweep settings on the same 20 cases. |
| 3 | Recalibrate threshold whenever the embedding model or corpus changes. | Score distributions overlap within a narrow range. | Preserve the measured false-accept policy. | Run `--calibrate` and record both error types. |

## Bonus experiments

No bonus experiment is claimed. The advanced reranker, conversation memory and
online deployment remain optional extensions and are excluded from the score.
