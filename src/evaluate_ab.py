"""
Evaluation harness — A/B dense-only vs hybrid + RRF, và calibrate threshold.

Owner: Đàm Quang Trung (2A202602525), role Evaluation/Integration.

Harness chia làm hai tầng, tách rời nhau có lý do:

1. Retrieval metrics (deterministic, không cần LLM, không cần API key)
   - hit@k: có chunk nào trong top_k đến từ đúng document chứa câu trả lời.
   - lexical context recall: tỉ lệ token của expected_context xuất hiện trong
     context đã retrieve.
   Đây KHÔNG phải 4 metric mà đề bài yêu cầu. Đây là proxy chạy được ngay để
   so sánh retrieval strategy mà không phụ thuộc vào generation, và để phát hiện
   lỗi retrieval trước khi tốn token cho evaluator.

2. RAGAS metrics (faithfulness, answer relevance, context recall,
   context precision) — cần generation của Task 10 và một evaluator LLM, nên chỉ
   chạy được sau khi Task 9/Task 10 xong.

Config A và Config B chỉ khác duy nhất retrieval strategy; cùng golden dataset,
cùng top_k, cùng generator, cùng evaluator.

Chạy:
    python -m src.evaluate_ab --calibrate     # phân bố score in/out-of-domain
    python -m src.evaluate_ab --retrieval     # hit@k + lexical recall cho A/B
    python -m src.evaluate_ab --ragas         # 4 metric (cần Task 9/10 + API key)
"""

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).parent.parent
EVALUATION_DIR = ROOT / "group_project" / "evaluation"
GOLDEN_PATH = EVALUATION_DIR / "golden_dataset.json"
CALIBRATION_PATH = EVALUATION_DIR / "calibration_queries.json"
RESULTS_PATH = EVALUATION_DIR / "ab_results.json"

TOP_K = 5

# Token quá ngắn ("of", "a") xuất hiện ở mọi chunk nên không phân biệt được
# context đúng với context sai.
MIN_TOKEN_LENGTH = 3


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w']+", text.lower()) if len(token) >= MIN_TOKEN_LENGTH}


def load_golden() -> list[dict]:
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if len(cases) < 15:
        raise SystemExit(f"Golden dataset has {len(cases)} cases, the lab requires at least 15.")
    return cases


# --------------------------------------------------------------------------- #
# Retrieval strategies
# --------------------------------------------------------------------------- #

def dense_only(query: str, top_k: int = TOP_K) -> list[dict]:
    """Config A — chỉ dense search."""
    from .task5_semantic_search import semantic_search

    return semantic_search(query, top_k=top_k)


def bm25_only(query: str, top_k: int = TOP_K) -> list[dict]:
    """Baseline phụ, không nằm trong A/B nhưng giúp giải thích RRF thắng ở đâu."""
    from .task6_lexical_search import lexical_search

    return lexical_search(query, top_k=top_k)


def hybrid_rrf(query: str, top_k: int = TOP_K) -> list[dict]:
    """Config B — dense + BM25 gộp bằng RRF của Task 7."""
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank_rrf

    # Lấy sâu hơn top_k ở mỗi nhánh rồi mới fuse: RRF cần đủ rank để xếp lại.
    dense = semantic_search(query, top_k=top_k * 4)
    lexical = lexical_search(query, top_k=top_k * 4)
    return rerank_rrf([dense, lexical], top_k, 60)


def available_configs() -> dict:
    """Config nào chạy được với code hiện tại trong repo.

    Config B chỉ được đăng ký khi rerank_rrf đã implement thật. Import lỗi và
    NotImplementedError phải xử lý tách nhau: gộp vào một "except Exception" sẽ
    đăng ký Config B cả khi module không import được, rồi vỡ giữa lúc đang eval.
    """
    configs = {"A_dense_only": dense_only, "bm25_only": bm25_only}

    try:
        from .task7_reranking import rerank_rrf
    except ImportError as error:
        print(f"SKIP config B: cannot import task7_reranking ({error}).")
        return configs

    try:
        rerank_rrf([], 1, 60)
    except NotImplementedError:
        print("SKIP config B: task7_reranking.rerank_rrf is not implemented yet (owner: Tuan).")
        return configs
    except Exception:
        # Hàm đã implement thật; lỗi ở đây chỉ vì gọi với ranked_lists rỗng.
        pass

    configs["B_hybrid_rrf"] = hybrid_rrf
    return configs


# --------------------------------------------------------------------------- #
# 1. Retrieval metrics
# --------------------------------------------------------------------------- #

def score_case(case: dict, results: list[dict]) -> dict:
    """hit@k và lexical context recall cho một golden case."""
    expected_tokens = tokenize(case["expected_context"])
    retrieved_tokens: set[str] = set()
    hit = False

    for item in results:
        retrieved_tokens |= tokenize(item["content"])
        if item["id"].startswith(case["expected_source"]):
            hit = True

    recall = (
        len(expected_tokens & retrieved_tokens) / len(expected_tokens)
        if expected_tokens
        else 0.0
    )
    return {"hit": hit, "lexical_recall": recall}


def run_retrieval_eval(top_k: int = TOP_K) -> dict:
    cases = load_golden()
    configs = available_configs()
    report: dict = {"top_k": top_k, "golden_cases": len(cases), "configs": {}}

    for name, retrieve_fn in configs.items():
        hits, recalls, per_case = 0, [], []
        for case in cases:
            results = retrieve_fn(case["question"], top_k)
            scored = score_case(case, results)
            hits += int(scored["hit"])
            recalls.append(scored["lexical_recall"])
            per_case.append({"id": case["id"], **scored, "returned": len(results)})

        report["configs"][name] = {
            "hit_at_k": round(hits / len(cases), 4),
            "mean_lexical_recall": round(statistics.mean(recalls), 4),
            "worst_cases": sorted(per_case, key=lambda item: item["lexical_recall"])[:3],
        }
        print(
            f"{name:16s}  hit@{top_k}={hits}/{len(cases)}"
            f"  mean_lexical_recall={statistics.mean(recalls):.4f}"
        )

    return report


# --------------------------------------------------------------------------- #
# 2. Threshold calibration
# --------------------------------------------------------------------------- #

def run_calibration() -> dict:
    """Phân bố dense top-1 score của query in-domain vs out-of-domain."""
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

    distribution: dict[str, list[float]] = {}
    for label in ("in_domain", "out_of_domain"):
        scores = []
        for query in data[label]:
            results = dense_only(query, top_k=1)
            scores.append(round(results[0]["score"], 4) if results else 0.0)
        distribution[label] = scores
        print(f"{label:14s} n={len(scores)} min={min(scores):.4f} max={max(scores):.4f} mean={statistics.mean(scores):.4f}")

    in_min = min(distribution["in_domain"])
    out_max = max(distribution["out_of_domain"])
    separated = in_min > out_max

    print(f"\nin_domain min      = {in_min:.4f}")
    print(f"out_of_domain max  = {out_max:.4f}")
    print(f"separated          = {separated}")

    sweep = sweep_threshold(distribution)
    best = sweep["best"]
    print(
        f"best threshold     = {best['threshold']:.4f}"
        f"  (accuracy={best['accuracy']:.2f},"
        f" false_accept={best['false_accept']}/{len(distribution['out_of_domain'])},"
        f" false_reject={best['false_reject']}/{len(distribution['in_domain'])})"
    )
    if not separated:
        print(
            "NOTE: distributions overlap, so no threshold classifies every query "
            "correctly. Report the error rates above instead of a clean cut."
        )

    return {
        "distribution": distribution,
        "in_domain_min": in_min,
        "out_of_domain_max": out_max,
        "separated": separated,
        "recommended_score_threshold": best["threshold"],
        "threshold_sweep": sweep,
    }


def sweep_threshold(distribution: dict[str, list[float]]) -> dict:
    """Quét threshold và chọn điểm ít lỗi nhất.

    Phân bố in-domain và out-of-domain chồng nhau thì không có threshold nào
    đúng hết, nên chọn theo accuracy và báo cáo kèm số ca lỗi mỗi phía:
    false_accept là query ngoài domain bị trả lời, false_reject là query đúng
    domain bị từ chối oan. Với chatbot chính sách, false_accept nguy hiểm hơn.
    """
    in_domain = distribution["in_domain"]
    out_domain = distribution["out_of_domain"]

    candidates = sorted({round(score, 4) for score in in_domain + out_domain})
    table = []
    for threshold in candidates:
        false_reject = sum(1 for score in in_domain if score < threshold)
        false_accept = sum(1 for score in out_domain if score >= threshold)
        table.append(
            {
                "threshold": threshold,
                "false_reject": false_reject,
                "false_accept": false_accept,
                "accuracy": round(
                    1 - (false_reject + false_accept) / (len(in_domain) + len(out_domain)), 4
                ),
            }
        )

    # Hoà accuracy thì ưu tiên ít false_accept hơn.
    best = max(table, key=lambda row: (row["accuracy"], -row["false_accept"]))
    return {"best": best, "table": table}


# --------------------------------------------------------------------------- #
# 3. RAGAS metrics
# --------------------------------------------------------------------------- #

def run_ragas(top_k: int = TOP_K) -> dict:
    """4 metric bắt buộc. Cần Task 10 và một evaluator LLM."""
    try:
        from .task10_generation import generate_with_citation
    except ImportError as error:
        raise SystemExit(f"Cannot import task10_generation: {error}")

    cases = load_golden()
    configs = available_configs()
    if "B_hybrid_rrf" not in configs:
        raise SystemExit(
            "Config B needs task7_reranking.rerank_rrf. Wait for Tuan's branch "
            "feat/fusion-fallback before running the A/B."
        )

    try:
        generate_with_citation(cases[0]["question"], top_k)
    except NotImplementedError:
        raise SystemExit(
            "task10_generation.generate_with_citation is not implemented yet "
            "(owner: Tuan). Run --retrieval in the meantime."
        )

    raise SystemExit(
        "TODO(Trung): wire ragas evaluate() here once Task 9/10 land.\n"
        "Plan: build a ragas EvaluationDataset from question / answer / contexts / "
        "reference, then evaluate with faithfulness, answer_relevancy, "
        "context_recall, context_precision for Config A and Config B."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG A/B evaluation harness")
    parser.add_argument("--calibrate", action="store_true", help="dense score distribution in vs out of domain")
    parser.add_argument("--retrieval", action="store_true", help="hit@k and lexical recall per config")
    parser.add_argument("--ragas", action="store_true", help="the 4 required metrics (needs task 9/10)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    if not any((args.calibrate, args.retrieval, args.ragas)):
        parser.error("pick at least one of --calibrate / --retrieval / --ragas")

    report: dict = {}
    if args.calibrate:
        print("=== threshold calibration ===")
        report["calibration"] = run_calibration()
    if args.retrieval:
        print("\n=== retrieval metrics ===")
        report["retrieval"] = run_retrieval_eval(args.top_k)
    if args.ragas:
        print("\n=== ragas metrics ===")
        report["ragas"] = run_ragas(args.top_k)

    existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    existing.update(report)
    RESULTS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
