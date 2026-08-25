"""Measure the cross-encoder reranker (src/rerank.py) against data/eval.jsonl.

For each question, takes the same top-20 article-deduplicated hybrid results
(at the winning DENSE_WEIGHT from the RRF sweep) as both the "before" ranking
and the candidate pool the reranker reorders — so the comparison isolates
what reranking itself changes, on a fixed candidate set. Reranking cannot
recover an article ranked below 20th in the original retrieval; recall@10
here is a ceiling test of the hybrid retriever's top-20, not the reranker.

Usage:
    .venv/bin/python src/evaluate_rerank.py
"""
import json
import time
from pathlib import Path

from evaluate import RECALL_KS, load_eval_set, rank_of_expected
from search import Searcher
from rerank import Reranker, RERANKER_MODEL

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_PATH = DATA_DIR / "rerank_results.json"

CANDIDATE_POOL = 20


def metrics_for_ranking(questions: list, ranked_lists: dict) -> dict:
    in_scope = [q for q in questions if q["expected_articles"]]
    ranks = []
    for q in in_scope:
        expected = set(q["expected_articles"])
        ranks.append(rank_of_expected(ranked_lists[q["id"]], expected))
    out = {"n": len(in_scope)}
    for K in RECALL_KS:
        out[f"recall@{K}"] = sum(1 for r in ranks if 0 < r <= K) / len(ranks)
    out["mrr"] = sum((1.0 / r) if r > 0 else 0.0 for r in ranks) / len(ranks)
    return out


def main():
    questions = load_eval_set()
    print(f"Loaded {len(questions)} eval questions")

    print("Loading searcher (hybrid retrieval, current DENSE_WEIGHT)...")
    searcher = Searcher()
    print(f"Loading cross-encoder reranker ({RERANKER_MODEL})...")
    reranker = Reranker()

    before_lists = {}
    after_lists = {}
    latencies = []

    print(f"Running top-{CANDIDATE_POOL} hybrid retrieval + rerank for each question...")
    for i, q in enumerate(questions, 1):
        candidates = searcher.search(q["question"], k=CANDIDATE_POOL)
        before_lists[q["id"]] = [c["article"] for c in candidates]

        t0 = time.perf_counter()
        reranked = reranker.rerank(q["question"], list(candidates))
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        after_lists[q["id"]] = [c["article"] for c in reranked]
        if i % 25 == 0 or i == len(questions):
            print(f"  {i}/{len(questions)}")

    before_metrics = metrics_for_ranking(questions, before_lists)
    after_metrics = metrics_for_ranking(questions, after_lists)

    avg_latency_ms = sum(latencies) / len(latencies) * 1000
    p50_latency_ms = sorted(latencies)[len(latencies) // 2] * 1000
    p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)] * 1000

    print()
    print("=" * 70)
    print(f"BEFORE vs AFTER RERANKING (candidate pool = top-{CANDIDATE_POOL} hybrid results)")
    print("=" * 70)
    print(f"{'metric':<12}{'before':>10}{'after':>10}{'delta':>10}")
    for k in RECALL_KS:
        b, a = before_metrics[f"recall@{k}"], after_metrics[f"recall@{k}"]
        print(f"recall@{k:<5}{b:>10.3f}{a:>10.3f}{a-b:>+10.3f}")
    b, a = before_metrics["mrr"], after_metrics["mrr"]
    print(f"{'mrr':<12}{b:>10.3f}{a:>10.3f}{a-b:>+10.3f}")

    print()
    print("=" * 70)
    print("ADDED LATENCY (reranking step only, per query)")
    print("=" * 70)
    print(f"mean: {avg_latency_ms:.1f}ms  p50: {p50_latency_ms:.1f}ms  p95: {p95_latency_ms:.1f}ms")

    verdict = "KEEP" if after_metrics["recall@3"] > before_metrics["recall@3"] else "DROP"
    print()
    print(f"Verdict on recall@3: {verdict} "
          f"(before={before_metrics['recall@3']:.3f}, after={after_metrics['recall@3']:.3f})")

    results = {
        "reranker_model": RERANKER_MODEL,
        "candidate_pool_size": CANDIDATE_POOL,
        "n_questions": len(questions),
        "before": before_metrics,
        "after": after_metrics,
        "latency_ms": {"mean": avg_latency_ms, "p50": p50_latency_ms, "p95": p95_latency_ms},
        "verdict_recall_at_3": verdict,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
