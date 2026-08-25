"""Evaluation harness for src/search.py, run against data/eval.jsonl.

Computes recall@1/3/5/10 and MRR over in-scope questions, an out-of-scope
abstention rate based on the raw dense confidence score, breakdowns by
language and by cross-linguality, and a weight sweep over the RRF
dense/keyword balance — so that balance is picked from evidence on 150
questions rather than tuned by hand on 4.

Usage:
    .venv/bin/python src/evaluate.py
"""
import json
import statistics
from pathlib import Path

from search import CHANNEL_TOP_N, RRF_K, Searcher

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_PATH = DATA_DIR / "eval.jsonl"
RESULTS_PATH = DATA_DIR / "eval_results.json"

# label -> (keyword_weight, dense_weight)
WEIGHT_SWEEP = [
    ("1x", 1.0, 1.0),
    ("3x", 1.0, 3.0),
    ("10x", 1.0, 10.0),
    ("30x", 1.0, 30.0),
    ("60x", 1.0, 60.0),
    ("dense-only", 0.0, 1.0),
]

RECALL_KS = (1, 3, 5, 10)
CONFIDENCE_THRESHOLDS = (0.75, 0.78, 0.80, 0.82, 0.83, 0.85, 0.87, 0.90)


def load_eval_set() -> list:
    return [json.loads(l) for l in EVAL_PATH.open(encoding="utf-8")]


def rank_of_expected(ranked_articles: list, expected: set) -> int:
    """1-indexed rank of the best-ranked expected article, or 0 if none of
    the expected articles appear at all."""
    for i, article in enumerate(ranked_articles, 1):
        if article in expected:
            return i
    return 0


def metrics_for(questions: list, ranked_lists: dict) -> dict:
    """recall@K and MRR over a set of in-scope questions.
    ranked_lists: question id -> full ranked list of article numbers."""
    in_scope = [q for q in questions if q["expected_articles"]]
    if not in_scope:
        return {"n": 0}

    ranks = []
    for q in in_scope:
        expected = set(q["expected_articles"])
        ranks.append(rank_of_expected(ranked_lists[q["id"]], expected))

    out = {"n": len(in_scope)}
    for K in RECALL_KS:
        out[f"recall@{K}"] = sum(1 for r in ranks if 0 < r <= K) / len(ranks)
    out["mrr"] = sum((1.0 / r) if r > 0 else 0.0 for r in ranks) / len(ranks)
    return out


def run_weight_sweep(searcher: Searcher, questions: list, channel_cache: dict) -> dict:
    """For each weight setting, re-fuse the cached channel results (no
    re-embedding, no re-querying FTS5) and compute overall + breakdown
    metrics."""
    sweep_results = {}
    for label, kw_w, dense_w in WEIGHT_SWEEP:
        ranked_lists = {}
        for q in questions:
            kw_results, dense_results = channel_cache[q["id"]]
            fused = searcher.fuse(kw_results, dense_results, k=len(searcher.chunks) + 1,
                                   keyword_weight=kw_w, dense_weight=dense_w)
            ranked_lists[q["id"]] = [r["article"] for r in fused]

        overall = metrics_for(questions, ranked_lists)
        by_lang = {
            lang: metrics_for([q for q in questions if q["language"] == lang], ranked_lists)
            for lang in ("fr", "en")
        }
        by_cross = {
            str(flag): metrics_for([q for q in questions if q["cross_lingual"] == flag], ranked_lists)
            for flag in (False, True)
        }
        sweep_results[label] = {
            "keyword_weight": kw_w,
            "dense_weight": dense_w,
            "overall": overall,
            "by_language": by_lang,
            "by_cross_lingual": by_cross,
        }
    return sweep_results


def out_of_scope_analysis(questions: list, top_dense_scores: dict) -> dict:
    oos = [q for q in questions if not q["expected_articles"]]
    in_scope = [q for q in questions if q["expected_articles"]]

    oos_scores = [top_dense_scores[q["id"]] for q in oos]
    in_scope_scores = [top_dense_scores[q["id"]] for q in in_scope]

    def dist(scores):
        if not scores:
            return {}
        return {
            "min": min(scores), "max": max(scores),
            "mean": statistics.mean(scores), "median": statistics.median(scores),
        }

    abstention_by_threshold = {}
    for t in CONFIDENCE_THRESHOLDS:
        oos_abstained = sum(1 for s in oos_scores if s < t) / len(oos_scores) if oos_scores else None
        in_scope_would_lose = sum(1 for s in in_scope_scores if s < t) / len(in_scope_scores) if in_scope_scores else None
        abstention_by_threshold[str(t)] = {
            "oos_abstention_rate": oos_abstained,
            "in_scope_questions_that_would_also_fall_below": in_scope_would_lose,
        }

    return {
        "n_out_of_scope": len(oos),
        "top_dense_score_distribution": {"out_of_scope": dist(oos_scores), "in_scope": dist(in_scope_scores)},
        "abstention_by_threshold": abstention_by_threshold,
    }


def collect_failures(questions: list, ranked_lists: dict, top_dense_scores: dict, k: int = 3) -> list:
    failures = []
    for q in questions:
        ranked = ranked_lists[q["id"]]
        top_k = ranked[:k]
        if q["expected_articles"]:
            expected = set(q["expected_articles"])
            if not (expected & set(top_k)):
                failures.append({
                    "id": q["id"], "question": q["question"], "language": q["language"],
                    "category": q["category"], "expected_articles": q["expected_articles"],
                    "top_k_returned": top_k, "rank_of_expected": rank_of_expected(ranked, expected),
                    "top_dense_score": top_dense_scores[q["id"]],
                    "failure_type": "expected_article_missing_from_top_k",
                })
        else:
            if top_dense_scores[q["id"]] >= 0.83:
                failures.append({
                    "id": q["id"], "question": q["question"], "language": q["language"],
                    "category": q["category"], "expected_articles": [],
                    "top_k_returned": top_k, "top_dense_score": top_dense_scores[q["id"]],
                    "failure_type": "out_of_scope_but_confident",
                })
    return failures


def print_sweep_table(sweep_results: dict):
    print(f"{'weight':<12}{'recall@1':>10}{'recall@3':>10}{'recall@5':>10}{'recall@10':>10}{'mrr':>10}")
    for label, r in sweep_results.items():
        o = r["overall"]
        print(f"{label:<12}{o['recall@1']:>10.3f}{o['recall@3']:>10.3f}{o['recall@5']:>10.3f}"
              f"{o['recall@10']:>10.3f}{o['mrr']:>10.3f}")


def main():
    questions = load_eval_set()
    print(f"Loaded {len(questions)} eval questions "
          f"({sum(1 for q in questions if q['expected_articles'])} in-scope, "
          f"{sum(1 for q in questions if not q['expected_articles'])} out-of-scope)")

    print("Loading searcher (model, embeddings, FTS index)...")
    searcher = Searcher()

    print("Running each question's keyword + dense channels once (cached for the sweep)...")
    channel_cache = {}
    for q in questions:
        kw_results = searcher._keyword_search(q["question"], CHANNEL_TOP_N)
        dense_results = searcher._dense_search(q["question"], CHANNEL_TOP_N)
        channel_cache[q["id"]] = (kw_results, dense_results)

    top_dense_scores = {qid: (dense[0][1] if dense else 0.0) for qid, (_, dense) in channel_cache.items()}

    print("Sweeping RRF weights...")
    sweep_results = run_weight_sweep(searcher, questions, channel_cache)

    print()
    print("=" * 70)
    print("WEIGHT SWEEP (recall@1/3/5/10, MRR over in-scope questions)")
    print("=" * 70)
    print_sweep_table(sweep_results)

    best_label = max(sweep_results, key=lambda l: sweep_results[l]["overall"]["recall@3"])
    print()
    print(f"Winner by recall@3: {best_label} "
          f"(keyword_weight={sweep_results[best_label]['keyword_weight']}, "
          f"dense_weight={sweep_results[best_label]['dense_weight']})")

    # Full breakdown + failures for the winning setting.
    kw_w = sweep_results[best_label]["keyword_weight"]
    dense_w = sweep_results[best_label]["dense_weight"]
    ranked_lists = {}
    for q in questions:
        kw_results, dense_results = channel_cache[q["id"]]
        fused = searcher.fuse(kw_results, dense_results, k=len(searcher.chunks) + 1,
                               keyword_weight=kw_w, dense_weight=dense_w)
        ranked_lists[q["id"]] = [r["article"] for r in fused]

    oos_stats = out_of_scope_analysis(questions, top_dense_scores)
    failures = collect_failures(questions, ranked_lists, top_dense_scores, k=3)

    print()
    print("=" * 70)
    print(f"OUT-OF-SCOPE ANALYSIS (winning weight: {best_label}, n={oos_stats['n_out_of_scope']})")
    print("=" * 70)
    d = oos_stats["top_dense_score_distribution"]
    print(f"top_dense_score  in-scope: mean={d['in_scope']['mean']:.3f} median={d['in_scope']['median']:.3f} "
          f"min={d['in_scope']['min']:.3f} max={d['in_scope']['max']:.3f}")
    print(f"top_dense_score  out-of-scope: mean={d['out_of_scope']['mean']:.3f} median={d['out_of_scope']['median']:.3f} "
          f"min={d['out_of_scope']['min']:.3f} max={d['out_of_scope']['max']:.3f}")
    print()
    print(f"{'threshold':<12}{'oos_abstention_rate':>22}{'in_scope_also_below':>22}")
    for t_str, row in oos_stats["abstention_by_threshold"].items():
        print(f"{t_str:<12}{row['oos_abstention_rate']:>22.3f}{row['in_scope_questions_that_would_also_fall_below']:>22.3f}")

    print()
    print("=" * 70)
    print(f"BREAKDOWN (winning weight: {best_label})")
    print("=" * 70)
    winner = sweep_results[best_label]
    print("By language:")
    for lang, m in winner["by_language"].items():
        print(f"  {lang}: n={m['n']} recall@1={m['recall@1']:.3f} recall@3={m['recall@3']:.3f} "
              f"recall@5={m['recall@5']:.3f} recall@10={m['recall@10']:.3f} mrr={m['mrr']:.3f}")
    print("By cross-lingual (question language != fr, the statute's drafting language):")
    for flag, m in winner["by_cross_lingual"].items():
        print(f"  cross_lingual={flag}: n={m['n']} recall@1={m['recall@1']:.3f} recall@3={m['recall@3']:.3f} "
              f"recall@5={m['recall@5']:.3f} recall@10={m['recall@10']:.3f} mrr={m['mrr']:.3f}")

    print()
    print("=" * 70)
    print(f"FAILURES at k=3 (winning weight: {best_label}), {len(failures)} total")
    print("=" * 70)
    for f in failures:
        print(f"[{f['failure_type']}] {f['id']} ({f['language']}/{f['category']}): {f['question']}")
        if f["failure_type"] == "expected_article_missing_from_top_k":
            print(f"    expected={f['expected_articles']} got_top3={f['top_k_returned']} "
                  f"rank_of_expected={f['rank_of_expected'] or '>len'} top_dense_score={f['top_dense_score']:.3f}")
        else:
            print(f"    got_top3={f['top_k_returned']} top_dense_score={f['top_dense_score']:.3f}")

    results = {
        "n_questions": len(questions),
        "rrf_k": RRF_K,
        "channel_top_n": CHANNEL_TOP_N,
        "weight_sweep": sweep_results,
        "winning_weight_label": best_label,
        "winning_keyword_weight": kw_w,
        "winning_dense_weight": dense_w,
        "out_of_scope_analysis": oos_stats,
        "failures_at_k3": failures,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Saved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
