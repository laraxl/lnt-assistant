"""End-to-end evaluation: jurisdiction gate -> hybrid retrieval -> rerank.

Uses the gate decisions already measured in data/gate_results.json (rerunning
150 LLM calls here would just burn money re-measuring the same thing) and
runs hybrid+rerank fresh for every question the gate allows through
(applies="true" or "unclear" — same ALLOW rule as evaluate_gate.py). A
question the gate blocks contributes an automatic miss (rank 0) to recall,
regardless of what retrieval would have found — that's the real cost of a
wrong block, and it's exactly what should show up if the gate hurts more
than it helps.

Usage:
    .venv/bin/python src/evaluate_e2e.py
"""
import json
from pathlib import Path

from evaluate import RECALL_KS, load_eval_set, rank_of_expected
from search import Searcher
from rerank import Reranker

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GATE_RESULTS_PATH = DATA_DIR / "gate_results.json"
RESULTS_PATH = DATA_DIR / "e2e_results.json"

CANDIDATE_POOL = 20
NO_GATE_RERANK_RECALL_AT_3 = 0.874  # from data/rerank_results.json ("after")


def main():
    questions = {q["id"]: q for q in load_eval_set()}
    gate_rows = {r["id"]: r for r in json.loads(GATE_RESULTS_PATH.read_text(encoding="utf-8"))["rows"]}

    in_scope_ids = [qid for qid, q in questions.items() if q["expected_articles"]]
    print(f"{len(in_scope_ids)} in-scope questions")

    blocked_ids = [qid for qid in in_scope_ids if gate_rows[qid]["blocked"]]
    allowed_ids = [qid for qid in in_scope_ids if not gate_rows[qid]["blocked"]]
    print(f"  gate blocks {len(blocked_ids)}, allows {len(allowed_ids)}")

    print("Loading searcher + reranker...")
    searcher = Searcher()
    reranker = Reranker()

    ranked_lists = {}
    for qid in blocked_ids:
        ranked_lists[qid] = []  # blocked -> nothing returned -> automatic miss

    print(f"Running hybrid retrieval + rerank for the {len(allowed_ids)} allowed questions...")
    for i, qid in enumerate(allowed_ids, 1):
        q = questions[qid]
        candidates = searcher.search(q["question"], k=CANDIDATE_POOL)
        reranked = reranker.rerank(q["question"], candidates)
        ranked_lists[qid] = [c["article"] for c in reranked]
        if i % 25 == 0 or i == len(allowed_ids):
            print(f"  {i}/{len(allowed_ids)}")

    ranks = []
    for qid in in_scope_ids:
        expected = set(questions[qid]["expected_articles"])
        ranks.append(rank_of_expected(ranked_lists[qid], expected))

    metrics = {"n": len(ranks)}
    for K in RECALL_KS:
        metrics[f"recall@{K}"] = sum(1 for r in ranks if 0 < r <= K) / len(ranks)
    metrics["mrr"] = sum((1.0 / r) if r > 0 else 0.0 for r in ranks) / len(ranks)

    print()
    print("=" * 70)
    print("END-TO-END: gate -> hybrid -> rerank (in-scope questions only)")
    print("=" * 70)
    for K in RECALL_KS:
        print(f"recall@{K}: {metrics[f'recall@{K}']:.3f}")
    print(f"mrr: {metrics['mrr']:.3f}")

    print()
    delta = metrics["recall@3"] - NO_GATE_RERANK_RECALL_AT_3
    print(f"recall@3 WITHOUT gate (hybrid+rerank only, measured earlier): {NO_GATE_RERANK_RECALL_AT_3:.3f}")
    print(f"recall@3 WITH gate (end-to-end):                              {metrics['recall@3']:.3f}")
    print(f"delta: {delta:+.3f}")
    if delta < 0:
        print("=> The gate COSTS recall@3. Its false-block rate is not covered by what it screens out.")
    else:
        print("=> The gate does not cost recall@3 on this eval set.")

    # Failures purely attributable to the gate (would have been found, but blocked).
    gate_caused_misses = []
    for qid in blocked_ids:
        q = questions[qid]
        gate_caused_misses.append({
            "id": qid, "question": q["question"], "expected_articles": q["expected_articles"],
            "gate_reason": gate_rows[qid]["reason"],
        })

    gate_data = json.loads(GATE_RESULTS_PATH.read_text(encoding="utf-8"))
    harm_prevented = gate_data.get("harm_prevented")

    print()
    print("=" * 70)
    print("BOTH NUMBERS SIDE BY SIDE")
    print("=" * 70)
    print(f"end-to-end recall@3 (gate -> hybrid -> rerank): {metrics['recall@3']:.3f}  "
          f"(no-gate baseline: {NO_GATE_RERANK_RECALL_AT_3:.3f}, delta {delta:+.3f})")
    if harm_prevented:
        print(f"out-of-scope questions correctly redirected/excluded (not a confident wrong citation): "
              f"{harm_prevented['out_of_scope_correct']}/{harm_prevented['out_of_scope_total']} "
              f"({harm_prevented['correct_rate']*100:.1f}%)")
    print("recall@3 measures retrieval accuracy on in-scope questions. It does NOT measure "
          "the harm the gate exists to prevent — a correctly-redirected out-of-scope question "
          "never touches retrieval, so it cannot appear in this number either way. The two "
          "numbers have to be read together, not netted into one.")

    results = {
        "note": (
            "recall@3 measures retrieval accuracy on in-scope questions only. It does NOT "
            "measure the harm this gate exists to prevent: a correctly-redirected "
            "out-of-scope question never reaches retrieval, so it can't show up here either "
            "as a gain or a loss. The harm-prevention number is harm_prevented_summary below "
            "(and the full detail in gate_results.json) — read the two together, not as one "
            "combined score."
        ),
        "n_in_scope": len(in_scope_ids),
        "n_blocked_by_gate": len(blocked_ids),
        "n_allowed_by_gate": len(allowed_ids),
        "metrics": metrics,
        "no_gate_rerank_recall_at_3": NO_GATE_RERANK_RECALL_AT_3,
        "delta_recall_at_3": delta,
        "harm_prevented_summary": harm_prevented,
        "gate_caused_misses": gate_caused_misses,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
