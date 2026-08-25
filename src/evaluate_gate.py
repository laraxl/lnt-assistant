"""Measure the jurisdiction gate (src/gate.py) against data/eval.jsonl.

Runs the LLM classifier over all 150 questions and reports two separate
things that must not be conflated:

1. Retrieval impact: only route=="REDIRECT" blocks retrieval (no articles
   returned). EXCLUDED still retrieves normally (the point of EXCLUDED is
   "answer with article 3 plus a flag", not silence) and ALLOW is the
   default. So "in-scope wrongly blocked" now only counts REDIRECT
   misfires — this is what evaluate_e2e.py's recall@3 measures.

2. Harm prevented: of the 23 out-of-scope questions, how many get a
   *correct* redirect/exclusion instead of a confident wrong citation to LNT
   articles that don't actually apply to them. This is the number recall@3
   cannot see — a question the gate never let reach retrieval doesn't show
   up as a retrieval failure, but a bad redirect (or a confident ALLOW on a
   federally-regulated worker) is still real harm to the user, and needs its
   own count. "Correct" is judged per out-of-scope subcategory:
     - federal / construction: route=="REDIRECT" AND the redirect text names
       the right body (Canada Labour Code / Labour Program, or CCQ).
     - unrelated (not an employment question at all): route=="REDIRECT".
     - self-employed / independent contractor: route in {"REDIRECT",
       "EXCLUDED"} — i.e. NOT a confident ALLOW telling them LNT provisions
       apply, since a genuine gig-work/contractor line is fact-dependent and
       the harm is a false confident yes, not the hedge itself.

Usage:
    ANTHROPIC_API_KEY=... .venv/bin/python src/evaluate_gate.py
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

import gate
from gate import classify, query_cost

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_PATH = DATA_DIR / "eval.jsonl"
RESULTS_PATH = DATA_DIR / "gate_results.json"

MAX_WORKERS = 8

FEDERAL_REDIRECT_KEYWORDS = re.compile(
    r"canada labour code|labour program|code canadien du travail|programme du travail"
    r"|federal|f[ée]d[ée]ral|emploi et d[ée]veloppement social canada|esdc",
    re.IGNORECASE,
)
CONSTRUCTION_REDIRECT_KEYWORDS = re.compile(r"\bccq\b|construction", re.IGNORECASE)


def load_eval_set() -> list:
    return [json.loads(l) for l in EVAL_PATH.open(encoding="utf-8")]


def run_one(client, q):
    t0 = time.perf_counter()
    result, usage = classify(q["question"], client=client)
    elapsed = time.perf_counter() - t0
    return {
        "id": q["id"],
        "question": q["question"],
        "language": q["language"],
        "category": q["category"],
        "is_out_of_scope_label": not q["expected_articles"],
        "route": result.route,
        "confident": result.confident,
        "reason": result.reason,
        "redirect": result.redirect,
        "cost_usd": query_cost(usage),
        "latency_s": elapsed,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
    }


def judge_oos_correctness(row: dict) -> bool:
    """Was the gate's handling of this out-of-scope question actually
    correct (redirect/exclusion), as opposed to a confident wrong citation?"""
    cat = row["category"]
    redirect_text = row["redirect"] or ""
    if cat == "out_of_scope_federal":
        return row["route"] == "REDIRECT" and bool(FEDERAL_REDIRECT_KEYWORDS.search(redirect_text))
    if cat == "out_of_scope_construction":
        return row["route"] == "REDIRECT" and bool(CONSTRUCTION_REDIRECT_KEYWORDS.search(redirect_text))
    if cat == "out_of_scope_unrelated":
        return row["route"] == "REDIRECT"
    if cat == "out_of_scope_self_employed":
        return row["route"] in ("REDIRECT", "EXCLUDED")
    raise ValueError(f"unhandled out-of-scope category: {cat}")


def main():
    questions = load_eval_set()
    client = anthropic.Anthropic()

    print(f"Classifying {len(questions)} questions with {MAX_WORKERS} workers (model={gate.MODEL})...")
    t_start = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, client, q): q for q in questions}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 25 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)}")
    total_wall_s = time.perf_counter() - t_start

    rows.sort(key=lambda r: r["id"])

    # Retrieval impact: only REDIRECT blocks. EXCLUDED and ALLOW both retrieve.
    for r in rows:
        r["blocked"] = r["route"] == "REDIRECT"

    oos_rows = [r for r in rows if r["is_out_of_scope_label"]]
    in_scope_rows = [r for r in rows if not r["is_out_of_scope_label"]]

    for r in oos_rows:
        r["oos_correct"] = judge_oos_correctness(r)

    route_counts_oos = {rt: sum(1 for r in oos_rows if r["route"] == rt) for rt in ("REDIRECT", "EXCLUDED", "ALLOW")}
    route_counts_in_scope = {rt: sum(1 for r in in_scope_rows if r["route"] == rt) for rt in ("REDIRECT", "EXCLUDED", "ALLOW")}

    in_scope_blocked = route_counts_in_scope["REDIRECT"]
    oos_correct = sum(1 for r in oos_rows if r["oos_correct"])
    oos_confident_wrong = sum(1 for r in oos_rows if not r["oos_correct"] and r["confident"])
    oos_hedged_wrong = sum(1 for r in oos_rows if not r["oos_correct"] and not r["confident"])

    n_unclear_oos = sum(1 for r in oos_rows if not r["confident"])
    n_unclear_in_scope = sum(1 for r in in_scope_rows if not r["confident"])

    total_cost = sum(r["cost_usd"] for r in rows)
    avg_cost = total_cost / len(rows)
    avg_latency = sum(r["latency_s"] for r in rows) / len(rows)

    print()
    print("=" * 70)
    print("ROUTE DISTRIBUTION")
    print("=" * 70)
    print(f"{'':20s}{'REDIRECT':>12s}{'EXCLUDED':>12s}{'ALLOW':>12s}")
    print(f"{'out-of-scope (23)':20s}{route_counts_oos['REDIRECT']:>12d}{route_counts_oos['EXCLUDED']:>12d}{route_counts_oos['ALLOW']:>12d}")
    print(f"{'in-scope (127)':20s}{route_counts_in_scope['REDIRECT']:>12d}{route_counts_in_scope['EXCLUDED']:>12d}{route_counts_in_scope['ALLOW']:>12d}")

    print()
    print("=" * 70)
    print("RETRIEVAL IMPACT (what evaluate_e2e.py's recall@3 measures)")
    print("=" * 70)
    print(f"In-scope questions blocked from retrieval (route=REDIRECT): {in_scope_blocked}/{len(in_scope_rows)}")

    print()
    print("=" * 70)
    print("HARM PREVENTED (what recall@3 does NOT measure)")
    print("=" * 70)
    print(f"Out-of-scope questions handled correctly (redirect/exclusion, not a confident wrong citation): "
          f"{oos_correct}/{len(oos_rows)} ({oos_correct/len(oos_rows)*100:.1f}%)")
    print(f"  ...of which route breakdown for the correct ones is above.")
    print(f"Out-of-scope questions given a CONFIDENT wrong citation: {oos_confident_wrong}/{len(oos_rows)}")
    print(f"Out-of-scope questions given a hedged (confident=false) wrong citation: {oos_hedged_wrong}/{len(oos_rows)}")

    print()
    print("=" * 70)
    print("HEDGE RATE (confident=false)")
    print("=" * 70)
    print(f"out-of-scope: {n_unclear_oos}/{len(oos_rows)}   in-scope: {n_unclear_in_scope}/{len(in_scope_rows)}")

    print()
    print("=" * 70)
    print("COST / LATENCY")
    print("=" * 70)
    print(f"Total cost for {len(rows)} queries: ${total_cost:.4f}")
    print(f"Average cost per query: ${avg_cost:.6f}")
    print(f"Average latency per query: {avg_latency:.2f}s "
          f"(wall clock for the whole {MAX_WORKERS}-way parallel run: {total_wall_s:.1f}s)")

    print()
    print("In-scope questions blocked (route=REDIRECT):")
    for r in in_scope_rows:
        if r["blocked"]:
            print(f"  {r['id']} ({r['language']}/{r['category']}): {r['question']}")
            print(f"      reason given: {r['reason']}")

    print()
    print("Out-of-scope questions NOT handled correctly:")
    for r in oos_rows:
        if not r["oos_correct"]:
            print(f"  {r['id']} ({r['language']}/{r['category']}) route={r['route']} confident={r['confident']}: {r['question']}")
            print(f"      reason given: {r['reason']}  redirect={r['redirect']!r}")

    results = {
        "model": gate.MODEL,
        "n_questions": len(rows),
        "note": (
            "recall@3 (see e2e_results.json) does NOT measure the harm this gate exists "
            "to prevent — a question the gate correctly routes to REDIRECT never reaches "
            "retrieval, so it can't show up as a retrieval failure either way. The number "
            "that measures harm prevention is oos_correct / n_out_of_scope below: how many "
            "out-of-scope questions get a correct redirect/exclusion instead of a confident "
            "wrong citation to LNT articles that don't actually apply to them."
        ),
        "route_distribution": {"out_of_scope": route_counts_oos, "in_scope": route_counts_in_scope},
        "retrieval_impact": {
            "in_scope_blocked": in_scope_blocked,
            "in_scope_total": len(in_scope_rows),
        },
        "harm_prevented": {
            "out_of_scope_total": len(oos_rows),
            "out_of_scope_correct": oos_correct,
            "out_of_scope_confident_wrong": oos_confident_wrong,
            "out_of_scope_hedged_wrong": oos_hedged_wrong,
            "correct_rate": oos_correct / len(oos_rows),
        },
        "hedge_rate": {
            "out_of_scope_unclear": n_unclear_oos,
            "out_of_scope_n": len(oos_rows),
            "in_scope_unclear": n_unclear_in_scope,
            "in_scope_n": len(in_scope_rows),
        },
        "total_cost_usd": total_cost,
        "avg_cost_usd": avg_cost,
        "avg_latency_s": avg_latency,
        "total_wall_clock_s": total_wall_s,
        "max_workers": MAX_WORKERS,
        "rows": rows,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
