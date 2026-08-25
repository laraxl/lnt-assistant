"""FastAPI backend for the LNT Q&A app.

Loads the embedding model, dense index, and cross-encoder reranker once at
startup (not per request), then exposes:

  POST /ask     {question, lang} -> gate -> hybrid search (reranked) -> answer
  GET  /health  -> {"status": "ok"}

Static frontend is served from ../static.

Usage:
    .venv/bin/uvicorn api:app --app-dir src --reload --port 8000
"""
import json
import threading
from collections import defaultdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from answer import generate_answer
from gate import classify
from search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
ARTICLES_PATH = DATA_DIR / "articles.jsonl"

RATE_LIMIT_PER_DAY = 10
SNIPPET_CHARS = 220
RETRIEVAL_K = 3

# Below this cross-encoder rerank score, a rank-2/3 citation is hidden from
# display rather than shown as padding. Chosen from data/eval.jsonl: across
# 127 in-scope questions, correct and incorrect top-3 rerank scores overlap
# too broadly for any threshold to cleanly separate them (correct-citation
# scores range from 0.0007 to 0.999) — but conditioning on "rank #1 is always
# shown regardless of score" changes the picture, because #1 is virtually
# always the retriever's actual best guess and rarely needs filtering. With
# that rule, 0.05 hides 115 of 268 incorrect rank-2/3 citations across the
# eval set while costing only 6 of 127 questions their only correct citation
# (the ones where the correct article happened to land at rank 2/3, not 1).
# It also clears the motivating case: article 81.4.1 padding a vacation-length
# question at rank 2 scored 0.047. Lower thresholds (0.01-0.03) cost fewer
# correct citations but leave most of the padding visible; higher ones
# (0.1+) start cutting meaningfully more correct citations for diminishing
# padding removal. See the analysis this was swept from in the chat history
# around this change (not persisted as a script — a one-off check against
# data/eval.jsonl, reproducible by reranking each question's top-20 hybrid
# results and comparing rerank_score for expected vs. non-expected articles).
CITATION_RERANK_FLOOR = 0.05

DISCLAIMER = {
    "fr": (
        "Ceci est de l'information juridique générale, pas un avis juridique. "
        "Pour votre situation, consultez la CNESST (cnesst.gouv.qc.ca) ou un "
        "conseiller juridique."
    ),
    "en": (
        "This is general legal information, not legal advice. For your "
        "specific situation, consult the CNESST (cnesst.gouv.qc.ca) or a "
        "legal professional."
    ),
}


# ------------------------------------------------------------------ startup
app = FastAPI(title="LNT Q&A API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"searcher": None, "articles_by_number": None}


def _load_articles() -> dict:
    articles = {}
    with ARTICLES_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            articles[rec["article"]] = rec
    return articles


@app.on_event("startup")
def load_once():
    print("Loading embeddings, e5 model, and FTS index...")
    searcher = Searcher()
    print("Loading cross-encoder reranker...")
    from rerank import Reranker
    searcher._reranker = Reranker()  # force-load now instead of on first request
    print("Loading article text for citation lookup...")
    _state["searcher"] = searcher
    _state["articles_by_number"] = _load_articles()
    print("Ready.")


# -------------------------------------------------------------- rate limit
_rate_lock = threading.Lock()
_rate_state = defaultdict(lambda: {"date": None, "count": 0})


def check_rate_limit(ip: str) -> bool:
    """Returns True if the request is allowed, False if the daily limit for
    this IP is already used up. In-memory, resets at UTC midnight, resets on
    process restart — fine for this scale, not meant to survive a redeploy."""
    today = date.today().isoformat()
    with _rate_lock:
        entry = _rate_state[ip]
        if entry["date"] != today:
            entry["date"] = today
            entry["count"] = 0
        if entry["count"] >= RATE_LIMIT_PER_DAY:
            return False
        entry["count"] += 1
        return True


# -------------------------------------------------------------------- /ask
class AskRequest(BaseModel):
    question: str
    lang: str = "fr"


def _lang(req_lang: str) -> str:
    return req_lang if req_lang in ("fr", "en") else "fr"


def _redirect_answer(route_result, lang: str) -> str:
    text = route_result.reason.strip()
    if route_result.redirect:
        text = f"{text}\n\n{route_result.redirect}"
    return text


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(
            status_code=429,
            detail="Daily limit reached (10 questions per day). Please try again tomorrow.",
        )

    lang = _lang(req.lang)
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    searcher = _state["searcher"]
    articles_by_number = _state["articles_by_number"]

    route_result, _usage = classify(question)

    if route_result.route == "REDIRECT":
        return {
            "route": "REDIRECT",
            "answer": _redirect_answer(route_result, lang),
            "citations": [],
            "disclaimer": DISCLAIMER[lang],
        }

    results = searcher.search(question, k=RETRIEVAL_K, rerank=True)

    citations = []
    context_articles = []
    for i, r in enumerate(results):
        art = articles_by_number.get(r["article"])
        if not art:
            continue
        text = art["text_fr"] if lang == "fr" else art["text_en"]
        url = art["url_fr"] if lang == "fr" else art["url_en"]
        if not text:
            continue
        # Full top-3 always grounds the answer — more context doesn't hurt,
        # and the answer prompt is instructed to say when it doesn't know.
        # But rank 2/3 padding below the relevance floor gets hidden from the
        # citation *display*: showing a barely-related article as citation
        # #2 next to a correct #1 reads as broken, not as "here's more
        # context." See CITATION_RERANK_FLOOR below for where 0.05 came from.
        keep_citation = (i == 0) or (r.get("rerank_score", 1.0) >= CITATION_RERANK_FLOOR)
        context_articles.append({"article": r["article"], "text": text})
        if keep_citation:
            snippet = text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")
            citations.append({"article": r["article"], "url": url, "snippet": snippet})

    caveat = route_result.reason if (route_result.route == "EXCLUDED" or not route_result.confident) else None

    answer_text = generate_answer(question, lang, context_articles, caveat=caveat)

    return {
        "route": route_result.route,
        "answer": answer_text,
        "citations": citations,
        "disclaimer": DISCLAIMER[lang],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# Mounted last so /ask and /health are matched first.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
