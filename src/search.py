"""Hybrid bilingual retrieval over data/chunks.jsonl.

Combines SQLite FTS5 keyword search (data/index.db) with dense cosine
search over data/embeddings.npy, fused with reciprocal rank fusion (RRF).
Results are deduplicated by article, keeping each article's best-scoring
chunk.

Usage:
    .venv/bin/python src/search.py                  # builds index if needed, runs demo queries
    from search import search
    search("combien de semaines de vacances après 3 ans", k=5)
"""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CHUNK_IDS_PATH = DATA_DIR / "chunk_ids.json"
META_PATH = DATA_DIR / "embedding_meta.json"
INDEX_DB_PATH = DATA_DIR / "index.db"

RRF_K = 60  # standard RRF damping constant
CHANNEL_TOP_N = 100  # how deep each channel is ranked before fusion

# Channel weights for RRF: score(d) = KEYWORD_WEIGHT/(k+rank_kw+1) +
# DENSE_WEIGHT/(k+rank_dense+1).
#
# This was first hand-tuned to 60x on just the 4 example queries from the
# initial build — which turned out to be overfitting: src/evaluate.py, run
# over the 150-question eval set in data/eval.jsonl, swept
# {1x,3x,10x,30x,60x,dense-only} and 60x was NOT the winner. 10x had the best
# recall@3 (0.752 vs 60x's 0.728) and recall@10 (0.904 vs 0.880); 3x had the
# best recall@1 (0.536) and MRR (0.660). Dense-only was competitive
# (recall@3=0.728) but never actually won on any metric, so a real, if
# modest, hybrid benefit survives contact with 150 questions — it just isn't
# 60x. Keyword still matters for exact-term queries the dense model
# paraphrases past (e.g. the tips query's literal "pourboires" match), it's
# just weighted the way the sweep showed it should be. See data/eval_results.json
# for the full breakdown; re-run src/evaluate.py before changing these.
KEYWORD_WEIGHT = 1.0
DENSE_WEIGHT = 10.0

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _load_chunks():
    chunks = [json.loads(l) for l in CHUNKS_PATH.open(encoding="utf-8")]
    return {c["chunk_id"]: c for c in chunks}, chunks


def build_fts_index(chunks=None, force: bool = False):
    """(Re)build the FTS5 keyword index at data/index.db from chunks.jsonl."""
    if INDEX_DB_PATH.exists() and not force:
        return
    if chunks is None:
        _, chunks = _load_chunks()

    if INDEX_DB_PATH.exists():
        INDEX_DB_PATH.unlink()

    conn = sqlite3.connect(INDEX_DB_PATH)
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
        "chunk_id UNINDEXED, article UNINDEXED, lang UNINDEXED, url UNINDEXED, "
        "text, tokenize='unicode61 remove_diacritics 2')"
    )
    conn.executemany(
        "INSERT INTO chunks_fts (chunk_id, article, lang, url, text) VALUES (?, ?, ?, ?, ?)",
        [(c["chunk_id"], c["article"], c["lang"], c["url"], c["text"]) for c in chunks],
    )
    conn.commit()
    conn.close()


class Searcher:
    """Loads the model, embeddings, chunk text, and FTS index once."""

    def __init__(self, keyword_weight: float = KEYWORD_WEIGHT, dense_weight: float = DENSE_WEIGHT, rrf_k: float = RRF_K):
        self.keyword_weight = keyword_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        self.chunks_by_id, self.chunks = _load_chunks()
        build_fts_index(self.chunks)

        self.embeddings = np.load(EMBEDDINGS_PATH)
        self.chunk_ids = json.loads(CHUNK_IDS_PATH.read_text(encoding="utf-8"))
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.model_name = meta["model"]
        self.query_prefix = meta.get("query_prefix", "")
        self.model = SentenceTransformer(self.model_name)

        self.conn = sqlite3.connect(INDEX_DB_PATH)
        self._reranker = None  # lazily loaded only if search(..., rerank=True) is used

    # -- keyword channel ------------------------------------------------
    def _keyword_search(self, query: str, top_n: int) -> list:
        tokens = TOKEN_RE.findall(query)
        if not tokens:
            return []
        match_query = " OR ".join(f'"{t}"' for t in tokens)
        rows = self.conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (match_query, top_n),
        ).fetchall()
        # bm25() in SQLite FTS5 is a *cost*: lower is more relevant.
        return [(chunk_id, score) for chunk_id, score in rows]

    # -- dense channel ----------------------------------------------------
    def _dense_search(self, query: str, top_n: int) -> list:
        q_vec = self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        sims = self.embeddings @ q_vec  # cosine similarity (both L2-normalized)
        top_idx = np.argsort(-sims)[:top_n]
        return [(self.chunk_ids[i], float(sims[i])) for i in top_idx]

    # -- fusion -------------------------------------------------------------
    def fuse(self, kw_results: list, dense_results: list, k: int = 5,
             keyword_weight: float = None, dense_weight: float = None, rrf_k: float = None) -> list:
        """Fuse pre-computed keyword/dense channel results into a deduplicated,
        per-article ranked list. Weights default to the instance's, but can be
        overridden per call (used by the weight sweep in evaluate.py so the
        same cached channel results can be re-fused under different weights
        without re-running the query through the model or FTS5)."""
        keyword_weight = self.keyword_weight if keyword_weight is None else keyword_weight
        dense_weight = self.dense_weight if dense_weight is None else dense_weight
        rrf_k = self.rrf_k if rrf_k is None else rrf_k

        kw_rank = {cid: i for i, (cid, _) in enumerate(kw_results)}
        kw_score = dict(kw_results)
        dense_rank = {cid: i for i, (cid, _) in enumerate(dense_results)}
        dense_score = dict(dense_results)
        # The single highest raw dense cosine similarity found for this query,
        # regardless of fusion weights or rank — a rank-independent confidence
        # signal, unlike the RRF score (which is only ever a sum of 1/(k+rank)
        # terms and carries no information about how good the *best* match
        # actually was in absolute terms).
        top_dense_score = dense_results[0][1] if dense_results else 0.0

        candidate_ids = set(kw_rank) | set(dense_rank)
        fused = []
        for cid in candidate_ids:
            rrf = 0.0
            if cid in kw_rank:
                rrf += keyword_weight / (rrf_k + kw_rank[cid] + 1)
            if cid in dense_rank:
                rrf += dense_weight / (rrf_k + dense_rank[cid] + 1)
            fused.append(
                {
                    "chunk_id": cid,
                    "rrf_score": rrf,
                    "bm25_score": kw_score.get(cid),
                    "dense_score": dense_score.get(cid),
                }
            )
        fused.sort(key=lambda r: r["rrf_score"], reverse=True)

        # Deduplicate by article, keeping the best-scoring chunk for each.
        best_per_article = {}
        for r in fused:
            chunk = self.chunks_by_id[r["chunk_id"]]
            article = chunk["article"]
            if article not in best_per_article or r["rrf_score"] > best_per_article[article]["rrf_score"]:
                best_per_article[article] = {**r, "chunk": chunk}

        results = sorted(best_per_article.values(), key=lambda r: r["rrf_score"], reverse=True)[:k]

        return [
            {
                "article": r["chunk"]["article"],
                "text": r["chunk"]["text"],
                "url": r["chunk"]["url"],
                "lang": r["chunk"]["lang"],
                "score": r["rrf_score"],
                "bm25_score": r["bm25_score"],
                "dense_score": r["dense_score"],
                "top_dense_score": top_dense_score,
            }
            for r in results
        ]

    def search(self, query: str, k: int = 5, rerank: bool = False, rerank_pool: int = 20) -> list:
        kw_results = self._keyword_search(query, CHANNEL_TOP_N)
        dense_results = self._dense_search(query, CHANNEL_TOP_N)
        if not rerank:
            return self.fuse(kw_results, dense_results, k=k)

        # Rerank a wider candidate pool with a cross-encoder (see rerank.py),
        # then cut to k. evaluate_rerank.py measured this on data/eval.jsonl:
        # recall@3 0.756 -> 0.874, recall@1 0.528 -> 0.693, MRR 0.658 -> 0.788,
        # at ~1.1s/query added latency (CPU). Worth it when latency budget
        # allows; the reranker model is loaded lazily and cached on first use.
        pool = self.fuse(kw_results, dense_results, k=rerank_pool)
        if self._reranker is None:
            from rerank import Reranker
            self._reranker = Reranker()
        return self._reranker.rerank(query, pool, top_k=k)


_searcher = None


def search(query: str, k: int = 5, rerank: bool = False) -> list:
    """Hybrid bilingual search. Returns up to k results (one per article),
    each with article, text, url, score (RRF, rank-derived — not a usable
    confidence signal), the underlying bm25_score / dense_score for that
    chunk, and top_dense_score: the raw cosine similarity of the single best
    dense match for this query, repeated on every result. Use
    top_dense_score, not score, to decide whether an answer is confident
    enough to show.

    rerank=True reorders the top 20 hybrid results with a cross-encoder
    before cutting to k — substantially more accurate (see rerank.py /
    evaluate_rerank.py) at the cost of ~1.1s added latency per query and a
    second model loaded into memory on first use."""
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher.search(query, k=k, rerank=rerank)


def main():
    demo_queries = [
        ("combien de semaines de vacances après 3 ans", "fr"),
        ("how much notice before firing someone", "en"),
        ("am I owed extra pay for working over 40 hours", "en"),
        ("est-ce que mon patron peut garder mes pourboires", "fr"),
        ("do I get paid for a doctor's appointment", "en"),
    ]
    for query, _ in demo_queries:
        print("=" * 70)
        print(f"QUERY: {query}")
        results = search(query, k=3)
        for i, r in enumerate(results, 1):
            print(
                f"  #{i} article {r['article']} [{r['lang']}] "
                f"rrf={r['score']:.5f} bm25={r['bm25_score']} dense={r['dense_score']}"
            )
            print(f"     {r['url']}")
            print(f"     {r['text'][:200]}")
        print()


if __name__ == "__main__":
    main()
