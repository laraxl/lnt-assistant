"""Cross-encoder reranking over the top-N hybrid search results.

Takes the (article-deduplicated) hybrid results from search.py and rescoresb
them with a multilingual cross-encoder, which reads the query and each
candidate's full text together (unlike the bi-encoder dense channel, which
scores them independently) — a stronger but slower relevance signal, meant
to fix ordering within a shortlist rather than to search the whole corpus.

Usage:
    from search import Searcher
    from rerank import Reranker
    s = Searcher()
    r = Reranker()
    candidates = s.search("...", k=20)
    reranked = r.rerank("...", candidates, top_k=5)
"""
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list, top_k: int = None) -> list:
        """candidates: list of dicts with a 'text' field (as returned by
        Searcher.search/fuse). Returns the same dicts, resorted by
        cross-encoder score descending, with a 'rerank_score' field added."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k] if top_k else reranked
