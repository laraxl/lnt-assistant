"""Embed every chunk in data/chunks.jsonl with a multilingual sentence
embedding model.

Outputs:
    data/embeddings.npy      float32 array, shape (n_chunks, dim), L2-normalized
    data/chunk_ids.json      list of chunk_id, same order as the rows above
    data/embedding_meta.json model name, dimension, chunk count

Usage:
    .venv/bin/python src/embed.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CHUNK_IDS_PATH = DATA_DIR / "chunk_ids.json"
META_PATH = DATA_DIR / "embedding_meta.json"

# paraphrase-multilingual-mpnet-base-v2 was the starting point, but it's a
# *symmetric* paraphrase/STS model: it scores two same-length, same-register
# texts as similar, which is the wrong objective for retrieval, where a short
# colloquial question has to match a long formal statute passage. Testing
# confirmed the failure mode directly: e.g. for "am I owed extra pay for
# working over 40 hours" it ranked article 55 (the overtime-premium article)
# outside its own top-50 nearest neighbours, and for a French question about
# an employer keeping tips it ranked article 50 (the tips article) 155th out
# of 615 chunks. intfloat/multilingual-e5-base is contrastively trained
# specifically for asymmetric query/passage retrieval across languages (the
# same family the "E5" line is built for), and requires "query: "/"passage: "
# prefixes to activate that asymmetric behaviour. With it, both failing
# cases moved to the top of the ranking on the dense signal alone.
MODEL_NAME = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def main():
    chunks = [json.loads(l) for l in CHUNKS_PATH.open(encoding="utf-8")]
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [PASSAGE_PREFIX + c["text"] for c in chunks]

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_embedding_dimension()

    print(f"Embedding {len(texts)} chunks (dim={dim})...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
    ).astype("float32")

    np.save(EMBEDDINGS_PATH, embeddings)
    CHUNK_IDS_PATH.write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    META_PATH.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "dimension": int(dim),
                "n_chunks": len(chunk_ids),
                "normalized": True,
                "metric": "cosine",
                "query_prefix": QUERY_PREFIX,
                "passage_prefix": PASSAGE_PREFIX,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved {embeddings.shape} embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved {len(chunk_ids)} chunk ids to {CHUNK_IDS_PATH}")
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
