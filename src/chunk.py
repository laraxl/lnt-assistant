"""Split data/articles.jsonl into retrieval-sized chunks (data/chunks.jsonl).

Most articles are short enough to stay as a single chunk. Longer articles are
split on natural boundaries only — alinéa (paragraph) breaks and enumerated
sub-items (1°/2°, (1)/(2), a)/b), (a)/(b), i./ii., (i)/(ii)) — so a chunk
never ends mid-sentence.

Usage:
    .venv/bin/python src/chunk.py
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTICLES_PATH = DATA_DIR / "articles.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

MAX_CHUNK_CHARS = 1500

# LegisQuebec's HTML has each alinéa (and title/heading) as a separate block
# element. Our scraper's get_text() has no separator, so two adjacent blocks
# end up glued together with *zero* whitespace between them, while a normal
# sentence boundary inside one alinéa keeps its original space
# ("...horaire.Malgré..." vs "...payées. Cependant..."). That gap is exactly
# the alinéa boundary, and it's a reliable, structural signal rather than a
# guess based on sentence length.
ALINEA_BOUNDARY_RE = re.compile(r"(?<=[.;])(?=[A-ZÀ-ÖØ-Þ])")

# Enumerated sub-items are glued onto the punctuation that introduces them
# the same way, e.g. "par:1°", ";2°", "personnesa)", "who(a)", "duquel:i.".
# Each alternative is a zero-width lookahead anchored on the preceding ':'
# or ';' (or, for lettered items, any word character) so real prose that
# happens to contain "50%" or "(2022, c. 22)" doesn't get mistaken for a
# list marker.
_ROMAN = r"(?:i{1,3}|iv|vi{0,3})"
ITEM_BOUNDARY_RE = re.compile(
    r"(?<=[:;])(?=\d{1,2}(?:\.\d+)*°)"  # French numbered: 1°, 2°, 3.1°
    r"|(?<=[:;])(?=\(\d{1,2}\))"  # English numbered: (1), (2)
    r"|(?<=[:;\w])(?=[a-z]\)[\s ])"  # French lettered: a), b)
    r"|(?<=[:;])(?=\([a-z]\))"  # English lettered: (a), (b)
    r"|(?<=[:;])(?=" + _ROMAN + r"\.[\s ])"  # French roman: i., ii.
    r"|(?<=[:;])(?=\(" + _ROMAN + r"\))"  # English roman: (i), (ii)
)


def boundary_positions(text: str) -> list:
    """Sorted, deduplicated character offsets where a chunk split is safe."""
    positions = {0, len(text)}
    for rx in (ALINEA_BOUNDARY_RE, ITEM_BOUNDARY_RE):
        for m in rx.finditer(text):
            positions.add(m.start())
    return sorted(positions)


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """Split text into chunks of at most ~max_chars, only at boundaries
    returned by boundary_positions(); never mid-sentence."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    bounds = boundary_positions(text)
    chunks = []
    start = 0
    while start < len(text):
        limit = start + max_chars
        # Widest boundary that still fits the budget.
        candidates = [b for b in bounds if start < b <= limit]
        if candidates:
            end = candidates[-1]
        else:
            # The next segment alone exceeds max_chars (e.g. one very long
            # alinéa): take the next boundary anyway rather than cut
            # mid-sentence.
            after = [b for b in bounds if b > start]
            end = after[0] if after else len(text)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end
    return chunks


def chunk_article(article: str, lang: str, text: str, url: str) -> list:
    pieces = split_text(text)
    n = len(pieces)
    return [
        {
            "chunk_id": f"{article}:{lang}:{i}",
            "article": article,
            "lang": lang,
            "chunk_index": i,
            "n_chunks": n,
            "text": piece,
            "url": url,
        }
        for i, piece in enumerate(pieces)
    ]


def main():
    records = []
    with ARTICLES_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)

    chunks = []
    for rec in records:
        article = rec["article"]
        if rec["text_fr"] is not None:
            chunks.extend(chunk_article(article, "fr", rec["text_fr"], rec["url_fr"]))
        if rec["text_en"] is not None:
            chunks.extend(chunk_article(article, "en", rec["text_en"], rec["url_en"]))

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lengths = [len(c["text"]) for c in chunks]
    multi = [rec["article"] for rec in records if rec["text_fr"] and len(rec["text_fr"]) > MAX_CHUNK_CHARS]

    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")
    print(f"  from {len(records)} articles ({len(records) * 2} article/language pairs)")
    print(f"  articles split into >1 chunk (by fr length): {len(multi)}")
    print()
    print("Chunk length distribution (chars):")
    print(f"  min={min(lengths)}  max={max(lengths)}  "
          f"mean={statistics.mean(lengths):.1f}  median={statistics.median(lengths):.1f}")
    lengths_sorted = sorted(lengths)
    for p in (50, 75, 90, 95, 99):
        idx = min(len(lengths_sorted) - 1, int(len(lengths_sorted) * p / 100))
        print(f"  p{p}={lengths_sorted[idx]}")


if __name__ == "__main__":
    main()
