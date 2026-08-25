"""Scrape the Loi sur les normes du travail (chapter N-1.1) from LegisQuebec
into a clean, article-level JSONL dataset (data/articles.jsonl).

Usage:
    .venv/bin/python src/scrape_lnt.py
"""
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CHAPTER = "N-1.1"
BASE_URL = f"https://www.legisquebec.gouv.qc.ca/fr/document/lc/{CHAPTER}"
URLS = {
    "fr": f"{BASE_URL}?langCont=fr",
    "en": f"{BASE_URL}?langCont=en",
}
CACHE_FILES = {
    "fr": DATA_DIR / f"{CHAPTER}_fr.html",
    "en": DATA_DIR / f"{CHAPTER}_en.html",
}

# A descriptive but browser-shaped User-Agent: LegisQuebec's CloudFront front
# end returns 403 for User-Agents that don't start with "Mozilla/5.0", even
# ones that are otherwise honest about being a bot.
USER_AGENT = "Mozilla/5.0 (compatible; LNT-QA-Bot/0.1; +mailto:chrisattayek@gmail.com)"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-CA,fr;q=0.9,en;q=0.8",
}

# Anchor marking the start of each article, e.g. <a class="linkOtherLang"
# href="#se:79_1" data-code="se:79_1">. Underscores map to dots in the real
# article number: se:79_1 -> "79.1", se:39_0_0_1 -> "39.0.0.1".
ARTICLE_HREF_RE = re.compile(r"^#se:(\d+(?:_\d+)*)$")

# A private-use character can't appear in the page's real text, so it's a
# safe delimiter to split on after replacing anchors with markers.
MARKER_CHAR = ""
MARKER_RE = re.compile(MARKER_CHAR + r"(\d+(?:_\d+)*)" + MARKER_CHAR)

# One amendment-history entry, e.g. "1979, c. 45, a. 69" or "2022, c. 22, ss.
# 152 and 179". French uses "a." (article); English uses "s."/"ss." (section
# /sections). The reference part is "everything up to the next ; or .".
_STD_YEAR_ENTRY = r"\d{4},\s*c\.\s*\d+(?:,\s*(?:a\.|ss?\.)\s*[^;.]+)?"
# A rare foreign-jurisdiction citation, e.g. "R.-U., 1982, c. 11, ann. B,
# ptie I, a. 33" (French) / "U. K., 1982, c. 11, Sch. B, Part I, s. 33"
# (English). Its reference part itself contains periods (e.g. "ann."), so
# unlike the standard entry it can't stop at the first ".": it's only ever
# observed as the last entry in a history line, so it's safe to let it run
# up to the true terminal period.
_JURISDICTION_ENTRY = r"[A-Z]\.\s*-?\s*[A-Z]\.,\s*\d{4},\s*c\.\s*\d+,\s*[^;]+"
# A handful of entries are administrative notes instead of amendments, e.g.
# "N.I. 2016-01-01 (NCPC)" (French) / "I.N. 2016-01-01 (NCCP)" (English).
_NOTE_ENTRY = r"[A-Z]\.[A-Z]\.\s*[\d-]+\s*\([^)]*\)"
_ENTRY = r"(?:" + _JURISDICTION_ENTRY + r"|" + _STD_YEAR_ENTRY + r"|" + _NOTE_ENTRY + r")"
HISTORY_RE = re.compile(r"(" + _ENTRY + r"(?:\s*;\s*" + _ENTRY + r")*)\s*\.")

# LegisQuebec's consolidation markup sometimes concatenates a citation with
# itself, e.g. "2022, c. 222022, c. 22, a. 179" instead of "2022, c. 22, a.
# 179". Collapse an immediately-repeated "YYYY, c. NN" prefix before parsing.
DOUBLED_CITATION_RE = re.compile(r"(\d{4},\s*c\.\s*\d+)\1")

# Articles whose body is nothing but one of these placeholders carry no
# substantive content (repealed, replaced by another provision, or omitted
# from the consolidated text) and are dropped entirely.
PLACEHOLDER_BODIES = {
    "fr": {"(Abrogé)", "(Remplacé)", "(Omis)"},
    "en": {"(Repealed)", "(Replaced)", "(Omitted)"},
}

WHITESPACE_CHARS = ("\xa0", " ", "​", "‎", "‏")


def fetch(lang: str) -> str:
    """Download (or load a cached copy of) the Act in the given language."""
    cache_path = CACHE_FILES[lang]
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    resp = requests.get(URLS[lang], headers=HEADERS, timeout=30)
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def article_number(raw: str) -> str:
    """se:79_1 -> "79.1"; se:39_0_0_1 -> "39.0.0.1"."""
    return raw.replace("_", ".")


def normalize_whitespace(text: str) -> str:
    for ch in WHITESPACE_CHARS:
        text = text.replace(ch, " " if ch not in ("​", "‎", "‏") else "")
    return re.sub(r"\s+", " ", text).strip()


def extract_marked_text(html: str) -> str:
    """Strip boilerplate/noise elements, replace each article-start anchor
    with a unique marker, and return the full page text."""
    soup = BeautifulSoup(html, "html.parser")

    # Scope to the document body itself: the full page also has nav, footer,
    # and a floating "copy selection" toolbar outside this container, which
    # would otherwise be swept into the last article's chunk.
    content = soup.find(id="mainContent-document")
    if content is None:
        raise RuntimeError("Could not find #mainContent-document container")
    soup = content

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # LegisQuebec's consolidation markup keeps superseded text around in
    # elements with class="Hidden" (toggled off via CSS). get_text() doesn't
    # know about CSS, so it must be dropped explicitly, or article/history
    # text comes out doubled and jumbled.
    for tag in soup.select(".Hidden"):
        tag.decompose()

    # Trailing schedules (Annexe I / Schedule I, Annexes abrogatives /
    # Repealing Schedules) live outside the se:NNN section elements and have
    # no article anchor of their own; without this the last article's chunk
    # (which runs to the end of the document) would swallow them.
    for tag in soup.select(".Schedule-abrogative"):
        tag.decompose()

    marker_count = 0
    for a in soup.find_all("a", href=ARTICLE_HREF_RE):
        m = ARTICLE_HREF_RE.match(a["href"])
        a.replace_with(f"{MARKER_CHAR}{m.group(1)}{MARKER_CHAR}")
        marker_count += 1

    if marker_count == 0:
        raise RuntimeError("No article anchors (href=#se:...) found on the page")

    text = soup.get_text()
    text = DOUBLED_CITATION_RE.sub(r"\1", text)
    return text


def parse_articles(html: str, lang: str) -> dict:
    """Return {article_number: {"text": ..., "history": ...}} for a page,
    skipping repealed/placeholder articles."""
    text = extract_marked_text(html)
    parts = MARKER_RE.split(text)
    # parts = [preamble, num1, chunk1, num2, chunk2, ...]

    placeholders = PLACEHOLDER_BODIES[lang]
    articles = {}
    for i in range(1, len(parts), 2):
        raw_num = parts[i]
        chunk = parts[i + 1]
        num = article_number(raw_num)

        m = HISTORY_RE.search(chunk)
        if not m:
            raise RuntimeError(
                f"Could not find amendment history for article {num} ({lang}); "
                f"chunk starts: {chunk[:200]!r}"
            )
        body = normalize_whitespace(chunk[: m.start()])
        history = normalize_whitespace(m.group(1)) + "."

        if body.rstrip(".").strip() in placeholders:
            continue

        if num in articles:
            raise RuntimeError(f"Duplicate article number {num} ({lang})")

        articles[num] = {"text": body, "history": history}

    return articles


def article_sort_key(num: str):
    return [int(part) for part in num.split(".")]


def build_dataset(fr_articles: dict, en_articles: dict) -> list:
    fr_only = sorted(set(fr_articles) - set(en_articles), key=article_sort_key)
    en_only = sorted(set(en_articles) - set(fr_articles), key=article_sort_key)
    if fr_only:
        print(f"Articles only in French ({len(fr_only)}): {fr_only}")
    if en_only:
        print(f"Articles only in English ({len(en_only)}): {en_only}")

    all_numbers = sorted(set(fr_articles) | set(en_articles), key=article_sort_key)

    records = []
    for num in all_numbers:
        fr = fr_articles.get(num)
        en = en_articles.get(num)
        history = (fr or en)["history"]
        anchor = f"#se:{num.replace('.', '_')}"
        records.append(
            {
                "id": f"{CHAPTER}:{num}",
                "article": num,
                "text_fr": fr["text"] if fr else None,
                "text_en": en["text"] if en else None,
                "history": history,
                "url_fr": URLS["fr"] + anchor,
                "url_en": URLS["en"] + anchor,
            }
        )
    return records


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching French text...")
    fr_html = fetch("fr")
    print("Fetching English text...")
    en_html = fetch("en")

    print("Parsing French articles...")
    fr_articles = parse_articles(fr_html, "fr")
    print(f"  {len(fr_articles)} articles (non-repealed)")

    print("Parsing English articles...")
    en_articles = parse_articles(en_html, "en")
    print(f"  {len(en_articles)} articles (non-repealed)")

    records = build_dataset(fr_articles, en_articles)

    out_path = DATA_DIR / "articles.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    missing_en = sum(1 for r in records if r["text_en"] is None)
    missing_fr = sum(1 for r in records if r["text_fr"] is None)
    print(f"\nWrote {len(records)} articles to {out_path}")
    print(f"Missing English translation: {missing_en}")
    print(f"Missing French translation: {missing_fr}")


if __name__ == "__main__":
    main()
