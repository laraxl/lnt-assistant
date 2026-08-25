# LNT Q&A

A bilingual (French/English) question-answering system over Quebec's *Loi sur
les normes du travail* (LNT, chapter N-1.1) — the province's employment
standards act: minimum wage, hours of work, overtime, vacation, statutory
holidays, leaves of absence, termination notice, and related complaint
procedures.

Given a plain-language question in French or English, the system finds the
right article(s) of the Act, or tells you — and, since October 2025, tries to
tell you *why* — when the Act isn't the right place to look at all
(federally regulated work, the construction industry, or the question isn't
about employment law).

**Who this is for:** a starting point for someone building an LNT-grounded
assistant — a retrieval layer with its accuracy actually measured, not
assumed. It is not itself a finished consumer product, and it is not legal
advice (see [Limitations](#limitations)).

## What it does

```
question ──▶ jurisdiction gate ──▶ hybrid search (BM25 + dense) ──▶ cross-encoder rerank ──▶ top-k articles
             (LLM classifier)       (keyword ∪ semantic, RRF-fused)   (bge-reranker-v2-m3)
```

- **Scraper** (`src/scrape_lnt.py`) — downloads the Act from LégisQuébec in
  both languages and parses it into 286 clean, article-level records.
- **Chunker** (`src/chunk.py`) — splits the ~12 articles long enough to need
  it into ≤1500-character pieces, only at real sentence/paragraph
  boundaries.
- **Embedder** (`src/embed.py`) — embeds all 615 chunks with a multilingual
  retrieval model.
- **Hybrid search** (`src/search.py`) — SQLite FTS5 keyword search + dense
  cosine similarity, combined by reciprocal rank fusion (RRF).
- **Reranker** (`src/rerank.py`) — cross-encoder reordering of the top 20
  hybrid results, opt-in via `rerank=True`.
- **Jurisdiction gate** (`src/gate.py`) — an LLM call that runs *before*
  retrieval and decides whether the LNT applies at all.
- **Evaluation harness** (`src/evaluate*.py`) — a 150-question, hand-audited
  eval set (`data/eval.jsonl`) and scripts that measure every one of the
  above, honestly, instead of on 4 hand-picked examples.

## 60-second quickstart

```bash
git clone <this-repo>
cd lnt
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Add an Anthropic API key (only needed for the jurisdiction gate — search and
rerank work without it):

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Build the dataset and indexes (takes a few minutes; downloads the Act and
two ML models on first run):

```bash
./.venv/bin/python3 src/scrape_lnt.py    # -> data/articles.jsonl (286 articles)
./.venv/bin/python3 src/chunk.py         # -> data/chunks.jsonl (615 chunks)
./.venv/bin/python3 src/embed.py         # -> data/embeddings.npy
```

Ask a question:

```python
import sys; sys.path.insert(0, "src")
from search import search

for r in search("combien de semaines de vacances après 3 ans", k=3, rerank=True):
    print(r["article"], r["url"], r["text"][:120])
```

Or with the jurisdiction gate in front of it:

```python
from gate import classify
route = classify("I fly for an airline, does this law apply to me?")[0]
print(route.route, route.reason, route.redirect)
# REDIRECT  ...federally regulated...  Canada Labour Code / Federal Labour Program
```

Reproduce every number in this README:

```bash
./.venv/bin/python3 src/evaluate.py          # RRF weight sweep, base recall/MRR
./.venv/bin/python3 src/evaluate_rerank.py   # cross-encoder before/after
./.venv/bin/python3 src/evaluate_gate.py     # gate confusion matrix + harm prevented (costs ~$0.40 in API calls)
./.venv/bin/python3 src/evaluate_e2e.py      # full pipeline recall@3
```

## Architecture, and why

**Scraping over an official API, because there isn't one.** LégisQuébec has
no structured export; the HTML has undocumented quirks that would silently
corrupt the text if handled naively — consolidation markup duplicates
superseded text in `class="Hidden"` spans (fixed by stripping them before
extracting text), and the trailing Schedule/Annexe content has no article
anchor of its own, so it had to be explicitly excluded or it would bleed into
the last real article. Every article's history citation is parsed with a
regex tuned against the actual variety of citation formats in the text
(standard `1979, c. 45, a. 69`, administrative notes like `N.I. 2016-01-01
(NCPC)`, and one foreign-jurisdiction citation to the UK Canada Act).

**Article-aligned chunking on structural signals, not fixed windows.** A
naive fixed-character split would cut mid-sentence. Because the scraper's
text has no paragraph markers, alinéa boundaries were recovered from a
genuine structural artifact: adjacent HTML blocks get concatenated with
*zero* whitespace (`"...horaire.Malgré..."`), while a real sentence boundary
inside one paragraph keeps its space (`"...payées. Cependant..."`). Numbered
sub-items (`1°`, `(1)`, `a)`, `(a)`, `i.`, `(i)`) are detected the same way,
glued onto the punctuation that introduces them. Only 12 of 286 articles
exceed 1500 characters and needed splitting at all.

**A retrieval-tuned embedding model, chosen by measurement, not the
starting recommendation.** The build started with
`paraphrase-multilingual-mpnet-base-v2` as instructed. It's a *symmetric*
paraphrase/STS model — the wrong objective for retrieval, where a short
colloquial question has to match a long formal statute passage. Direct
testing exposed the failure: for "am I owed extra pay for working over 40
hours," it ranked the actual answer (article 55, the overtime-premium
article) **outside its own top 50 nearest neighbours**; for a French
question about an employer keeping tips, it ranked article 50 (the tips
article) **155th out of 615 chunks**. Switching to
`intfloat/multilingual-e5-base` — contrastively trained for asymmetric
cross-lingual query/passage retrieval, with `"query: "`/`"passage: "`
prefixes — moved both to the top of the ranking on the dense signal alone.

**Hybrid search, but weighted by measurement, not intuition.** Plain
1:1 RRF between BM25 and dense search missed the correct article for
several held-out queries: this corpus is full of repetitive legal
boilerplate ("employee," "employer," "hours," "wages"), so BM25 regularly
rewards a lexically-adjacent-but-wrong article over the semantically
correct one that shares no literal vocabulary with the question. The
dense/keyword balance was tuned by sweeping `{1x, 3x, 10x, 30x, 60x,
dense-only}` over the full 150-question eval set — see
[Results](#results) for why 10x won and why that matters.

**A cross-encoder reranker, kept because it earned it, not by default.**
Bi-encoder dense search scores the query and each passage independently;
a cross-encoder reads them together, which is slower but substantially
more accurate. It's opt-in (`rerank=True`) because the accuracy gain comes
with real, disclosed latency — see [Results](#results).

**A jurisdiction gate that routes three ways, not two.** The first version
was a binary allow/block classifier and it actively hurt end-to-end
accuracy (see [Results](#results)) — a hedge or a topical misjudgment cost
the user their answer outright. The second version replaces block with two
distinct outcomes: `REDIRECT` (wrong jurisdiction entirely — federally
regulated work, or construction under R-20 — no article will help, so
retrieval doesn't run) and `EXCLUDED` (the Act's own article 3 excludes this
worker, but article 3 explaining that exclusion *is* the answer, so
retrieval still runs). Only `REDIRECT` suppresses retrieval now; a hedge
(`confident=false`) defaults to `ALLOW` rather than blocking.

## Results

All numbers below are reproducible from `data/*.json` via the
`evaluate*.py` scripts, over `data/eval.jsonl`: 150 hand-written questions
(88 French / 62 English, 127 in-scope / 23 out-of-scope), phrased the way a
worker with no legal training would actually ask — deliberately avoiding
the Act's own vocabulary, so the eval doesn't flatter the system by echoing
answer text back at it.

### RRF weight sweep — why 10x, not the hand-tuned 60x

The dense/keyword weight was first hand-tuned to 60x against 4 demo
queries. That was overfitting: swept honestly over all 150 questions, 60x
was not the best setting on any metric.

| weight | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| 1x | 0.496 | 0.669 | 0.811 | 0.843 | 0.614 |
| 3x | 0.535 | 0.740 | 0.835 | 0.882 | **0.661** |
| **10x (winner, recall@3)** | 0.528 | **0.756** | 0.835 | **0.906** | 0.660 |
| 30x | 0.520 | 0.740 | 0.819 | 0.882 | 0.652 |
| 60x (the original hand-tuned pick) | 0.496 | 0.732 | 0.811 | 0.882 | 0.635 |
| dense-only | 0.488 | 0.732 | 0.795 | 0.882 | 0.627 |

10x wins on recall@3 and recall@10; 3x is marginally better on recall@1 and
MRR. Every weighted setting beats dense-only on recall@3, so hybrid search
still earns its place in the design — just not at the weight it was first
shipped with. `DENSE_WEIGHT = 10.0` in `src/search.py`.

### Embedding model switch — the ranks that exposed it

| query | article | rank, `paraphrase-multilingual-mpnet-base-v2` | rank, `intfloat/multilingual-e5-base` |
|---|---|---|---|
| "am I owed extra pay for working over 40 hours" | 55 (overtime premium) | outside top 50 (of 615) | top of ranking |
| "est-ce que mon patron peut garder mes pourboires" | 50 (tips) | 155th of 615 | top of ranking |

The starting model is a *symmetric* similarity model; the wrong objective
for a short question against a long formal passage. See
[Architecture](#architecture-and-why) for why.

### Cross-encoder reranking — before/after, with the latency cost stated

Reranking the top-20 hybrid results with `BAAI/bge-reranker-v2-m3`, over
the same 127 in-scope questions:

| metric | before (hybrid only) | after (+ rerank) | delta |
|---|---|---|---|
| recall@1 | 0.528 | 0.693 | **+0.165** |
| recall@3 | 0.756 | 0.874 | **+0.118** |
| recall@5 | 0.835 | 0.906 | +0.071 |
| recall@10 | 0.906 | 0.937 | +0.031 |
| MRR | 0.658 | 0.788 | **+0.130** |

**Added latency: ~1.1s per query** (mean 1117ms, p50 1147ms, p95 1322ms,
CPU — no GPU in the build environment). That's real, user-facing cost, not
free — kept opt-in for that reason. The accuracy gain is large and
consistent across every k, not a rounding error, so it clears the bar: kept.

### Jurisdiction gate — two iterations

**v1, binary allow/block:** correctly blocked 21/23 out-of-scope questions
(91.3%) and wrongly blocked 4/127 in-scope questions (3.1%). Measured
end-to-end:

| | recall@3 |
|---|---|
| hybrid + rerank, no gate | 0.874 |
| gate (binary) → hybrid → rerank | 0.843 |
| delta | **−0.031** |

**Rejected.** A 3.1% false-block rate outweighed the 91.3% correct-block
rate in the metric that matters, because a block is an unconditional miss —
even for the 2 of 4 false blocks that were outright reasoning bugs (the
model cited unrelated statutes for a temp-agency permit question and a
group-insurance-continuity question that articles 92.5 and 79.3 answer
directly — it was judging by *topic* familiarity instead of *worker*
coverage, which the rewritten prompt fixed generally, not by special-casing
those two questions).

**v2, three-way routing (`REDIRECT` / `EXCLUDED` / `ALLOW`):**

| | recall@3 |
|---|---|
| hybrid + rerank, no gate | 0.874 |
| gate (three-way) → hybrid → rerank | **0.874** |
| delta | **0.000** |

| | value |
|---|---|
| Out-of-scope questions correctly redirected/excluded (not a confident wrong citation) | **23/23 (100%)** |
| In-scope questions blocked from retrieval | **0/127** |

**Kept.** Removing retrieval-blocking from every route except `REDIRECT`
(genuinely wrong jurisdiction) removed the tradeoff rather than tuning it:
the gate now costs nothing on end-to-end recall while catching every
out-of-scope case. *Recall@3 does not measure the harm the gate exists to
prevent* — a correctly redirected question never reaches retrieval, so it
can't appear in that number either way; the 23/23 figure is what actually
measures whether the gate is doing its job, and the two numbers have to be
read together, not netted into one. Cost: ~$0.0027/query with prompt
caching (~$0.0133 cold). This is stated explicitly, not just here, inside
`data/gate_results.json` and `data/e2e_results.json`.

### Label audit — checking the eval set against itself

The eval set's questions and gold labels were both written by the same
process that built the retrieval system, so a self-serving error would be
invisible by construction. All 50 questions the system missed at k=3 were
reviewed adversarially, article text against article text, and bucketed:

| bucket | count |
|---|---|
| TRUE MISS (label correct, retrieval failed) | 26 |
| BAD LABEL (a better answer existed than the original label) | 3 |
| AMBIGUOUS (genuinely multiple correct articles) | 4 |
| out-of-scope, correctly labeled, system just overconfident (doesn't fit the above) | 17 |

**7 of 150 labels changed** (4.7%) — 3 corrected outright (one moved from
"article 3" to "article 1," the actual controlling provision for a
self-supplied-equipment courier's employment status; two moved out of the
out-of-scope bucket entirely, since the honest answer to an
occasional-babysitter or family-business exclusion question is article 3
explaining the exclusion, not silence), 4 given a second acceptable article.
Re-run after correction: **recall@3 0.752 → 0.756, winning weight
unchanged at 10x** — the correction moved the eval set, not the conclusion.

### A bug worth naming: the judging regex, not the gate

The first pass at scoring the gate's out-of-scope handling checked whether
the `redirect` field contained English phrases like "Canada Labour Code."
The gate, correctly answering in French for French questions, wrote "Code
canadien du travail" and "Programme du travail" — text the regex didn't
recognize. Four genuinely correct redirects were scored as failures. Fixed
the regex (no API calls needed to re-judge already-saved responses) and the
true figure was 23/23, not 19/23. Recorded here because a report is only as
trustworthy as its willingness to say when its own scoring script was wrong.

## Limitations

- **~13% of in-scope questions are still missed at k=3** (recall@3 = 0.874
  end-to-end). The remaining failures — full list in `data/eval_results.json`
  under `failures_at_k3` — are mostly genuine paraphrase gaps: e.g. the
  marriage-leave question ranks the correct article (81) around 53rd, and
  the union-contract-override question ranks its answer (93/94) 20th–37th.
  These are real limits of dense retrieval on colloquial phrasing, not
  noise to explain away.
- **The eval set was written by the same process that built the system it
  measures, and was only spot-audited, not independently re-derived.** The
  label audit reviewed the 50 questions the system got wrong, adversarially,
  and corrected 7 — but it did not re-derive gold labels for the 100
  questions the system got *right*, and it can't rule out a shared blind
  spot between how the questions were phrased and how the retriever thinks,
  since one person (assisted by one model) built both.
- **This is one statute.** Every design choice here — the exclusion logic
  in the gate, the citation regex, the chunking heuristics — is specific to
  the LNT's actual text and HTML. None of it is validated against, or
  necessarily transferable to, any other law.
- **The Act's text is not the whole of Quebec employment law.** Tribunal
  decisions (Tribunal administratif du travail), CNESST interpretation
  policy, and case law can change how a provision actually applies in
  practice — sometimes substantially — in ways the statute's bare text
  doesn't show and this system has no way to know about. An article that's
  the right citation is not automatically the complete or final answer.
- **This is legal information, not legal advice**, and isn't a substitute
  for consulting a lawyer or the CNESST for an actual employment situation.
  Nothing in this system's output should be treated as a determination of
  legal rights.
