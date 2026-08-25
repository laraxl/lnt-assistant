# LNT Q&A

A bilingual (French/English) question-answering system over Quebec's *Loi sur
les normes du travail* (LNT, chapter N-1.1) — the province's employment
standards act: minimum wage, hours of work, overtime, vacation, statutory
holidays, leaves of absence, termination notice, and related complaint
procedures.

Given a plain-language question in French or English, the system finds the
right article(s) of the Act, or tells you why the Act isn't the right place
to look at all (federally regulated work, the construction industry, or the
question isn't about employment law).

**Who this is for:** a starting point for someone building an LNT-grounded
assistant — a retrieval layer with its accuracy actually measured, not
assumed. It is not itself a finished consumer product, and it is not legal
advice (see [Limitations](#limitations)).

## What it does

```
question ──▶ jurisdiction gate ──▶ hybrid search (BM25 + dense) ──▶ cross-encoder rerank ──▶ answer generation ──▶ answer + citations
             (LLM classifier)       (keyword ∪ semantic, RRF-fused)   (bge-reranker-v2-m3)     (LLM, grounded only
                                                                                                 in the top-3 articles)
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
- **Answer generation** (`src/answer.py`) — an LLM call grounded *only* in
  the top-3 retrieved article texts: cites article numbers inline, and is
  explicitly instructed to say when the retrieved articles don't answer the
  question rather than inventing a rule.
- **Web app** (`src/api.py`, `static/index.html`) — a FastAPI backend
  loading both ML models once at startup, and a single-file frontend (FR
  default, EN toggle, no build step) that calls it.
- **Evaluation harness** (`src/evaluate*.py`) — a 150-question, hand-audited
  eval set (`data/eval.jsonl`) and scripts that measure every one of the
  above, honestly, instead of on 4 hand-picked examples.

## 60-second quickstart

```bash
git clone https://github.com/laraxl/lnt-assistant.git
cd lnt-assistant
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

**Answer generation grounded only in what was retrieved, nothing else.**
`src/answer.py` is where hallucination risk actually lives — an LLM call
that gets *only* the top-3 retrieved article texts and is instructed to
state a rule only if it's written there, cite the article number inline for
every claim, and say plainly when the provided text doesn't answer the
question rather than stretching a related article into one it doesn't
support. The disclaimer and answer language follow the UI's language
toggle, not whatever language the question happens to be typed in — a user
can toggle to EN and ask a French question and get an English answer,
English citations, and an English disclaimer, because the toggle is what's
sent to the backend as `lang`, not inferred from the question text.

**A relevance floor on displayed citations, not on retrieval.** Hybrid
search + rerank always return the top-3 candidates — always 3, even when
only 1 or 2 are actually relevant — so a weak third result could show up as
a citation card next to two strong ones and make the whole response look
unreliable. `src/api.py` still feeds all 3 articles to answer generation
(more grounding context doesn't hurt, and the prompt already handles
irrelevant context by saying so), but only *displays* a rank-2/3 citation
if its cross-encoder rerank score clears `CITATION_RERANK_FLOOR = 0.05`;
rank 1 is always shown, since it's virtually always the retriever's actual
best guess. That threshold came from checking, across the 127 in-scope
eval questions, where correct vs. incorrect rank-2/3 citations' rerank
scores actually fall — see [Results](#results) for the numbers, because the
honest finding is that correct and incorrect scores overlap substantially
and no threshold cleanly separates them; 0.05 was chosen as the best
available tradeoff, not a clean cut.

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

### Citation relevance floor — an honest tradeoff, not a clean cutoff

Reported failure: "how much vacation after 2 years" returned article 81.4.1
(maternity leave after a late delivery — completely unrelated) as citation
#2, next to two genuinely relevant articles. The written answer was
correct; the citation panel looked broken.

Checked, across all 127 in-scope eval questions, the cross-encoder rerank
scores of citations that landed in the top-3: correct ones ranged from
0.0007 to 0.999 (median 0.33), and incorrect ones that still made the
top-3 ranged from 0.0001 to 0.99 (median 0.05). **The two distributions
overlap almost completely — no single threshold cleanly separates correct
from incorrect.** Restricting the threshold to rank 2/3 only (rank 1 is
always shown — it's virtually always the retriever's genuine best guess and
essentially never needs filtering) makes the tradeoff far more favorable:

| threshold | correct citations lost (of 127 questions) | incorrect rank-2/3 citations hidden (of 268) |
|---|---|---|
| 0.01 | 5 | 50 |
| 0.02 | 5 | 72 |
| 0.03 | 5 | 89 |
| 0.04 | 5 | 107 |
| **0.05 (chosen)** | **6** | **115** |
| 0.06 | 6 | 124 |
| 0.10 | 11 | 138 |

0.05 was picked because it's the smallest threshold that clears the
motivating case (article 81.4.1 scored 0.047) without jumping the correct-citations-lost
count — 0.04 and 0.05 cost the same 5-6 questions, so there's no reason to
stop short of catching that case. **6 of 127 in-scope questions (4.7%) lose
a citation that was actually correct** at this threshold — specifically,
the ones where the correct article happened to land at rank 2 or 3 rather
than rank 1. That cost is disclosed, not hidden: `CITATION_RERANK_FLOOR` in
`src/api.py` documents this analysis inline.

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

## Reproducing these results

`data/eval.jsonl` and every `data/*_results.json` file are tracked in this
repo specifically so the numbers above can be checked against real output,
not just read on faith — `data/articles.jsonl` too (it's under 1MB). The
rest of `data/` (raw scraped HTML, `embeddings.npy`, `index.db`,
`chunks.jsonl`) is regenerable and gitignored to keep the repo small.

Run these in order from a fresh clone:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

| # | command | needs | first run | cached re-run |
|---|---|---|---|---|
| 1 | `./.venv/bin/python3 src/scrape_lnt.py` | network (LégisQuébec) | ~10–30s, occasionally longer if the site rate-limits | instant (reads cached `data/*.html`) |
| 2 | `./.venv/bin/python3 src/chunk.py` | nothing | <5s | <5s |
| 3 | `./.venv/bin/python3 src/embed.py` | network (downloads `multilingual-e5-base`, ~1.1GB, from Hugging Face) | a few minutes, mostly the download | ~10s |
| 4 | `./.venv/bin/python3 src/evaluate.py` | nothing (reuses the embedding model, no API) | ~1–2 min | ~1–2 min |
| 5 | `./.venv/bin/python3 src/evaluate_rerank.py` | network (downloads `bge-reranker-v2-m3`, ~1.1GB, first run only) | several minutes (download) + ~3 min (150 questions × ~1.1s/query on CPU) | ~3 min |
| 6 | `./.venv/bin/python3 src/evaluate_gate.py` | **`ANTHROPIC_API_KEY` in `.env`, a few dollars of credit** | ~65s (150 questions, 8-way parallel) | same — it's a live API, not cached locally |
| 7 | `./.venv/bin/python3 src/evaluate_e2e.py` | nothing (reads step 6's saved gate decisions — does not call the API again) | ~2–3 min | ~2–3 min |

Steps 1, 3, and 5 need network access the first time (LégisQuébec, then two
Hugging Face model downloads); step 6 needs a funded Anthropic API key.
Everything else runs offline once the models and dataset are cached.

**Approximate total cost of a full eval run: ~$0.40**, all of it from step 6
— the only step that calls a paid API. Steps 1–5 and 7 use local models and
cost nothing beyond compute time. Re-running step 6 alone re-costs ~$0.40
each time (prompt caching cuts the per-query cost after the first call, but
each run is a fresh set of live requests, not something that stays cached
between runs).

## Deployment

The web app (`src/api.py` + `static/index.html`) deploys to
[Render](https://render.com) as a single always-on web service, configured
by the committed `render.yaml`. Render fits this project well: a dashboard-driven,
GitHub-linked native Python runtime with instance sizes that scale to the
RAM this needs, and Blueprint (`render.yaml`) support so the whole
configuration is one reviewable file instead of dashboard clicks nobody can
audit later. (Fly.io and Railway are broadly comparable for a service this
size; Fly.io leans more CLI-driven, which matters less once this is set up
but is a worse first-time experience — Render's dashboard flow was chosen
for that reason, not because the others don't fit.)

### Instance size: Pro (4GB RAM), $85/month

Measured actual RAM use of the running server (both models loaded, after
handling real requests): **~1.35GB**, on this machine. That already leaves
uncomfortably little headroom on Render's Standard plan (2GB RAM, $25/month)
once you add Linux/container overhead, concurrent-request buffers, and the
fact that PyTorch's memory footprint on Linux x86_64 doesn't necessarily
match macOS ARM. An out-of-memory kill on a 2GB instance means the process
gets killed and restarted — reloading both models from scratch, several
minutes of downtime, repeating for as long as memory stays tight.

**Pro (4GB RAM / 2 CPU, $85/month)** is the size actually configured in
`render.yaml`. Standard ($25/month) is cheaper and might work — try it and
watch the logs for OOM kills if the cost matters more than the safety
margin — but it isn't what's committed, because "might work" isn't a
foundation to deploy legal-information infrastructure on.

One consequence worth stating plainly: **do not scale this to multiple
instances or add `--workers` to the uvicorn start command.** Both models
load fully into each process's own memory — a second worker or a second
instance doubles RAM use, not throughput-per-dollar. The rate limiter's
in-memory per-IP state is also local to one process; horizontal scaling
would need it moved to shared storage (e.g. Redis) to keep working
correctly. Neither is set up here, deliberately — this is a small, single-instance
deployment, not infrastructure for real scale.

### Data files: regenerated at build time from the tracked `articles.jsonl`, not re-scraped

`data/embeddings.npy` and `data/index.db` are gitignored (large-ish binary
build artifacts, not source). Two options existed for producing them on
Render, with a real tradeoff:

- **Re-scrape LégisQuébec on every deploy.** Always reflects the current
  page. But it makes every deploy depend on a government website being
  reachable and not rate-limiting Render's IP — a routine redeploy
  shouldn't be able to fail because of that, and LégisQuébec *did*
  rate-limit this project mid-session at least once. Worse for a legal-information
  app specifically: the Act's text would change on Render's servers the
  moment LégisQuébec's page changes, with no human ever reviewing the diff
  before it goes live.
- **Regenerate from the already-committed `data/articles.jsonl`** (chosen).
  `render.yaml`'s build command runs `chunk.py` then `embed.py` against the
  tracked article text — deterministic, no dependency on LégisQuébec being
  up, and no legal-content change ships without a human first re-running
  the scraper locally, reviewing the diff, and committing it. The real cost
  is staleness risk: if the LNT is amended and nobody re-runs the pipeline,
  the deployed text silently falls behind. That's an accepted, disclosed
  tradeoff, not an oversight — for legal content, a reviewed update beats
  an automatic one.

`data/index.db` (the FTS5 keyword index) isn't part of the build command at
all — it builds itself automatically the moment the app starts, inside
`Searcher.__init__` (see `search.py`), since a fresh deploy never has one
yet. That happens before the app opens its port, so it can't be reached by
a first request — indistinguishable in effect from "at build time" for the
purpose of the earlier "don't make the first user wait" requirement, even
though it's technically a startup step, not a build step.

### The two ML models download at build time, not on first request

`render.yaml`'s build command explicitly imports and instantiates both
`SentenceTransformer('intfloat/multilingual-e5-base')` and
`CrossEncoder('BAAI/bge-reranker-v2-m3')` before anything else — this pulls
both models' weights from Hugging Face into the build environment's local
cache. Render's native Python runtime builds and runs a service in the same
persistent environment (the build step doesn't get discarded the way some
platforms' isolated build containers do), so by the time `startCommand`
runs `uvicorn`, both models load from local disk in seconds, not over the
network. Without this, the *first real user's* request would trigger both
downloads inline — several minutes, not the 1-2 seconds `rerank=True`
normally costs.

### API cost estimate

Measured, not guessed: a typical answer-generation call (3 articles of
context, a normal-length question) costs **777 input tokens / 184 output
tokens** — at `claude-sonnet-4-6` pricing ($3/$15 per 1M tokens), about
**$0.0051**. Add the gate call ahead of it (~$0.0027 with prompt caching
warm, per the measurements above) and a typical in-scope question costs
**~$0.008**. A question the gate routes to `REDIRECT` costs only the gate
call, ~$0.0027, since answer generation never runs.

| daily questions | assuming mostly in-scope (~$0.008/q) | monthly |
|---|---|---|
| 50/day | ~$0.40/day | **~$12/month** |
| 500/day | ~$4.00/day | **~$120/month** |

That's Anthropic API spend only, separate from the $85/month Render
instance. The 10-questions-per-day-per-IP rate limit stays on in
production specifically to put a ceiling under this number — without it, a
single abusive IP could run this bill up arbitrarily.

### Exact steps in Render's dashboard

1. Make sure `render.yaml` is committed and pushed to the `main` branch of
   your GitHub repo (it already is, if you're reading this after the
   relevant commit).
2. Go to [dashboard.render.com](https://dashboard.render.com) and sign up
   or log in — signing in with GitHub is the simplest option, since you'll
   need to connect your GitHub account regardless.
3. Click the **New +** button (top right of the dashboard) and choose
   **Blueprint** from the dropdown.
4. If you haven't connected GitHub yet, Render prompts you to do so now —
   click **Connect GitHub**, follow the OAuth flow, and grant Render access
   to the repository (either all repos, or just this one — your choice).
5. Select your `lnt-assistant` repository from the list. Render reads
   `render.yaml` from the repo root automatically and shows you a preview
   of what it's about to create: one web service named `lnt-assistant`.
6. Because `render.yaml` marks `ANTHROPIC_API_KEY` with `sync: false`,
   Render shows a text field asking you to provide that value right there
   in this setup flow. Paste in a real Anthropic API key from an account
   with credit on it. This value is stored by Render, shown only to you in
   the dashboard, and never written into the repo.
7. Click **Apply** (sometimes labeled **Create New Resources**) to confirm
   and start the first deploy.
8. You're taken to the new service's page. Click the **Logs** tab to watch
   the build happen in real time — `pip install`, the two Hugging Face
   model downloads, `chunk.py`, `embed.py`, then the server starting up.
   This first build is slow (torch plus ~2.2GB of model weights) — budget
   10-15 minutes, not 1-2.
9. When the logs show `Uvicorn running on http://0.0.0.0:$PORT` and the
   service's status badge turns green ("Live"), the app is up. Render
   assigns it a URL of the form `https://lnt-assistant.onrender.com` —
   click it (or find it at the top of the service page) to open the app.
10. Test it: ask a real question in the browser. If nothing loads or you
    get a 500, click **Logs** again — a missing/invalid `ANTHROPIC_API_KEY`
    or an out-of-memory kill are the two most likely first-deploy problems,
    and both show up clearly there.
11. **Auto-deploy is on by default**: every push to `main` triggers a new
    build and deploy automatically. To ship an updated Act (after
    re-running the scraper locally and reviewing the diff — see the
    tradeoff above), commit the new `data/articles.jsonl` and push; no
    manual redeploy step is needed. To change the API key later, go to the
    service's **Environment** tab and edit `ANTHROPIC_API_KEY` there.

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
- **The deployed web app is a single instance with in-memory state.** The
  per-IP rate limiter resets on every redeploy and isn't shared across
  instances — this is fine at the current single-instance scale (see
  [Deployment](#deployment)) but wouldn't survive naive horizontal scaling
  without moving that state somewhere shared. There's also no logging or
  monitoring of what gets asked in production beyond Render's raw request
  logs — no way to know from the deployed app alone whether real usage
  looks like the eval set or not.
