"""Jurisdiction gate: an LLM classifier that routes a worker's question to one
of three outcomes before retrieval runs — REDIRECT (wrong jurisdiction
entirely, no articles), EXCLUDED (the LNT covers the topic but excludes this
worker — retrieve anyway and flag it), or ALLOW (normal retrieval).

Grounded in the full text of articles 2 (territorial scope), 3 (exclusions),
and 3.1 (exceptions to the exclusions). Article 3 doesn't cover federal
jurisdiction (banks, airlines, telecom, interprovincial transport) — that
exclusion comes from the constitutional division of powers, not the statute
itself — so the system prompt states it explicitly as background the model
needs but the Act doesn't spell out.

v1 (binary applies=true/false/unclear) wrongly blocked two in-scope
questions — a temp-agency permit question and a group-insurance-continuity
question — both of which the LNT answers directly (articles 92.5 and 79.3).
In both cases the model's stated reason was that the *topic* "belongs to"
some other area of law (agency licensing, insurance contracts) — it was
classifying by subject-matter feel using its own background knowledge,
instead of only checking whether *this worker* is in the population articles
2/3/3.1 describe. The prompt below says this explicitly: the gate's only job
is worker-coverage, not topic-ownership, and it is not expected to know
everything the Act regulates.

Usage:
    .venv/bin/python src/gate.py "some question"   # one-off check
    from gate import classify
    classify("I fly for an airline, does this apply to me?")
"""
import json
import os
from pathlib import Path
from typing import Literal, Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTICLES_PATH = DATA_DIR / "articles.jsonl"

load_dotenv(ROOT / ".env")  # loads ANTHROPIC_API_KEY from the project's .env file

MODEL = "claude-sonnet-4-6"

# Pricing per the Anthropic API, $/1M tokens (see report for source).
INPUT_PRICE_PER_M = 3.00
OUTPUT_PRICE_PER_M = 15.00


class GateResult(BaseModel):
    route: Literal["REDIRECT", "EXCLUDED", "ALLOW"]
    confident: bool  # false = this was a genuine judgment call, not clear-cut
    reason: str
    redirect: Optional[str] = None


def _load_scope_articles() -> dict:
    articles = {}
    with ARTICLES_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["article"] in ("2", "3", "3.1"):
                articles[rec["article"]] = rec
    return articles


def build_system_prompt() -> str:
    arts = _load_scope_articles()

    def block(num, title):
        r = arts[num]
        return (
            f"--- Article {num} ({title}) ---\n"
            f"French: {r['text_fr']}\n"
            f"English: {r['text_en']}\n"
        )

    return f"""You decide, before any document retrieval happens, which of three routes a
worker's question about Quebec's Loi sur les normes du travail (LNT, chapter
N-1.1) should take. The question is short and may be in French or English.

--- Your ONLY job: is this worker in the population the Act covers? ---

You are answering exactly one question: is the person asking — as an
employee, in their described sector and employment situation — within the
population that articles 2 and 3 define, or excluded from it? That is the
entire test. Use it, and nothing else, to choose a route.

Do NOT decide based on whether the specific TOPIC of their question (wage
garnishment, agency permits, insurance continuity during leave, whatever it
is) sounds like it "belongs to" some other specialized area of law you know
about. The LNT regulates many topics that sound adjacent to other fields —
that a topic sounds specialized or unfamiliar to you is not evidence it's
outside the Act. You are not expected to know everything sections 4 through
172 of this Act cover, and you must not guess that a topic falls outside the
Act just because you can't immediately place which provision addresses it.
Retrieval — which runs after you, over the full text of the Act — is what
finds the specific provision. Your only question is whether this WORKER, in
this SECTOR, doing this KIND of work, is in-scope at all. If they are a
covered employee in a covered sector, route them ALLOW no matter how narrow,
technical, or adjacent-to-another-field the specific topic feels.

{block("2", "territorial scope — who and where")}
{block("3", "exclusions — who the Act does not cover")}
{block("3.1", "exceptions — a few standards from the Act still apply to some excluded groups (occasional domestic/caregiving workers, construction workers, senior managers) even though they're otherwise excluded, e.g. certain family/parental leave and anti-retaliation provisions")}

One thing these articles do NOT say explicitly, but that you need to know:
Canada's constitutional division of powers puts certain sectors under
exclusively federal jurisdiction, governed by the Canada Labour Code instead
of any province's labour standards act — this includes banks, airlines and
other air transport, telecommunications and broadcasting companies,
interprovincial and international trucking/rail/shipping, and federal Crown
corporations (e.g. Radio-Canada/CBC). A worker in one of these federally
regulated industries is NOT covered by the LNT no matter what article 3
says, because the LNT (a provincial statute) has no jurisdiction over them
at all.

--- Choosing a route ---

route="REDIRECT" — a different jurisdiction or body governs this entirely,
or the question isn't about an employment relationship at all. No article
will help; retrieval should not even run. Cases:
  - Federally regulated sector (see above) -> redirect names the Canada
    Labour Code / the federal Labour Program (Employment and Social
    Development Canada).
  - Construction industry under R-20 (article 3, paragraph 3) -> redirect
    names the CCQ (Commission de la construction du Québec).
  - Not an employment question at all (e.g. tenant/landlord disputes, EI
    claims, immigration status, a parking ticket, anything with no employer
    and no work being performed) -> redirect is null; say so plainly in
    "reason".

route="EXCLUDED" — the worker falls into one of article 3's OTHER named
exclusions (occasional/casual domestic or caregiving work, work-study
students, athletes tied to a school program, senior managers, workers whose
pay is government-tariff-regulated under another Act) — categories with no
single alternate law or body to name. Retrieval should still run: article 3
itself, which names and explains the exclusion, is a genuinely correct and
useful answer, and article 3.1 may restore a handful of specific
protections (e.g. harassment, some family/parental leave) that still apply
despite the exclusion — those are worth surfacing too. redirect is usually
null here; "reason" should say which exclusion applies and, if relevant,
that a few 3.1 protections still apply.

route="ALLOW" — the worker is a covered employee in a covered sector, full
stop. This is the default. Do not downgrade to EXCLUDED or REDIRECT just
because you're not sure which specific provision answers their question —
being unsure which article applies is not the same as being sure the Act
doesn't apply. Genuine self-employment / independent-contractor situations
(freelancers, gig-platform workers with their own equipment and schedule)
are usually not "employees" under article 1's definition at all, which is a
real basis for something other than ALLOW even though it isn't spelled out
in articles 2/3/3.1 — treat these like EXCLUDED (retrieval can still be
informative) rather than REDIRECT, and set confident=false when the
employee/contractor line genuinely depends on facts you don't have (e.g.
platform work, family businesses without a formal contract).

Set confident=false whenever this was a genuine judgment call rather than a
clear-cut read of the articles — do NOT let low confidence push you toward
REDIRECT or EXCLUDED. A hedge should default to ALLOW (with reason noting
the uncertainty as a caveat), not cost the person their answer.

Keep "reason" to one or two sentences, in the same language as the
question."""


def classify(question: str, client: anthropic.Anthropic = None) -> tuple:
    """Returns (GateResult, usage) where usage has input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens."""
    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": build_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
        output_format=GateResult,
    )
    usage = response.usage
    return response.parsed_output, usage


def query_cost(usage) -> float:
    """Cost in USD for one classify() call, accounting for cache discounts."""
    input_cost = (usage.input_tokens or 0) * INPUT_PRICE_PER_M / 1_000_000
    output_cost = (usage.output_tokens or 0) * OUTPUT_PRICE_PER_M / 1_000_000
    cache_read = (getattr(usage, "cache_read_input_tokens", None) or 0) * (INPUT_PRICE_PER_M * 0.1) / 1_000_000
    cache_write = (getattr(usage, "cache_creation_input_tokens", None) or 0) * (INPUT_PRICE_PER_M * 1.25) / 1_000_000
    return input_cost + output_cost + cache_read + cache_write


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: gate.py \"question\"")
        return
    question = sys.argv[1]
    result, usage = classify(question)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print(f"\ncost: ${query_cost(usage):.6f}  usage: {usage}")


if __name__ == "__main__":
    main()
