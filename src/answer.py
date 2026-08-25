"""Answer generation: an Anthropic call grounded ONLY in the top-k retrieved
article texts. This is the boundary where hallucination would enter the
system, so the prompt is deliberately restrictive: state a rule only if it's
written in the provided text, cite the article number for every claim, and
say plainly when the provided articles don't answer the question rather than
constructing a plausible-sounding answer from a nearby-but-different article.

Usage:
    from answer import generate_answer
    generate_answer(
        "combien de semaines de vacances après 3 ans", "fr",
        [{"article": "69", "text": "Une personne salariée qui..."}],
    )
"""
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 700

NO_ARTICLES_MESSAGE = {
    "fr": (
        "Je n'ai trouvé aucun article de la Loi sur les normes du travail qui "
        "réponde à cette question. Je préfère vous le dire plutôt que de deviner."
    ),
    "en": (
        "I couldn't find any article of the Act that answers this question. "
        "I'd rather tell you that than guess."
    ),
}


def build_system_prompt(lang: str) -> str:
    lang_name = "French" if lang == "fr" else "English"
    return f"""You answer questions about Quebec's Loi sur les normes du travail (LNT,
chapter N-1.1) using ONLY the article text provided in the user message
below. You have no other source of truth here — not your training data
about this Act, not what you recall a provision "usually" says, only the
text given to you in this exact request.

Rules, in order of importance:

1. State something as a rule ONLY if it is written in the provided article
   text. Do not add thresholds, dollar amounts, percentages, day/week counts,
   exceptions, or conditions that aren't there — even if they sound right,
   even if you're confident you know the real number. If the provided text
   doesn't give a specific figure the question asks for, say that plainly
   instead of supplying one from memory.

2. If the provided articles do not actually answer the question — including
   if they're only adjacent or related but don't address what was asked —
   say so directly and plainly. Do not stretch a related article into an
   answer it doesn't give. A partial answer should be presented as partial,
   naming what is and isn't covered by the text you were given.

3. Cite the article number inline every time you state something drawn from
   it — e.g. "(art. 69)" in French, "(section 69)" in English — placed right
   next to the specific claim it supports, not just once at the end.

4. Answer in {lang_name}, regardless of what language the provided article
   text happens to be in.

5. Be concise: 2-5 sentences is normal. This is a direct answer, not a legal
   memo.

6. Plain text only — no markdown (no **bold**, no bullet lists, no headers).
   The interface that displays this renders it as plain text, so markdown
   syntax would show up as literal asterisks.

7. Do not give legal advice, strategic recommendations, or predict how a
   specific case would turn out. State what the text says; do not tell the
   person what to do about it.

If you were given a note under "Additional context to mention," work it
into the answer naturally — it's information the retrieval/routing system
already determined applies (e.g. an exclusion under article 3), not
something to omit or second-guess."""


def generate_answer(question: str, lang: str, articles: list, caveat: str = None,
                     client: anthropic.Anthropic = None) -> str:
    """articles: list of {"article": "69", "text": "full article text"},
    already in the target language. Returns the answer text."""
    if not articles:
        return NO_ARTICLES_MESSAGE.get(lang, NO_ARTICLES_MESSAGE["en"])

    client = client or anthropic.Anthropic()

    context = "\n\n".join(f"--- Article {a['article']} ---\n{a['text']}" for a in articles)
    user_content = f"Provided articles:\n\n{context}\n\nQuestion: {question}"
    if caveat:
        user_content += f"\n\nAdditional context to mention in your answer: {caveat}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_prompt(lang),
        messages=[{"role": "user", "content": user_content}],
    )
    return next(b.text for b in response.content if b.type == "text")
