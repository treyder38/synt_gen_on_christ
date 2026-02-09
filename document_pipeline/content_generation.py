import os
import re
from openai import OpenAI
from functools import lru_cache


GENERATE_DOCUMENT_DATA_PROMPT = """You are an expert in content creation and have broad knowledge about various topics.
My persona is: "{persona}"
I need materials about "{topic}" that can be used to generate a realistic document.

Here are the requirements:
1. The materials must be directly related to the topic and customized according to the given persona.
2. Do NOT use placeholder names such as xxA, xxB, John Doe, or template markers like [Name], [Date], etc.
3. The materials should cover different relevant aspects of the topic to make the document informative.
4. All materials must be written in Russian, even if the persona is non-Russian.
5. Document MUST contain from 300 to 400 words.
6. Structure requirements:
- Start with a clear TITLE on the first line (1 sentence, no bullet/numbering).
- Do NOT use markdown headings with # symbols. Use plain text headers without any leading symbols.
- Do NOT use asterisks for emphasis or formatting. Use plain text only.
- If there are mulitple paragraphs, include a short header before each of them.
- If lists are helpful for clarity, include bullet list OR numbered list.
- Any list must be contextually appropriate (e.g., steps, criteria, pros/cons, checklist). If not appropriate, use normal paragraphs instead.
- Include concrete numbers where appropriate. Numbers must be plausible and consistent with the topic.
- Keep the tone professional and realistic; avoid generic fluff.
7. Strict character set:
- Do NOT use any English/Latin letters (A–Z, a–z). The whole document must be written using only Russian Cyrillic letters for words.
- Do NOT use special symbols such as @, %, & anywhere in the text."""


@lru_cache(maxsize=32)
def _get_openai_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )


def _contains_non_latin_or_cyrillic_letters(text: str) -> bool:
    """Return True if text contains forbidden characters.

    Forbidden:
      - Any Latin letters (A-Z, a-z)
      - Any alphabetic letters outside Cyrillic (including Ёё)
      - Specific special symbols: @, %, &

    Allowed:
      - Cyrillic letters (including Ёё)
      - Digits, whitespace, and common punctuation (.,:;!?-()[]{}"' etc.)
    """
    if not text:
        return False

    forbidden_symbols = {"@", "%", "&"}

    for ch in text:
        if ch in forbidden_symbols:
            return True

        if ch.isalpha():
            code = ord(ch)

            # Disallow any basic Latin letters explicitly.
            is_basic_latin = (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A)
            if is_basic_latin:
                return True

            # Allow Cyrillic letters (including Ёё).
            is_cyrillic = (
                (0x0400 <= code <= 0x04FF)  # Cyrillic
                or (0x0500 <= code <= 0x052F)  # Cyrillic Supplement
                or (0x2DE0 <= code <= 0x2DFF)  # Cyrillic Extended-A
                or (0xA640 <= code <= 0xA69F)  # Cyrillic Extended-B
                or (0x1C80 <= code <= 0x1C8F)  # Cyrillic Extended-C
            )
            if not is_cyrillic:
                return True

    return False


def generate_text(persona: str, topic: str, model: str, base_url: str) -> str:

    client = _get_openai_client(base_url)

    prompt = GENERATE_DOCUMENT_DATA_PROMPT.format(persona=persona, topic=topic)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY the document body as plain text in Russian. Do not add any extra commentary. Do NOT use markdown"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=3000,
    )

    text = (completion.choices[0].message.content or "").strip()

    if _contains_non_latin_or_cyrillic_letters(text):
        raise ValueError(
            "Generated text contains forbidden characters (Latin letters, non-Cyrillic letters, or @/%/&); aborting."
        )

    return text