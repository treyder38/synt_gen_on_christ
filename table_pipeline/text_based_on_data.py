import os
import json
from typing import Any, Dict
from openai import OpenAI
from functools import lru_cache


GENERATE_DOCUMENT_TEXT_JSON_PROMPT = """You are an expert writer.
You are given:
- a persona: "{persona}"
- structured data describing the topic: {topic} (referred to as "data")

Here is the data in JSON format:
<data>
{data}
</data>

Your task is to generate a realistic, coherent, single-page document text based strictly on the provided data and adapted to the given persona. 
Do not invent facts that contradict the data; use the data as the factual backbone and expand it into natural language where appropriate.

Here are the requirements:
1. The text must be directly based on the provided data and customized according to the given persona.
2. The text must be realistic and grounded in Russian real-world context. Use Russian real-world entities, names, places, dates, and organizations when they are present in the data.
   Do NOT introduce placeholder names such as xxA, xxB, John Doe, or template markers like [Name], [Date], etc.
3. The text should cover different relevant aspects implied by the data to make the document informative, but concise.
4. The output must be written entirely in Russian, even if the persona is non-Russian.
5. Structure requirements:
- Start with a clear TITLE on the first line (1 sentence, no bullet/numbering).
- Do NOT use asterisks for emphasis or formatting. Use plain text only."""


@lru_cache(maxsize=32)
def _get_openai_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )


def _contains_non_latin_or_cyrillic_letters(text: str) -> bool:
    """Return True if text contains alphabetic letters outside Latin/Cyrillic.

    We allow:
      - Latin letters: A-Z, a-z, plus Latin-1 Supplement/Extended blocks (to keep it strict we ONLY allow basic Latin).
      - Cyrillic letters: including Ёё.

    Digits, punctuation, whitespace, and other symbols are allowed.
    """
    if not text:
        return False

    for ch in text:
        if ch.isalpha():
            code = ord(ch)
            is_basic_latin = (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A)
            is_cyrillic = (
                (0x0400 <= code <= 0x04FF)  # Cyrillic
                or (0x0500 <= code <= 0x052F)  # Cyrillic Supplement
                or (0x2DE0 <= code <= 0x2DFF)  # Cyrillic Extended-A
                or (0xA640 <= code <= 0xA69F)  # Cyrillic Extended-B
                or (0x1C80 <= code <= 0x1C8F)  # Cyrillic Extended-C
            )
            if not (is_basic_latin or is_cyrillic):
                return True

    return False


def generate_text(persona: str, topic: str, model: str, data: str, base_url: str) -> str:
    
    client = _get_openai_client(base_url)

    prompt = GENERATE_DOCUMENT_TEXT_JSON_PROMPT.format(persona=persona, topic=topic, data=data)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY the document body as plain text in Russian. Do not add any extra commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=3000,
    )

    text = (completion.choices[0].message.content or "").strip()

    if _contains_non_latin_or_cyrillic_letters(text):
        raise ValueError(
            "Generated text contains letters outside English/Russian alphabets; aborting."
        )

    return text