import os
import json
from typing import Any, Dict
from openai import OpenAI


GENERATE_TEXT_SPLIT_PROMPT = """You are given a text inside <data>. Your job is to split that text into logical text blocks.

<data>
{data}
</data>

OUTPUT (MANDATORY):
Return ONE valid JSON object with EXACTLY this structure:
{{
  "blocks": [
    {{ "id": "b1", "type": "title", "content": "..." }},
    {{ "id": "b2", "type": "header", "content": "..." }},
    {{ "id": "b3", "type": "paragraph", "content": "..." }}
  ]
}}

RULES (follow ALL):
1) Allowed block types ONLY: "title", "header", "paragraph".
2) VERBATIM TEXT ONLY:
   - Do NOT summarize, shorten, paraphrase, rewrite, translate, or correct the text.
   - Every "content" MUST be copied verbatim from <data> (same wording, punctuation, numbers, and casing).
3) COMPLETE COVERAGE:
   - ALL text from <data> MUST appear in the output blocks.
   - Do NOT drop any characters/sentences.
   - Do NOT duplicate any text between blocks.
4) CONTIGUOUS SLICES + ORDER:
   - Each block's "content" MUST be a contiguous substring of <data>.
   - Blocks MUST preserve the original order of the text.
5) SPLITTING HEURISTICS:
   - Separate the main title into a single "title" block (if it exists).
   - Each section heading/subheading MUST be its own "header" block.
   - Each paragraph MUST be its own "paragraph" block.
   - If the text contains bullet/numbered items (e.g., lines starting with "-", "–", "—", "•", "*", or "1.", "1)", "2.", etc.), then ALL contiguous list lines MUST be kept together inside ONE single "paragraph" block (verbatim).
   - If a paragraph is long, split it into multiple "paragraph" blocks at natural sentence boundaries (without removing any text).
6) IDS:
   - "id" must be unique and sequential: b1, b2, b3, ...
   """


def generate_split_json(model: str, text: str) -> str:

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    prompt = GENERATE_TEXT_SPLIT_PROMPT.format(data=text)
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        max_tokens=4000,
    )

    text = (completion.choices[0].message.content or "").strip()
    obj: Dict[str, Any] = json.loads(text)
    return obj