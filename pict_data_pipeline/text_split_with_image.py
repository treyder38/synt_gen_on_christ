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
    {{ "id": "b3", "type": "paragraph", "content": "..." }},
    {{
      "id": "b4",
      "type": "figure",
      "figure_type": "{figure_type}"
    }}
  ],
  "reading_order": ["b1", "b2", "b3", "b4"]
}}

RULES (follow ALL):
1) Allowed block types ONLY: "title", "header", "paragraph", "figure".
2) EXACTLY ONE IMAGE BLOCK:
   - The output MUST contain exactly ONE block with "type": "figure".
   - That figure block MUST have EXACTLY this structure (no "content" field):
     {{
       "id": "<unique id>",
       "type": "figure",
       "figure_type": "{figure_type}"
     }}
   - Use integers for bbox values.
   - The figure block MAY be placed anywhere in the blocks list, but it MUST appear in reading_order.
3) VERBATIM TEXT ONLY:
   - Do NOT summarize, shorten, paraphrase, rewrite, translate, or correct the text.
   - Every "content" for text block MUST be copied verbatim from <data> (same wording, punctuation, numbers, and casing).
4) COMPLETE COVERAGE:
   - ALL text from <data> MUST appear in the output blocks.
   - Do NOT drop any characters/sentences.
   - Do NOT duplicate any text between blocks.
5) CONTIGUOUS SLICES + ORDER:
   - Each text block's "content" MUST be a contiguous substring of <data>.
   - Blocks MUST preserve the original order of the text.
6) SPLITTING HEURISTICS:
   - Separate the main title into a single "title" block (if it exists).
   - Each section heading/subheading MUST be its own "header" block.
   - Each paragraph MUST be its own "paragraph" block.
   - If the text contains bullet/numbered items (e.g., lines starting with "-", "–", "—", "•", "*", or "1.", "1)", "2.", etc.), then ALL contiguous list lines MUST be kept together inside ONE single "paragraph" block (verbatim).
   - If a paragraph is long, split it into multiple "paragraph" blocks at natural sentence boundaries (without removing any text).
7) IDS:
   - "id" must be unique and sequential: b1, b2, b3, ...
"""


def generate_split_json(model: str, text: str, figure_type: str) -> str:
    
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = GENERATE_TEXT_SPLIT_PROMPT.format(data=text, figure_type=figure_type)
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY a single valid JSON object. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        max_tokens=4500,
    )

    text = (completion.choices[0].message.content or "").strip()
    obj: Dict[str, Any] = json.loads(text)
    return obj