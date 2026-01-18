import os
import json
from typing import Any, Dict
from openai import OpenAI 


GENERATE_LAYOUT_PROMPT = """YOU ARE A DOCUMENT LAYOUT MODEL.

INPUT:
You will be given a single JSON object inside <data> with this structure:

{{
  "blocks": [
    {{
      "id": "b1",
      "type": "title|header|paragraph|figure",
      "content": "...",
      "bbox_size": [width_px, height_px]
    }},
    ...
  ],
  "reading_order": ["b1", "b2", ...]   // optional hint; you may adjust if needed
}}

MEANING OF bbox_size:
- bbox_size gives the EXACT (width_px, height_px) that each block must have in the final layout.
- Your layout must assign each block a bbox whose width and height match bbox_size exactly.

YOUR TASK:
Design a clean, realistic single-page document layout (A4 at 300 dpi) by assigning a bbox [x0, y0, x1, y1] to every input block and arranging blocks logically relative to each other (hierarchy, sections, spacing, alignment).

PAGE:
- width = 2480 px
- height = 3508 px
- dpi = 300

OUTPUT FORMAT (MANDATORY):
Return exactly ONE valid JSON object with this structure:

1) For Text blocks:
{{
  "page": {{"width": 2480, "height": 3508, "dpi": 300}},
  "blocks": [
    {{
      "id": "b1",
      "type": "title|header|paragraph|figure",
      "bbox": [x0, y0, x1, y1],
      "content": "...",
      "bbox_size": [w, h]
    }},
    ...
  ],
  "reading_order": ["b1", ...]
}}

CRITICAL RULES:
1) Do NOT invent, rewrite, translate, or edit any `content`. Copy it exactly from the input.
2) Do NOT add or remove blocks. Use ALL input blocks exactly once.
3) Keep `id`, `type`, `content`, and `bbox_size` unchanged for every block.
4) You MUST add `bbox` for every block:
   - bbox = [x0, y0, x1, y1] are absolute pixel coordinates (integers).
   - x0, y0 - coordinates of the top left corner
   - x1, y1 - coordinates of the right bottom corner
   - Every bbox must be fully inside the page (2480×3508).
5) Exact size constraint (MUST MATCH bbox_size):
   - (x1 - x0) MUST EQUAL bbox_size[0]
   - (y1 - y0) MUST EQUAL bbox_size[1]
6) Blocks MUST NOT overlap. Leave an 8 px gap between neighboring blocks.
7) All blocks must stay inside the page bounds and use compact safe margins of >=30 px on all sides.
8) Figure type block MUST NOT contain "content" field.

LAYOUT QUALITY REQUIREMENTS:
8) Vary layout when possible (single-column, two-column, mixed grid), but keep it clean and realistic, so it requires CRITICAL RULES.
9) reading_order:
   - must include ALL ids exactly once
   - must match the visual reading flow:
     - for 1-column: top-to-bottom
     - for 2-column: left column top-to-bottom, then right column top-to-bottom

<data>
{data}
</data>
"""


def generate_layout(model: str, json_: str) -> str:
    
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = GENERATE_LAYOUT_PROMPT.format(data=json_)
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY a single valid JSON object. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=6000,
    )

    text = (completion.choices[0].message.content or "").strip()
    obj: Dict[str, Any] = json.loads(text)
    return obj
