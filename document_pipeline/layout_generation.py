import os
import json
from typing import Any, Dict
from openai import OpenAI 


GENERATE_LAYOUT_PROMPT = """YOU ARE A SINGLE-PAGE DOCUMENT LAYOUT ENGINE.

Given input blocks with fixed sizes, place them on an A4 page (300 dpi) by assigning a bbox to each block.

PAGE (A4 @ 300 dpi)
- width: 2480 px
- height: 3508 px

INPUT (inside <data>)
A single JSON object:
{{
  "blocks": [
    {{"id": "b1", "type": "title|header|paragraph", "bbox_size": [w_px, h_px]}},
    ...
  ]
}}

IMPORTANT: bbox_size IS EXACT
For each block, the output bbox must have:
- (x1 - x0) == bbox_size[0]
- (y1 - y0) == bbox_size[1]

OUTPUT (MANDATORY)
Return EXACTLY ONE valid JSON object:
{{
  "page": {{"width": 2480, "height": 3508, "dpi": 300}},
  "blocks": [
    {{"id": "b1", "type": "...", "bbox": [x0,y0,x1,y1], "bbox_size": [w,h]}},
    ...
  ]
}}

CRITICAL RULES (MUST FOLLOW)
2) DO NOT add/remove blocks. Use ALL input blocks exactly once.
3) Keep `id`, `type`, `bbox_size` unchanged for every block.
4) You MUST output `bbox` for every block:
   - bbox = [x0, y0, x1, y1] are absolute integer pixel coordinates.
   - (x0,y0) = top-left, (x1,y1) = bottom-right.
5) Exact size constraint (required):
   - (x1 - x0) MUST equal bbox_size[0]
   - (y1 - y0) MUST equal bbox_size[1]
6) No overlap. Keep at least an 8 px gap between any two blocks.
7) All blocks must be fully inside the page bounds.

LAYOUT QUALITY (MUST STILL OBEY ALL RULES)
8) STRONGLY PREFER a non-trivial layout (avoid "everything in one vertical column"):
   - Choose ONE archetype and follow it consistently:
     A) Headline + two columns: title full-width, then body split into left/right columns.
     B) Sidebar: main column + narrow sidebar for 1–3 smaller blocks.
     C) Mixed grid: occasional full-width headers, then two-column sections.
   - Use full-width blocks only for title/major section breaks or if a block cannot fit a column.
   - Keep alignment clean: consistent column edges, consistent vertical rhythm, and 8 px gaps.
9) Column planning guidance (still must match bbox_size exactly):
   - Page margins: 120 px left/right, 140 px top, 120 px bottom.
   - Preferred column gutter: 40 px.
   - If bbox_size[0] <= (usable_width - gutter)/2, it CAN fit in a column.
   - If bbox_size[0] is larger than a single column width, place it full-width (within margins).
10) Reading order:
   - match the visual reading flow.
   - for 2-column: left column top-to-bottom, then right column top-to-bottom.
   - for sidebar: main column top-to-bottom, then sidebar top-to-bottom.

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
