import json
import logging
from typing import Dict, Optional
from document_pipeline.topic_generation import generate_topic
from document_pipeline.content_generation import generate_text
from document_pipeline.text_split import generate_split_json
from document_pipeline.layout_generation import generate_layout
from utils.generate_json import generate_json_with_sizes
from utils.render_ans import render_blocks_json_to_pdf

logger = logging.getLogger(__name__)

def strip_content_fields(json_obj: Dict) -> Dict:
    """Return a copy of the layout input JSON without any `content` fields in blocks.

    This is useful when the layout model should not see the raw text.
    """
    blocks = json_obj.get("blocks", [])
    stripped_blocks = []
    for b in blocks:
        if isinstance(b, dict):
            b2 = dict(b)
            b2.pop("content", None)
            stripped_blocks.append(b2)
        else:
            stripped_blocks.append(b)
    out = dict(json_obj)
    out["blocks"] = stripped_blocks
    return out

def merge_content_fields(layout_json: Dict, source_json: Dict) -> Dict:
    """Return a copy of `layout_json` where blocks get their `content` field restored
    from `source_json` (matched by block `id`).

    `source_json` is expected to contain the full text content in `blocks[*].content`.
    """
    src_blocks = source_json.get("blocks", [])
    id_to_content = {}
    for b in src_blocks:
        if isinstance(b, dict) and "id" in b and "content" in b:
            id_to_content[b["id"]] = b["content"]

    out = dict(layout_json)
    blocks = layout_json.get("blocks", [])
    merged_blocks = []
    for b in blocks:
        if isinstance(b, dict):
            b2 = dict(b)
            bid = b2.get("id")
            if bid in id_to_content:
                b2["content"] = id_to_content[bid]
            merged_blocks.append(b2)
        else:
            merged_blocks.append(b)
    out["blocks"] = merged_blocks
    return out

def doc_pipeline(sampled_persona: str, style_map: Optional[Dict[str, Dict[str, float]]]) -> None:

    MODEL = "Qwen/Qwen2.5-32B-Instruct"

    topic = generate_topic(sampled_persona, model = MODEL)
    logger.info("Topic: %s", topic)

    text = generate_text(sampled_persona, topic, model = MODEL)
    logger.info("Text: %s", text)

    split_json = generate_split_json(model = MODEL, text=text)
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)
    logger.info("Split saved to: %s", out_path)

    json_with_bbox_sizes = generate_json_with_sizes(split_json, style_map=style_map)
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/json_with_bbox_sizes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_with_bbox_sizes, f, ensure_ascii=False, indent=2)
    logger.info("json_with_bbox_sizes saved to: %s", out_path)

    json_for_layout = strip_content_fields(json_with_bbox_sizes)
    final_layout = generate_layout(
        model=MODEL,
        json_=json.dumps(json_for_layout, ensure_ascii=False, separators=(",", ":")),
    )

    final_layout = merge_content_fields(final_layout, json_with_bbox_sizes)

    out_path = "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/ans.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_layout, f, ensure_ascii=False, indent=2)
    logger.info("Layout saved to: %s", out_path)

    pdf_path = render_blocks_json_to_pdf(
        "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/ans.json",
        out_pdf_path="/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/out.pdf",
        draw_frames=True,
        style_map=style_map
    )
    logger.info("Render saved to: %s", pdf_path)