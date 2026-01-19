import json
import logging
from typing import Dict, Optional
from document_pipeline.topic_generation import generate_topic
from document_pipeline.content_generation import generate_text
from document_pipeline.text_split import split_to_blocks
from document_pipeline.layout_generation import generate_layout
from utils.generate_json import generate_json_with_sizes
from utils.render_ans import render_blocks_json_to_pdf

logger = logging.getLogger(__name__)


def doc_pipeline(sampled_persona: str, style_map: Optional[Dict[str, Dict[str, float]]]) -> None:

    MODEL = "Qwen/Qwen2.5-32B-Instruct"

    topic = generate_topic(sampled_persona, model = MODEL)
    logger.info("Topic: %s", topic)

    text = generate_text(sampled_persona, topic, model = MODEL)
    logger.info("Text: %s", text)

    split_json = split_to_blocks(text)
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)
    logger.info("Split saved to: %s", out_path)

    json_with_bbox_sizes = generate_json_with_sizes(split_json, style_map=style_map)
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/json_with_bbox_sizes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_with_bbox_sizes, f, ensure_ascii=False, indent=2)
    logger.info("json_with_bbox_sizes saved to: %s", out_path)

    final_layout = generate_layout(data=json_with_bbox_sizes)
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/ans.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_layout, f, ensure_ascii=False, indent=2)
    logger.info("Layout saved to: %s", out_path)

    pdf_path = render_blocks_json_to_pdf(
        json_path="/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/ans.json",
        out_pdf_path="/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/out.pdf",
        draw_frames=True,
        style_map=style_map
    )
    logger.info("Render saved to: %s", pdf_path)