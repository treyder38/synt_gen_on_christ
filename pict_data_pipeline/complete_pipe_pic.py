import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from pict_data_pipeline.topic_generation import generate_topic
from pict_data_pipeline.data_generation import generate_data
from pict_data_pipeline.code_generation import generate_code, save_generated_image
from pict_data_pipeline.text_based_on_data import generate_text
from pict_data_pipeline.text_split_with_image import split_to_blocks
from pict_data_pipeline.layout_generation_with_image import generate_layout
from utils.generate_json import generate_json_with_sizes
from utils.render_ans import render_blocks_json_to_pdf

import logging

logger = logging.getLogger(__name__)


def pic_pipeline(sampled_persona: str, figure_type: str, style_map: Optional[Dict[str, Dict[str, float]]]) -> None:

    MODEL = "Qwen/Qwen2.5-32B-Instruct"

    topic = generate_topic(sampled_persona, model = MODEL, figure_type=figure_type)
    logger.info("Topic: %s", topic)

    data = generate_data(sampled_persona, topic, model = MODEL, figure_type=figure_type)
    logger.info("Data: %s", data)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_code = executor.submit(
            generate_code,
            sampled_persona,
            topic,
            model=MODEL,
            data=data,
            figure_type=figure_type,
        )
        future_text = executor.submit(
            generate_text,
            sampled_persona,
            topic,
            model=MODEL,
            data=data,
        )

        code = future_code.result()
        text = future_text.result()

    logger.info("Code: %s", code)
    logger.info("Text: %s", text)

    filename = f"/home/jovyan/people/Glebov/synt_gen_2/pict_data_pipeline/out/graph.png"
    out_pic_file = save_generated_image(code, filename)
    logger.info("Pic saved to: %s", out_pic_file)

    split_json = split_to_blocks(text=text, figure_type=figure_type)
    # Put the generated data into the figure block content
    figure_payload = json.dumps(data, ensure_ascii=False)
    for b in split_json.get("blocks", []):
        if b.get("type") == "figure":
            b["content"] = figure_payload
            break
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/pict_data_pipeline/out/split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)
    logger.info("Split saved to: %s", out_path)

    json_with_bbox_sizes = generate_json_with_sizes(split_json, style_map=style_map, picture_path=out_pic_file)
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/pict_data_pipeline/out/json_with_bbox_sizes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_with_bbox_sizes, f, ensure_ascii=False, indent=2)
    logger.info("json_with_bbox_sizes saved to: %s", out_path)

    final_layout = generate_layout(data = json_with_bbox_sizes)
    out_path = "/home/jovyan/people/Glebov/synt_gen_2/pict_data_pipeline/out/ans.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_layout, f, ensure_ascii=False, indent=2)
    logger.info("Layout saved to: %s", out_path)

    pdf_path = render_blocks_json_to_pdf(
        "/home/jovyan/people/Glebov/synt_gen_2/pict_data_pipeline/out/ans.json",
        out_pdf_path="/home/jovyan/people/Glebov/synt_gen_2/pict_data_pipeline/out/out.pdf",
        draw_frames=False,
        style_map=style_map,
        picture_path=out_pic_file
    )
    logger.info("Render saved to: %s", pdf_path)