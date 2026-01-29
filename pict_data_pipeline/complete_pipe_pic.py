import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any
import logging
from pathlib import Path
from datetime import datetime, timezone
import uuid
from PIL import Image
import fitz
import os

from pict_data_pipeline.topic_generation import generate_topic
from pict_data_pipeline.data_generation import generate_data
from pict_data_pipeline.code_generation import generate_code, save_generated_image
from pict_data_pipeline.text_based_on_data import generate_text
from pict_data_pipeline.text_split_with_image import split_to_blocks
from pict_data_pipeline.layout_generation_with_image import generate_layout
from utils.generate_json_with_sizes import generate_json_with_sizes
from utils.render_ans import render_blocks_json_to_pdf


logger = logging.getLogger(__name__)


OUT_ROOT = Path("/home/jovyan/people/Glebov/synt_gen_2/pict_data_pipeline/out")


def save_jpeg(pdf_path: str, out_jpeg_path: str, dpi: int = 300, quality: int = 70) -> Optional[str]:
    """Render the first page of a PDF to JPEG with compression.

    Tries PyMuPDF (fitz) first, then pdf2image if available.
    Returns the output path on success, otherwise None.
    """

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.save(out_jpeg_path, format="JPEG", quality=int(quality), optimize=True, progressive=True)
    doc.close()
    return out_jpeg_path


def _make_next_run_dir() -> Path:
    """Create a unique run directory under out/<time>_<uuid>/.

    Time is UTC in format YYYYMMDDTHHMMSSZ to keep lexicographic order.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{uuid.uuid4().hex}"

    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def pic_pipeline(sampled_persona: str, figure_type: str, style_map: Dict[str, Dict[str, float]], base_url: str, sbx: Any | None = None) -> Path:

    MODEL = "Qwen/Qwen2.5-14B-Instruct"

    run_dir = _make_next_run_dir()

    topic = generate_topic(sampled_persona, model = MODEL, figure_type=figure_type, base_url=base_url)
    # logger.info("Topic: %s", topic)

    data = generate_data(sampled_persona, topic, model = MODEL, figure_type=figure_type, base_url=base_url)
    # logger.info("Data: %s", data)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_code = executor.submit(
            generate_code,
            sampled_persona,
            topic,
            model=MODEL,
            data=data,
            figure_type=figure_type,
            base_url=base_url
        )
        future_text = executor.submit(
            generate_text,
            sampled_persona,
            topic,
            model=MODEL,
            data=data,
            base_url=base_url
        )

        code = future_code.result()
        text = future_text.result()

    # logger.info("Code: %s", code)
    # logger.info("Text: %s", text)

    picture = save_generated_image(code, sbx=sbx)

    split_json = split_to_blocks(text=text, figure_type=figure_type)
    # Put the generated data into the figure block content
    figure_payload = json.dumps(data, ensure_ascii=False)
    for b in split_json.get("blocks", []):
        if b.get("type") == "figure":
            b["content"] = figure_payload
            break
        
    out_path = f"{str(run_dir)}/split.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(split_json, f, ensure_ascii=False, indent=2)
    # logger.info("Split saved to: %s", out_path)

    json_with_bbox_sizes = generate_json_with_sizes(split_json, style_map=style_map, picture=picture)
    # out_path = f"{str(run_dir)}/json_with_bbox_sizes.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(json_with_bbox_sizes, f, ensure_ascii=False, indent=2)
    # logger.info("json_with_bbox_sizes saved to: %s", out_path)

    final_layout = generate_layout(data = json_with_bbox_sizes, style_map=style_map)
    out_path = f"{str(run_dir)}/ans.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_layout, f, ensure_ascii=False, indent=2)
    # logger.info("Layout saved to: %s", out_path)

    pdf_path = render_blocks_json_to_pdf(
        f"{str(run_dir)}/ans.json",
        out_pdf_path=f"{str(run_dir)}/out.pdf",
        draw_frames=False,
        draw_word_bboxes=False,
        style_map=style_map,
        picture=picture
    )
    # logger.info("Render saved to: %s", pdf_path)

    jpeg_path = f"{str(run_dir)}/out.jpg"
    saved_jpeg = save_jpeg(str(pdf_path), jpeg_path, dpi=300, quality=70)
    
    os.remove(pdf_path)

    return run_dir