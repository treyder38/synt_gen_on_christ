import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
import numpy as np
import albumentations as A
from pathlib import Path
import fitz
from PIL import Image
from typing import Dict, Optional

from document_pipeline.topic_generation import generate_topic
from document_pipeline.content_generation import generate_text
from document_pipeline.text_split import split_to_blocks
from document_pipeline.layout_generation import generate_layout
from utils.generate_json_with_sizes import generate_json_with_sizes
from utils.render_ans import render_blocks_json_to_pdf

logger = logging.getLogger(__name__)


OUT_ROOT = Path("/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out")


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


def _bleed_through_image(x, **kwargs):
    """Approximate bleed-through by mixing the image with a flipped copy."""
    x_f = x.astype(np.float32)
    flipped_f = np.flip(x, axis=1).astype(np.float32)
    a = 0.78 + 0.07 * np.random.rand()
    b = 0.10 + 0.20 * np.random.rand()
    c = 8.0 * np.random.rand()
    return (x_f * a + flipped_f * b + c).clip(0, 255).astype(np.uint8)


def augment_image(
    pdf_path: str,
    dpi: int,
    out_image_path: str,
) -> None:

    out_path = Path(out_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(Path(pdf_path)))

    try:
        page = doc.load_page(0)
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")
        img = np.array(pil_img)

        aug = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(std_range=(0.03, 0.18), mean_range=(0.0, 0.0), p=1.0),
                        A.ISONoise(color_shift=(0.03, 0.10), intensity=(0.3, 1.0), p=1.0),
                    ],
                    p=0.85,
                ),
                A.OneOf(
                    [
                        A.CoarseDropout(
                            num_holes_range=(200, 1200),
                            hole_height_range=(1, 2),
                            hole_width_range=(1, 2),
                            fill=0,
                            p=1.0,
                        ),
                        A.CoarseDropout(
                            num_holes_range=(200, 1200),
                            hole_height_range=(1, 2),
                            hole_width_range=(1, 2),
                            fill=255,
                            p=1.0,
                        ),
                    ],
                    p=0.7,
                ),
                A.Lambda(image=_bleed_through_image, p=0.6),
                # Color cast: slightly yellowish (warm) OR grayish (desaturated)
                A.OneOf(
                    [
                        # Warm/yellowish: boost R+G and reduce B a bit
                        A.RGBShift(r_shift_limit=(0, 20), g_shift_limit=(0, 20), b_shift_limit=(-20, 0), p=1.0),
                        # Grayish: desaturate / convert to gray
                        A.ToGray(p=1.0),
                        # Mild desaturation without full grayscale
                        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-35, -10), val_shift_limit=0, p=1.0),
                    ],
                    p=0.75,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.12, p=0.7),
                A.ImageCompression(
                    quality_range=(25, 80),
                    p=0.6,
                ),
            ]
        )

        out = aug(image=img)["image"]
        Image.fromarray(out).convert("RGB").save(str(out_path))
    finally:
        doc.close()


def doc_pipeline(sampled_persona: str, style_map: Optional[Dict[str, Dict[str, float]]]) -> Path:

    MODEL = "Qwen/Qwen2.5-32B-Instruct"

    run_dir = _make_next_run_dir()
    logger.info("Run directory: %s", str(run_dir))

    topic = generate_topic(sampled_persona, model = MODEL)
    logger.info("Topic: %s", topic)

    text = generate_text(sampled_persona, topic, model = MODEL)
    logger.info("Text: %s", text)

    split_json = split_to_blocks(text)
    # out_path = run_dir / "split.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(split_json, f, ensure_ascii=False, indent=2)
    #logger.info("Split saved to: %s", str(out_path))

    json_with_bbox_sizes = generate_json_with_sizes(split_json, style_map=style_map)
    # out_path = run_dir / "json_with_bbox_sizes.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(json_with_bbox_sizes, f, ensure_ascii=False, indent=2)
    # logger.info("json_with_bbox_sizes saved to: %s", str(out_path))

    final_layout = generate_layout(data=json_with_bbox_sizes, style_map=style_map)
    out_path = f"{str(run_dir)}/ans.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_layout, f, ensure_ascii=False, indent=2)
    logger.info("Layout saved to: %s", str(out_path))

    pdf_path = render_blocks_json_to_pdf(
        json_path=f"{str(run_dir)}/ans.json",
        out_pdf_path=f"{str(run_dir)}/out.pdf",
        draw_frames=False,
        draw_word_bboxes=False,
        style_map=style_map,
    )
    #logger.info("Render saved to: %s", pdf_path)

    # TODO: подготовить все для генерации на s3
    aug_img_path = f"{str(run_dir)}/doc.png"
    augment_image(
        pdf_path=pdf_path,
        dpi=style_map["dpi"],
        out_image_path=aug_img_path
    )
    logger.info("Augmented page saved to: %s", aug_img_path)

    os.remove(pdf_path)

    return run_dir