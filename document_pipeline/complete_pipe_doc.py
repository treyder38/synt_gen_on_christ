import json
import logging
import os
import re
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


OUT_ROOT = Path("/home/jovyan/people/Glebov/synt_gen_2/out")


def _make_next_run_dir() -> Path:
    """Create out/run{N} where N == number of existing run* directories."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir() and p.name.startswith("run")]
    n = len(run_dirs)

    run_dir = OUT_ROOT / f"run{n}"
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
    in_pdf_path: str,
    out_image_path: str,
    dpi: int = 300,
    jpeg_quality_range: tuple[int, int] = (35, 95),
) -> Optional[str]:
    """Render a single-page PDF to an image, apply document-like augmentations, save as an image.

    Returns the written image path, or None if skipped.

    Notes:
    - Assumes the input PDF is single-page; if not, uses the first page only.
    - Requires PyMuPDF (fitz) + Pillow.
    - Requires numpy + albumentations for augmentation; if missing, saves a plain rendered page.
    """

    out_path = Path(out_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(Path(in_pdf_path)))

    try:
        page = doc.load_page(0)
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")
        img = np.array(pil_img)

        # Document-like augmentations: blur/noise, lighting, compression.
        aug = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(7, 15), p=1.0),
                        A.MotionBlur(blur_limit=(7, 15), p=1.0),
                    ],
                    p=0.8,
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
                A.Lambda(image=_bleed_through_image),
                A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.12, p=0.7),
                A.ImageCompression(
                    quality_range=(25, 80),
                    p=0.6,
                ),
            ]
        )

        out = aug(image=img)["image"]
        Image.fromarray(out).convert("RGB").save(str(out_path))
        return str(out_path)
    finally:
        doc.close()


def doc_pipeline(sampled_persona: str, style_map: Optional[Dict[str, Dict[str, float]]]) -> None:

    MODEL = "mistralai/Mistral-Nemo-Instruct-2407"

    run_dir = _make_next_run_dir()
    logger.info("Run directory: %s", str(run_dir))

    topic = generate_topic(sampled_persona, model = MODEL)
    logger.info("Topic: %s", topic)

    text = generate_text(sampled_persona, topic, model = MODEL)
    logger.info("Text: %s", text)

    split_json = split_to_blocks(text)
    out_path = run_dir / "split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)
    logger.info("Split saved to: %s", str(out_path))

    json_with_bbox_sizes = generate_json_with_sizes(split_json, style_map=style_map)
    out_path = run_dir / "json_with_bbox_sizes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_with_bbox_sizes, f, ensure_ascii=False, indent=2)
    logger.info("json_with_bbox_sizes saved to: %s", str(out_path))

    final_layout = generate_layout(data=json_with_bbox_sizes, style_map=style_map)
    out_path = run_dir / "ans.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_layout, f, ensure_ascii=False, indent=2)
    logger.info("Layout saved to: %s", str(out_path))

    pdf_path = render_blocks_json_to_pdf(
        json_path=str(run_dir / "ans.json"),
        out_pdf_path=str(run_dir / "out.pdf"),
        draw_frames=True,
        draw_word_bboxes=True,
        style_map=style_map,
    )
    logger.info("Render saved to: %s", pdf_path)

    # aug_img_path = "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/doc.png"
    # augmented_path = augment_image(
    #     in_pdf_path=pdf_path,
    #     out_image_path=aug_img_path,
    #     dpi=300,
    #     jpeg_quality_range=(35, 95),
    # )
    # logger.info("Augmented page saved to: %s", augmented_path)