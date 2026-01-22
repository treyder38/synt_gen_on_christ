import json
import os
import random
from typing import Any, Optional
from argparse import ArgumentParser
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging

from pict_data_pipeline.complete_pipe_pic import pic_pipeline 
from document_pipeline.complete_pipe_doc import doc_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

logger = logging.getLogger(__name__)


def register_fonts(style_map: dict[str, Any], fonts_dir: str = "/home/jovyan/people/Glebov/synt_gen_2/ruhw_fonts/") -> None:
    """Register all fonts referenced in style_map.

    Expects per-block styles like:
      style_map["title"]["font_name"] = "Caveat-Bold"
    and will register them from:
      /home/jovyan/people/Glebov/synt_gen_2/fonts/<font_name>.ttf

    Ignores non-dict entries and global numeric keys.
    """
    font_names: set[str] = set()

    for v in style_map.values():
        if isinstance(v, dict) and "font_name" in v:
            fn = str(v.get("font_name", "")).strip()
            if fn:
                font_names.add(fn)

    for font_name in sorted(font_names):
        path = f"{fonts_dir}/{font_name}.ttf"
        try:
            pdfmetrics.registerFont(TTFont(font_name, path))
        except Exception as e:
            logger.warning("Failed to register font '%s' from '%s': %s", font_name, path, e)


def sample_random_fonts_for_style_map(style_map: dict[str, Any], fonts_dir: str, *, seed: Optional[int] = None) -> None:
    """Pick a random font for each per-block style (e.g., title/header/paragraph).

    Expects fonts as .ttf files inside `fonts_dir`. Updates style_map in-place.
    """
    if seed is not None:
        random.seed(seed)

    try:
        files = os.listdir(fonts_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to list fonts_dir={fonts_dir}: {e}")

    font_names = sorted({f[:-4] for f in files if f.lower().endswith(".ttf")})
    if not font_names:
        raise RuntimeError(f"No .ttf fonts found in {fonts_dir}")

    # Choose distinct fonts when possible
    block_keys = [k for k, v in style_map.items() if isinstance(v, dict) and "font_name" in v]
    picks = random.sample(font_names, k=min(len(block_keys), len(font_names)))

    for i, key in enumerate(block_keys):
        style_map[key]["font_name"] = picks[i % len(picks)]


def sample_persona(path: str, seed: Optional[int] = None) -> str:
    """
    Samples a persona string from a .jsonl file.
    Assumes every line is: {"persona": "..."}.
    """
    if seed is not None:
        random.seed(seed)

    personas: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj: Any = json.loads(line)
            personas.append(obj["persona"].strip())

    return random.choice(personas)


if __name__ == "__main__":

    style_map = {
        "dpi": 300,
        "padding_pt": 3.0,
        "height_safety_factor": 1.0,
        "title": {"font_size": 18.0, "leading": 18.0, "font_name": "Caveat-Bold"},
        "header": {"font_size": 12.0, "leading": 10.0, "font_name": "Caveat-SemiBold"},
        "paragraph": {"font_size": 15.0, "leading": 12.0, "font_name": "Caveat-Regular"},
    }

    fonts_dir = "/home/jovyan/people/Glebov/synt_gen_2/ruhw_fonts"
    sample_random_fonts_for_style_map(style_map, fonts_dir)
    register_fonts(style_map, fonts_dir=fonts_dir)

    logger.info("Random fonts chosen: title=%s, header=%s, paragraph=%s",
                style_map["title"]["font_name"], style_map["header"]["font_name"], style_map["paragraph"]["font_name"])

    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default=None,
        help="The types of visualizations to generate.",
    )
    args = parser.parse_args()

    personas_path = "/home/jovyan/people/Glebov/synt_gen_2/utils/persona.jsonl"
    sampled_persona = sample_persona(personas_path)
    logger.info("Sampled persona: %s", sampled_persona)

    if args.type is None:
        doc_pipeline(sampled_persona, style_map)
    else:
        pic_pipeline(sampled_persona, args.type, style_map)