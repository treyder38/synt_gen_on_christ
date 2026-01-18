import json
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

def register_fonts(font_ttf_paths: dict[str, str]) -> None:
    for name, path in font_ttf_paths.items():
        pdfmetrics.registerFont(TTFont(name, path))

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

    default_styles = {
        "title": {"font_size": 18.0, "leading": 22.0, "font_name": "Caveat-Bold"},
        "header": {"font_size": 12.0, "leading": 12.0, "font_name": "Caveat-SemiBold"},
        "paragraph": {"font_size": 15.0, "leading": 13.0, "font_name": "Caveat-Regular"},
    }
    
    font_ttf_paths = {
        "Caveat-Bold": "/home/jovyan/people/Glebov/synt_gen_2/fonts/Caveat-Bold.ttf",
        "Caveat-SemiBold": "/home/jovyan/people/Glebov/synt_gen_2/fonts/Caveat-SemiBold.ttf",
        "Caveat-Regular": "/home/jovyan/people/Glebov/synt_gen_2/fonts/Caveat-Regular.ttf"
    }
    register_fonts(font_ttf_paths)

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
    logger.info("Sampled persona %s", sampled_persona)

    if args.type is None:
        doc_pipeline(sampled_persona, default_styles)
    else:
        pic_pipeline(sampled_persona, args.type, default_styles)