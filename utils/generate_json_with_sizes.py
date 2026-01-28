import json
import random
from typing import Any, Dict, List, Optional
from PIL import Image
from pathlib import Path
import re

from .count_bbox_size import measure_bbox_size_for_block, measure_bbox_size_for_one_word


def split_lines_to_tokens(
    lines: List[str],
) -> List[Dict[str, str]]:
    out = []
    for i, line in enumerate(lines):
        if line:
            for t in re.findall(r"\n|[^\s]+", line):
                if t != "":
                    out.append({"content": t})

        # Every boundary between returned lines becomes a newline token.
        if i < len(lines) - 1:
            out.append({"content": "\n"})

    return out


def generate_json_with_sizes(
    layout_json: str | Dict[str, Any],
    style_map: Dict[str, Any] = None,
    picture_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Input JSON:
    {
      "blocks": [{"id":..., "type":..., "content":...}, ...]
    }

    Output JSON:
    {
      "blocks": [... same blocks ... with "bbox_size": [w_px, h_px] ...] & bboxes for every word in the block
    }

    Note: max_width_px is chosen per-block based on a random split boundary:
      - blocks before split get max_width_px=2080
      - blocks after split get max_width_px=1040
    """
    if isinstance(layout_json, str):
        obj: Dict[str, Any] = json.loads(layout_json)
    else:
        obj = layout_json

    style_map = style_map
    lines = []
    out_blocks = []

    n_blocks = len(obj.get("blocks", []))
    if n_blocks <= 1:
        split_idx = n_blocks
    else:
        split_idx = random.randint(0, n_blocks)

    for i, b in enumerate(obj["blocks"]):
        b_type = b.get("type")
        content = b.get("content", "")

        if b_type != "figure":
            # Per-type typography
            style = style_map.get(str(b_type))
            if not isinstance(style, dict):
                style = style_map.get("paragraph", {})
            if not isinstance(style, dict):
                raise ValueError("style_map must contain a dict for 'paragraph' with font_name/font_size/leading")

            dpi = int(style_map.get("dpi"))
            padding_pt = float(style_map.get("padding_pt"))

            font_name = str(style["font_name"])
            font_size_pt = float(style["font_size"])
            leading_pt = float(style["leading"])

            if b_type == "title":
                max_width_px = random.choice([2080, 1040])
            else:
                if i < split_idx:
                    max_width_px = 2080
                else:
                    max_width_px = 1040

            w, h, lines = measure_bbox_size_for_block(
                content,
                max_width_px=max_width_px,
                font_name=font_name,
                font_size_pt=font_size_pt,
                leading_pt=leading_pt,
                padding_pt=padding_pt,
                dpi=dpi,
            )

            #### TOKENIZE AND COUNT SIZES ####
            words_out: List[Dict[str, Any]] = []
            words_in = split_lines_to_tokens(lines)
            if not isinstance(words_in, list):
                raise ValueError("Input JSON must contain a list field 'words' for each non-figure block")

            for wd in words_in:
                if not isinstance(wd, dict):
                    continue
                tok = wd.get("content")
                if not isinstance(tok, str):
                    continue

                out_wd = dict(wd)
                if tok == "\n":
                    out_wd["bbox_size"] = [0, 0]
                    words_out.append(out_wd)
                    continue

                ww, wh = measure_bbox_size_for_one_word(
                    tok,
                    font_name=font_name,
                    font_size_pt=font_size_pt,
                    dpi=dpi,
                )
                out_wd["bbox_size"] = [int(ww), int(wh)]
                words_out.append(out_wd)
            
        else:
            # Target page size for normalizing figure sizes (px)
            page_w_px = 2480
            page_h_px = 3508
            if isinstance(obj, dict) and isinstance(obj.get("page"), dict):
                try:
                    page_w_px = int(obj["page"].get("width", page_w_px))
                    page_h_px = int(obj["page"].get("height", page_h_px))
                except Exception:
                    pass

            # Choose figure max width similar to text blocks
            if i < split_idx:
                max_width_px = 2080
            else:
                max_width_px = 1040

            if picture_path is None:
                raise ValueError("picture_path must be provided when a block has type='figure'")

            with Image.open(picture_path) as im:
                w0, h0 = im.size

            # Force figure width to exactly max_width_px (no constraint on height here)
            scale = max_width_px / float(w0)
            w = int(max_width_px)
            h = max(1, int(round(h0 * scale)))

        b2 = dict(b)
        b2["bbox_size"] = [int(w), int(h)]

        if b_type != "figure":
            b2["words"] = words_out

        if lines != []:
            b2["lines"] = lines

        out_blocks.append(b2)

    return {"blocks": out_blocks}