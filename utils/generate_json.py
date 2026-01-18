import json
from typing import Any, Dict, List

from PIL import Image
from pathlib import Path

from .count_bbox_size import measure_bbox_size_for_block

def generate_json_with_sizes(
    layout_json: str | Dict[str, Any],
    *,
    # per-block max width constraint (px). Adjust if you want different per block type.
    max_width_px: int = 1040,
    dpi: int = 300,
    padding_pt: float = 3.0,
    height_safety_factor: float = 1,
    # typography by block type (same shape as render_text_blocks_json_to_pdf style_map)
    style_map: Dict[str, Dict[str, float | str]] | None = None,
    picture_path : str | Path | None = None
) -> Dict[str, Any]:
    """
    Parses a JSON of format:
    {
      "blocks": [{"id":..., "type":..., "content":...}, ...]
    }
    And returns the same JSON with each block extended by:
      "bbox_size": [width_px, height_px]

    Note: style_map must contain keys 'font_name', 'font_size', 'leading' for each block type.
    """
    if isinstance(layout_json, str):
        obj: Dict[str, Any] = json.loads(layout_json)
    else:
        obj = layout_json

    out_blocks: List[Dict[str, Any]] = []
    for b in obj["blocks"]:
        b_id = b.get("id")
        b_type = b.get("type")
        content = b.get("content", "")

        if b_type != "figure":
            style = style_map.get(str(b_type), style_map["paragraph"])
            font_name = str(style["font_name"])
            font_size_pt = float(style["font_size"])
            leading_pt = float(style["leading"])

            w, h = measure_bbox_size_for_block(
                content,
                max_width_px=max_width_px,
                font_name=font_name,
                font_size_pt=font_size_pt,
                leading_pt=leading_pt,
                padding_pt=padding_pt,
                dpi=dpi,
                height_safety_factor=height_safety_factor,
            )
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
            with Image.open(picture_path) as im:
                w0, h0 = im.size
            scale = min(page_w_px / float(w0), page_h_px / float(h0), 1.0)
            w = max(1, int(round(w0 * scale)))
            h = max(1, int(round(h0 * scale)))

        b2 = dict(b)
        b2["bbox_size"] = [int(w), int(h)]
        out_blocks.append(b2)

    return {
        "blocks": out_blocks
    }