import json
from typing import Any, Dict, List, Optional

from PIL import Image
from pathlib import Path

from .count_bbox_size import measure_bbox_size_for_block


def generate_json_with_sizes(
    layout_json: str | Dict[str, Any],
    style_map: Optional[Dict[str, Any]] = None,
    picture_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Input JSON:
    {
      "blocks": [{"id":..., "type":..., "content":...}, ...]
    }

    Output JSON:
    {
      "blocks": [... same blocks ... with "bbox_size": [w_px, h_px] ...]
    }
    """
    if isinstance(layout_json, str):
        obj: Dict[str, Any] = json.loads(layout_json)
    else:
        obj = layout_json

    style_map = style_map or {}
    out_blocks: List[Dict[str, Any]] = []

    for b in obj["blocks"]:
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
            max_width_px = int(style_map.get("max_width_px"))
            padding_pt = float(style_map.get("padding_pt"))
            height_safety_factor = float(style_map.get("height_safety_factor"))

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

            if picture_path is None:
                raise ValueError("picture_path must be provided when a block has type='figure'")

            with Image.open(picture_path) as im:
                w0, h0 = im.size

            scale = min(page_w_px / float(w0), page_h_px / float(h0), 1.0)
            w = max(1, int(round(w0 * scale)))
            h = max(1, int(round(h0 * scale)))

        b2 = dict(b)
        b2["bbox_size"] = [int(w), int(h)]
        out_blocks.append(b2)

    return {"blocks": out_blocks}