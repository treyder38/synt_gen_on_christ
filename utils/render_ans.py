import json
import os
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

from .count_bbox_size import pt_to_px


def _px_to_pt(px: float, dpi: float) -> float:
    return px * 72.0 / float(dpi)


def _bbox_px_to_rect_pt(
    bbox_px: List[float],
    page_w_px: float,
    page_h_px: float,
    dpi: float,
) -> Tuple[float, float, float, float]:
    """
    Input bbox_px: [x0, y0, x1, y1] with origin at TOP-LEFT in pixels.
    Return: (x_pt, y_pt, w_pt, h_pt) for ReportLab with origin at BOTTOM-LEFT in points.
    """
    x0, y0, x1, y1 = bbox_px
    w_px = x1 - x0
    h_px = y1 - y0

    x_pt = _px_to_pt(x0, dpi)
    w_pt = _px_to_pt(w_px, dpi)
    h_pt = _px_to_pt(h_px, dpi)

    y_bottom_px = page_h_px - y1
    y_pt = _px_to_pt(y_bottom_px, dpi)

    return x_pt, y_pt, w_pt, h_pt


def render_blocks_json_to_pdf(
    json_path: str,
    out_pdf_path: Optional[str],
    draw_frames: bool = True,
    draw_word_bboxes: bool = False,
    style_map: Optional[Dict[str, Dict[str, float]]] = None,
    picture_path : str | Path | None = None
) -> str:
    """
    Reads layout JSON of the form:
    {
      "page": {"width": 2480, "height": 3508, "dpi": 300},
      "blocks": [{"id": "...", "type": "title|header|paragraph", "bbox": [x0,y0,x1,y1], "content": "..."}]
    }

    Renders text inside each bbox and draws bbox frames on a single PDF page.

    Coordinates:
      - bbox uses TOP-LEFT origin in PIXELS.
      - PDF uses BOTTOM-LEFT origin in POINTS; conversion is handled.

    Returns the output PDF path.
    style_map must contain entries for 'title', 'header', 'paragraph' with keys 'font_name', 'font_size', 'leading'.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    page = data["page"]
    page_w_px = float(page["width"])
    page_h_px = float(page["height"])
    dpi = float(page.get("dpi", 300))

    page_w_pt = _px_to_pt(page_w_px, dpi)
    page_h_pt = _px_to_pt(page_h_px, dpi)

    c = canvas.Canvas(out_pdf_path, pagesize=(page_w_pt, page_h_pt))

    blocks = data.get("blocks", [])
    blocks_by_id = {b["id"]: b for b in blocks}

    order = [b["id"] for b in blocks]

    for bid in order:
        b = blocks_by_id.get(bid)
        if not b:
            continue

        btype = (b.get("type") or "paragraph").strip().lower()

        bbox = b["bbox"]
        content = b.get("content", "")

        x_pt, y_pt, w_pt, h_pt = _bbox_px_to_rect_pt(bbox, page_w_px, page_h_px, dpi)

        # Draw bbox frame
        if draw_frames:
            c.saveState()
            c.setLineWidth(0.25)
            c.rect(x_pt, y_pt, w_pt, h_pt, stroke=1, fill=0)
            c.restoreState()

        # Draw per-word bbox frames (only if present in JSON)
        if draw_word_bboxes:
            words = b.get("words")
            if isinstance(words, list):
                for wd in words:
                    if not isinstance(wd, dict):
                        continue
                    wb = wd.get("bbox")
                    if not isinstance(wb, (list, tuple)) or len(wb) != 4:
                        continue
                    wx_pt, wy_pt, ww_pt, wh_pt = _bbox_px_to_rect_pt(
                        [float(wb[0]), float(wb[1]), float(wb[2]), float(wb[3])],
                        page_w_px,
                        page_h_px,
                        dpi,
                    )
                    c.saveState()
                    c.setLineWidth(0.5)
                    c.rect(wx_pt, wy_pt, ww_pt, wh_pt, stroke=1, fill=0)
                    c.restoreState()

        # Render image for figure blocks
        if btype == "figure":
            img_path = Path(picture_path)
            c.drawImage(
                ImageReader(str(img_path)),
                x_pt,
                y_pt,
                width=w_pt,
                height=h_pt,
                preserveAspectRatio=True,
                anchor='c',
                mask='auto',
            )
            continue

        # Style lookup for text blocks
        if style_map is None:
            raise ValueError("style_map must be provided for text blocks")
        if btype not in style_map:
            btype = "paragraph"
        st = style_map[btype]

        # Prepare text
        font_name = st["font_name"]
        font_size = float(st["font_size"])
        leading = float(st["leading"])

        # Padding (points) must match what was used in measure_bbox_size_for_block.
        padding_pt = float(style_map["padding_pt"])
        padding_px = pt_to_px(padding_pt, dpi)

        c.setFont(font_name, font_size)

        lines = b.get("lines", [""])

        # --- Baseline placement must be consistent with bbox height math ---
        # measure_bbox_size_for_block assumes:
        #   total height = (ascent - descent) + (n_lines-1)*leading (+ padding)
        # and baselines advance by `leading`.
        asc_pt = float(pdfmetrics.getAscent(font_name, font_size))
        desc_pt = float(pdfmetrics.getDescent(font_name, font_size))

        top_y_pt = y_pt + h_pt
        left_x_pt = x_pt + _px_to_pt(padding_px, dpi)

        # Ensure the top of glyphs (baseline + ascent) sits at (top - padding)
        first_baseline_y = top_y_pt - padding_pt - asc_pt

        text = c.beginText()
        text.setFont(font_name, font_size)
        text.setLeading(leading)
        text.setTextOrigin(left_x_pt, first_baseline_y)

        # Manual vertical clipping aligned with ascent/descent + padding.
        # Ensure the bottom of glyphs (baseline + descent) stays above (bottom + padding).
        bottom_limit_y = y_pt + padding_pt
        min_baseline_y = bottom_limit_y - desc_pt  # since desc_pt is negative

        for line in lines:
            if text.getY() < min_baseline_y:
                break
            if line != "":
                text.textLine(line)

        c.drawText(text)

    c.showPage()
    c.save()
    return out_pdf_path