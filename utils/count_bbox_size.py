from typing import List, Tuple
from reportlab.pdfbase import pdfmetrics
import math
import os
import re
import hashlib
from reportlab.pdfgen import canvas as rl_canvas


def pt_to_px(pt: float, dpi: int = 300) -> float:
    return pt * dpi / 72.0


def string_width_px(s: str, font_name: str, font_size_pt: float, dpi: int) -> float:
    w_pt = pdfmetrics.stringWidth(s, font_name, font_size_pt)
    return pt_to_px(w_pt, dpi)


def wrap_text_to_lines(
    text: str,
    max_text_width_px: int,
    font_name: str,
    font_size_pt: float,
    dpi: int,
) -> List[str]:
    """
    Wraps text by spaces, respects explicit '\n' line breaks.
    Hard-breaks very long words by characters.
    """

    def wrap_one_line(line: str) -> List[str]:
        if line == "":
            return [""]

        line = line.replace("\t", "    ")
        words = line.split(" ")
        out: List[str] = []
        cur = ""

        def fits(s: str) -> bool:
            return string_width_px(s, font_name, font_size_pt, dpi) <= max_text_width_px

        for w in words:
            candidate = w if cur == "" else (cur + " " + w)
            if fits(candidate):
                cur = candidate
                continue

            if cur:
                out.append(cur)
                cur = ""

            if fits(w):
                cur = w
            else:
                # hard-break by characters
                chunk = ""
                for ch in w:
                    cand = chunk + ch
                    if fits(cand) or chunk == "":
                        chunk = cand
                    else:
                        out.append(chunk)
                        chunk = ch
                cur = chunk

        if cur:
            out.append(cur)
        return out

    lines: List[str] = []
    for raw in text.split("\n"):
        lines.extend(wrap_one_line(raw))
    return lines


def get_font_vmetrics_pt(font_name: str, font_size_pt: float) -> tuple[float, float, float]:
    """Return (ascent_pt, descent_pt, line_h_pt) using conservative metrics.

    Some fonts underreport ascent/descent via getAscent/getDescent.
    If the font face provides a bbox (llx,lly,urx,ury) in 1/1000 em, we use it
    to widen ascent/descent so we never underestimate line height.
    """
    asc = float(pdfmetrics.getAscent(font_name, font_size_pt))
    desc = float(pdfmetrics.getDescent(font_name, font_size_pt)) 

    try:
        fnt = pdfmetrics.getFont(font_name)
        face = getattr(fnt, "face", None)
        bbox = getattr(face, "bbox", None) if face is not None else None
        if bbox and len(bbox) == 4:
            llx, lly, urx, ury = bbox
            asc_bbox = float(ury) / 1000.0 * float(font_size_pt)
            desc_bbox = float(lly) / 1000.0 * float(font_size_pt)
            asc = max(asc, asc_bbox)
            desc = min(desc, desc_bbox)
    except Exception:
        pass

    line_h = max(1e-6, asc - desc)
    return asc, desc, line_h


def get_font_vmetrics_tight_pt(font_name: str, font_size_pt: float) -> tuple[float, float, float]:
    """Return (ascent_pt, descent_pt, line_h_pt) using tighter metrics.

    Unlike `get_font_vmetrics_pt`, this function does NOT widen ascent/descent using the font bbox,
    because the bbox often includes rare extremes and makes per-word bboxes слишком высокими.

    Preference order:
      1) face.ascent/face.descent (1/1000 em) if available
      2) pdfmetrics.getAscent/getDescent
    """
    asc = float(pdfmetrics.getAscent(font_name, font_size_pt))
    desc = float(pdfmetrics.getDescent(font_name, font_size_pt))

    try:
        fnt = pdfmetrics.getFont(font_name)
        face = getattr(fnt, "face", None)
        a = getattr(face, "ascent", None) if face is not None else None
        d = getattr(face, "descent", None) if face is not None else None
        if a is not None and d is not None:
            asc_face = float(a) / 1000.0 * float(font_size_pt)
            desc_face = float(d) / 1000.0 * float(font_size_pt)
            if math.isfinite(asc_face) and math.isfinite(desc_face) and asc_face > 0:
                asc = asc_face
                desc = desc_face
    except Exception:
        pass

    line_h = max(1e-6, asc - desc)
    return asc, desc, line_h


def measure_bbox_size_for_block(
    content: str,
    *,
    max_width_px: int,
    font_name: str,
    font_size_pt: float,
    leading_pt: float,
    padding_pt: float,
    dpi: int,
) -> Tuple[int, int]:
    """
    Returns (width_px, height_px, lines) that should fit the content inside.
    Width is capped by max_width_px. Height includes padding and safety factor.
    """
    padding_px = int(math.ceil(pt_to_px(float(padding_pt), dpi)))

    max_text_width_px = max_width_px - 2 * padding_px
    lines = wrap_text_to_lines(content, max_text_width_px, font_name, font_size_pt, dpi)
    if not lines:
        lines = [""]

    max_line_w_px = 0.0
    for ln in lines:
        w_px = string_width_px(ln, font_name, font_size_pt, dpi)
        max_line_w_px = max(max_line_w_px, w_px)

    asc_pt, desc_pt, one_line_h_pt = get_font_vmetrics_pt(font_name, font_size_pt)

    n_lines = len(lines)
    text_h_pt = one_line_h_pt + max(0, n_lines - 1) * float(leading_pt)

    scale = float(dpi) / 72.0

    total_h_pt = (text_h_pt + 2.0 * float(padding_pt))

    h = int(math.ceil(total_h_pt * scale))
    w = int(math.ceil(min(float(max_width_px), max_line_w_px + 2 * padding_px)))
    return w, h, lines


def measure_bbox_size_for_one_word(
    text: str,
    *,
    font_name: str,
    font_size_pt: float,
    dpi: int,
) -> tuple[int, int]:
    """Tight bbox for a single token (no padding/leading/wrapping).

    For per-word boxes we prefer *tighter* vertical metrics (no bbox widening), otherwise
    many fonts produce слишком высокий bbox по Y.
    """
    if not isinstance(text, str) or text == "":
        return (1, 1)

    # Width
    w_px = string_width_px(text, font_name, font_size_pt, dpi)

    # Height
    _, _, one_line_h_pt = get_font_vmetrics_tight_pt(font_name, font_size_pt)
    scale = float(dpi) / 72.0
    h_px = max(1, int(math.ceil(float(one_line_h_pt) * scale)))
    return (w_px, h_px)