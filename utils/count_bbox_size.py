from typing import List, Tuple
from reportlab.pdfbase import pdfmetrics


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
    if max_text_width_px <= 0:
        max_text_width_px = 1

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


def measure_bbox_size_for_block(
    content: str,
    *,
    max_width_px: int,
    font_name: str,
    font_size_pt: float,
    leading_pt: float,
    padding_pt: float,
    dpi: int,
    height_safety_factor: float,
) -> Tuple[int, int]:
    """
    Returns (width_px, height_px) that should fit the content inside.
    Width is capped by max_width_px. Height includes padding and safety factor.
    """
    padding_px = int(round(pt_to_px(padding_pt, dpi)))
    leading_px = int(round(pt_to_px(leading_pt, dpi)))

    max_text_width_px = max_width_px - 2 * padding_px
    lines = wrap_text_to_lines(content, max_text_width_px, font_name, font_size_pt, dpi)

    # widest line
    max_line_w = 0.0
    for ln in lines:
        w = string_width_px(ln, font_name, font_size_pt, dpi)
        if w > max_line_w:
            max_line_w = w

    # height estimate: lines * leading + padding*2
    text_h = len(lines) * leading_px
    h = int(round((text_h + 2 * padding_px) * height_safety_factor))

    # width: ensure it fits the longest line + padding, but not exceed max_width_px
    w = int(round(min(max_width_px, max_line_w + 2 * padding_px)))
    w = max(1, w)
    h = max(1, h)
    return w, h