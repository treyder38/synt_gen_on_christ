from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Tuple, Optional

from reportlab.pdfbase import pdfmetrics


def generate_layout(
    data: Dict[str, Any],
    *,
    style_map: Dict[str, Any],
    page_w: int = 2480,     # A4 @300dpi
    page_h: int = 3508,
    margin: int = 120,
    gutter: int = 40,
    v_gap: int = 24,
    scale_to_column: bool = True,
) -> Dict[str, Any]:
    """
    Делит страницу A4 на 2 колонки и раскладывает blocks сверху вниз.

    Правила:
    1) Обычные блоки (w <= ширины колонки) кладём в текущую колонку сверху вниз.
       Если в текущей колонке не хватает места по высоте — переходим в другую колонку и
       продолжаем с её текущей "наивысшей свободной строки" (т.е. с её текущего y-курсорa).
       Если и там не влезает — ошибка (всё должно уместиться на одну страницу).

    2) Если блок по ширине НЕ влезает в колонку (w > ширины колонки), то кладём его
       как full-width блок на всю область страницы между полями (между margin слева/справа),
       начиная с y = max(y_left, y_right), чтобы не пересекаться ни с одной колонкой.
       После размещения full-width блока обе колонки продолжаются ниже него.

    Вход:  {"blocks":[{"id":..., "type":..., "content": "...", "bbox_size":[w,h]}, ...]}
    Выход: тот же JSON + page + у каждого блока bbox=[x1,y1,x2,y2]
    """
    dpi = int(style_map.get("dpi", 300))

    out = deepcopy(data)
    out["page"] = {"width": page_w, "height": page_h, "dpi": dpi}

    # Геометрия страницы
    full_w = page_w - 2 * margin
    if full_w <= 0:
        raise ValueError("Слишком большие margin для заданной страницы")

    usable_w = page_w - 2 * margin - gutter
    if usable_w <= 0:
        raise ValueError("Слишком большие margin/gutter для заданной страницы")

    col_w = usable_w // 2
    col_x = [margin, margin + col_w + gutter]

    top_y = margin
    bottom_y = page_h - margin
    max_col_h = bottom_y - top_y
    if max_col_h <= 0:
        raise ValueError("Слишком большие margin для заданной страницы")

    # Независимые y-курсорa для колонок
    y_col = [top_y, top_y]
    col_idx = 0  # текущая колонка для узких блоков

    laid_blocks: List[Dict[str, Any]] = []

    def scale_to_width(w: int, h: int, width_limit: int) -> Tuple[int, int]:
        """Если включен scale_to_column и блок шире width_limit — ужимаем пропорционально."""
        if w <= width_limit:
            return w, h
        if not scale_to_column:
            return w, h
        s = width_limit / float(w)
        new_w = width_limit
        new_h = max(1, int(round(h * s)))
        return new_w, new_h

    def fits(y: int, h: int) -> bool:
        return (y + h) <= bottom_y

    def get_style(b: Dict[str, Any]) -> Dict[str, Any]:
        b_type = str(b.get("type") or "paragraph")
        st = style_map.get(b_type)
        if not isinstance(st, dict):
            st = style_map.get("paragraph", {})
        if not isinstance(st, dict):
            st = {}
        return st

    def pt_to_px(pt: float, dpi_used: int) -> float:
        return float(pt) * (float(dpi_used) / 72.0)

    def layout_words(
        b: Dict[str, Any],
        *,
        block_bbox: List[int],
        block_size_scaled: Tuple[int, int],
        block_size_orig: Tuple[int, int],
    ) -> Optional[List[Dict[str, Any]]]:
        """Compute per-word bbox positions for text blocks.

        Expects b['words'] = [{'content': str, 'bbox_size': [w,h]}, ...]
        Writes/returns a new list where each word also has 'bbox': [x1,y1,x2,y2] and
        'bbox_size' scaled if the block was scaled.

        Note: This is a simple greedy layout (left-to-right, wraps by width).
        """
        words = b.get("words")
        if not isinstance(words, list) or not words:
            return None

        x1b, y1b, x2b, y2b = (int(block_bbox[0]), int(block_bbox[1]), int(block_bbox[2]), int(block_bbox[3]))
        w_scaled, h_scaled = int(block_size_scaled[0]), int(block_size_scaled[1])
        w0, h0 = int(block_size_orig[0]), int(block_size_orig[1])
        if w0 <= 0 or h0 <= 0:
            return None

        s = w_scaled / float(w0)

        st = get_style(b)
        font_name = str(st.get("font_name") or "")
        font_size_pt = float(st.get("font_size") or 0.0)
        leading_pt = float(st.get("leading") or 0.0)

        if not font_name or font_size_pt <= 0:
            raise ValueError(f"Некорректный стиль в style_map для блока type={b.get('type')}: нужен font_name и font_size")

        # Space width from font metrics (pt -> px), then apply block scaling.
        space_pt = float(pdfmetrics.stringWidth(" ", font_name, font_size_pt))
        space_px_f = pt_to_px(space_pt, dpi) * s

        # Line advance: leading is baseline-to-baseline in style_map
        leading_px_f = (pt_to_px(leading_pt, dpi) * s) if leading_pt > 0 else 0.0

        # Font height (ascent-descent) in px, scaled
        asc_pt = float(pdfmetrics.getAscent(font_name, font_size_pt))
        desc_pt = float(pdfmetrics.getDescent(font_name, font_size_pt))
        font_h_pt = max(1e-6, asc_pt - desc_pt)
        font_h_px_f = pt_to_px(font_h_pt, dpi) * s

        out_words: List[Dict[str, Any]] = []

        x = float(x1b)
        y = float(y1b)
        line_h = 0.0
        max_x = float(x2b)  # use bbox right edge
        wrap_tol = 1.0  # px tolerance to avoid premature wrap due to rounding

        for idx, wd in enumerate(words):
            if not isinstance(wd, dict):
                continue

            # Explicit newline token: move to next line and reset cursor.
            # `generate_json.py` emits tokens with content='\n' and bbox_size=[0,0].
            if wd.get("content") == "\n":
                out_wd = dict(wd)
                out_wd["bbox_size"] = [0, 0]
                xi = int(round(x))
                yi = int(round(y))
                out_wd["bbox"] = [xi, yi, xi, yi]
                out_words.append(out_wd)

                if leading_px_f > 0:
                    step = leading_px_f
                else:
                    # If no leading is provided, advance by at least one line of font height.
                    step = max(1.0, font_h_px_f, line_h)

                y = y + step
                x = float(x1b)
                line_h = 0.0
                continue

            ws = wd.get("bbox_size")
            if not isinstance(ws, (list, tuple)) or len(ws) != 2:
                continue
            ww0, wh0 = int(ws[0]), int(ws[1])
            wh0 = int(ws[1])
            if ww0 <= 0 or wh0 <= 0:
                continue

            ww_f = max(1.0, float(ww0) * s)
            wh_f = max(1.0, float(wh0) * s)

            # Wrap to next line if doesn't fit
            if x != float(x1b) and (x + ww_f) > (max_x + wrap_tol):
                # Next line: advance exactly by style_map leading when provided.
                if leading_px_f > 0:
                    step = leading_px_f
                else:
                    step = max(1.0, line_h)

                y = y + step
                x = float(x1b)
                line_h = 0.0

            wx1_f = x
            wy1_f = y
            wx2_f = x + ww_f
            wy2_f = y + wh_f

            wx1 = int(math.floor(wx1_f))
            wy1 = int(math.floor(wy1_f))
            wx2 = int(math.ceil(wx2_f))
            wy2 = int(math.ceil(wy2_f))

            out_wd = dict(wd)
            out_wd["bbox_size"] = [max(1, wx2 - wx1), max(1, wy2 - wy1)]
            out_wd["bbox"] = [wx1, wy1, wx2, wy2]
            out_words.append(out_wd)

            # Advance x by the word width plus a font-accurate space (except before explicit newline / last token)
            next_is_newline = False
            if idx < (len(words) - 1) and isinstance(words[idx + 1], dict):
                nxt = words[idx + 1]
                next_is_newline = (nxt.get("content") == "\n")

            if idx < (len(words) - 1) and not next_is_newline:
                x = wx2_f + space_px_f
            else:
                x = wx2_f

            if wh_f > line_h:
                line_h = wh_f

        return out_words

    for b in out["blocks"]:
        if not isinstance(b, dict):
            raise ValueError("Каждый элемент blocks должен быть объектом (dict)")

        if "content" not in b or not isinstance(b["content"], str):
            raise ValueError(f"У блока {b.get('id')} нет корректного content (ожидаю строку)")

        if "bbox_size" not in b or not isinstance(b["bbox_size"], (list, tuple)) or len(b["bbox_size"]) != 2:
            raise ValueError(f"У блока {b.get('id')} нет корректного bbox_size=[w,h]")

        w0, h0 = int(b["bbox_size"][0]), int(b["bbox_size"][1])
        if w0 <= 0 or h0 <= 0:
            raise ValueError(f"У блока {b.get('id')} bbox_size должен быть > 0")

        # 1) FULL-WIDTH блок (шире колонки)
        if w0 > col_w:
            w, h = scale_to_width(w0, h0, full_w)

            if h > max_col_h:
                raise ValueError(
                    f"Блок {b.get('id')} слишком высокий для страницы: h={h} > max_h={max_col_h}. "
                    f"Нужно уменьшить bbox_size или реализовать разбиение блока."
                )

            y1 = max(y_col[0], y_col[1])
            if not fits(y1, h):
                raise ValueError(
                    f"Блок {b.get('id')} не помещается на страницу: не хватает места по высоте для full-width блока."
                )

            x1 = margin
            y2 = y1 + h
            x2 = x1 + w

            bb = deepcopy(b)
            bb["bbox_size"] = [w, h]
            bb["bbox"] = [x1, y1, x2, y2]

            words_laid = layout_words(
                bb,
                block_bbox=bb["bbox"],
                block_size_scaled=(w, h),
                block_size_orig=(w0, h0),
            )
            if words_laid is not None:
                bb["words"] = words_laid

            laid_blocks.append(bb)

            # full-width блок "срезает" обе колонки — продолжаем ниже него
            next_y = y2 + v_gap
            y_col[0] = next_y
            y_col[1] = next_y
            # col_idx оставляем как есть (чтобы заполнение продолжалось привычно)
            continue

        # 2) Обычный (колоночный) блок
        w, h = scale_to_width(w0, h0, col_w)  # здесь w0 <= col_w, но оставим на будущее

        if h > max_col_h:
            raise ValueError(
                f"Блок {b.get('id')} слишком высокий для колонки: h={h} > max_col_h={max_col_h}. "
                f"Нужно уменьшить bbox_size или реализовать разбиение блока."
            )

        # пробуем текущую колонку
        y_try = y_col[col_idx]
        if not fits(y_try, h):
            # не влезает — пробуем другую колонку, начиная с её текущего y (наивысшая свободная строка)
            other = 1 - col_idx
            y_try_other = y_col[other]
            if fits(y_try_other, h):
                col_idx = other
                y_try = y_try_other
            else:
                raise ValueError(
                    f"Блок {b.get('id')} не помещается на страницу: обе колонки заполнены."
                )

        x1 = col_x[col_idx]
        y1 = y_try
        x2 = x1 + w
        y2 = y1 + h

        bb = deepcopy(b)
        bb["bbox_size"] = [w, h]
        bb["bbox"] = [x1, y1, x2, y2]

        words_laid = layout_words(
            bb,
            block_bbox=bb["bbox"],
            block_size_scaled=(w, h),
            block_size_orig=(w0, h0),
        )
        if words_laid is not None:
            bb["words"] = words_laid

        laid_blocks.append(bb)

        y_col[col_idx] = y2 + v_gap

    out["blocks"] = laid_blocks
    return out