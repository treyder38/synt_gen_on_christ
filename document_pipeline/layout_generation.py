from copy import deepcopy
import math
from typing import Any, Dict, List, Optional
from reportlab.pdfbase import pdfmetrics

from utils.count_bbox_size import get_font_vmetrics_pt, pt_to_px


def generate_layout(
    data: Dict[str, Any],
    *,
    style_map: Dict[str, Any],
    page_w: int = 2480,     # A4 @300dpi
    page_h: int = 3508
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
    margin = int(style_map.get("margin", 120))
    gutter = int(style_map.get("gutter", 40))
    v_gap = int(style_map.get("v_gap", 24))

    out = deepcopy(data)

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

    def fits(y: int, h: int) -> bool:
        return (y + h) <= bottom_y

    def layout_words(
        b: Dict[str, Any],
        *,
        padding_pt: int,
        type_of_content: str,
        block_bbox: List[int],
    ) -> Optional[List[Dict[str, Any]]]:
        """Compute per-word bbox positions for text blocks.

        Expects b['words'] = [{'content': str, 'bbox_size': [w,h]}, ...]
        Writes/returns a new list where each word also has 'bbox': [x1,y1,x2,y2].
        Note: This is a simple greedy layout (left-to-right, wraps by width).
        """

        words = b.get("words")
        if not isinstance(words, list) or len(words) == 0:
            return None

        st = style_map[type_of_content]
        font_name = st["font_name"]
        font_size_pt = float(st["font_size"])

        dpi_used = int(style_map["dpi"])
        padding_px = pt_to_px(float(padding_pt), dpi_used)

        # Padded inner box for words.
        left = float(block_bbox[0]) + padding_px
        top = float(block_bbox[1]) + padding_px
        right = float(block_bbox[2]) - padding_px
        bottom = float(block_bbox[3]) - padding_px

        # Space width in px for the current font.
        space_w_pt = float(pdfmetrics.stringWidth(" ", font_name, font_size_pt))
        space_px = pt_to_px(space_w_pt, dpi_used)

        # Vertical metrics: keep line height and leading separated (both converted to px).
        # get_font_vmetrics_pt returns (ascent_pt, descent_pt, line_h_pt) in points.
        leading_pt = float(st["leading"])
        _, _, line_h_pt = get_font_vmetrics_pt(font_name, font_size_pt)

        base_line_h_px = pt_to_px(float(line_h_pt), dpi_used)
        leading_px = pt_to_px(float(leading_pt), dpi_used)

        # Track current line height (px) within the current line.
        # Leading is applied only once on newline.
        current_line_h = float(base_line_h_px)

        def _new_line(line_h_px: float) -> None:
            nonlocal cursor_x, cursor_y, current_line_h
            cursor_x = left
            cursor_y = cursor_y + float(leading_px)
            current_line_h = float(base_line_h_px)

        # Cursor starts at padded top-left.
        cursor_x = left
        cursor_y = top

        out_words: List[Dict[str, Any]] = []

        # Track which placed words belong to which logical line (separated by explicit "\n" tokens).
        # Each element stores indices into `out_words`.
        line_word_indices: List[List[int]] = [[]]

        for w in words:
            content = w["content"]
            bbox_size = w["bbox_size"]

            ww = float(bbox_size[0])
            wh = float(bbox_size[1])

            # Update current line height based on word height.
            current_line_h = max(current_line_h, wh)

            # Handle explicit newline token. We assume it's provided as a separate token.
            if content == "\n":
                _new_line(current_line_h)
                # Start a new logical line for overlap post-processing.
                line_word_indices.append([])
                continue

            # Place bbox at current cursor.
            x1 = cursor_x
            y1 = cursor_y
            x2 = x1 + ww
            y2 = y1 + wh

            ww_out = int(bbox_size[0])
            wh_out = int(bbox_size[1])
            placed = {
                "content": content,
                "bbox_size": [ww_out, wh_out],
                "bbox": [int(round(x1)), int(round(y1)), int(round(x1 + ww)), int(round(y1 + wh))],
            }
            # Preserve any extra fields from the input word dict (without overwriting bbox_size/bbox).
            for k, v in w.items():
                if k in ("content", "bbox_size", "bbox"):
                    continue
                placed[k] = v

            out_words.append(placed)

            # Remember which line this word belongs to.
            line_word_indices[-1].append(len(out_words) - 1)

            # Advance cursor to next word (word width + space).
            cursor_x = cursor_x + ww + space_px

        def _fix_vertical_overlaps(
            placed_words: List[Dict[str, Any]],
            lines: List[List[int]],
        ) -> None:
            """If bboxes from adjacent lines overlap in Y, adjust the boundary.

            For each pair of adjacent lines (i, i+1):
            - Let upper_bottom = max(y2) among words in line i
            - Let lower_top   = min(y1) among words in line i+1
            If upper_bottom > lower_top (overlap), set boundary to the harmonic mean: 2*upper_bottom*lower_top/(upper_bottom + lower_top).
            Then:
            - For all words in upper line: y2 := min(y2, boundary)
            - For all words in lower line: y1 := max(y1, boundary)

            This makes the lower boundary of the upper line and the upper boundary of the lower line equal to the harmonic mean.
            """

            for i in range(len(lines) - 1):
                upper = lines[i]
                lower = lines[i + 1]

                # Skip empty logical lines (possible with consecutive '\n').
                if not upper or not lower:
                    continue

                upper_bottom = max(float(placed_words[idx]["bbox"][3]) for idx in upper)
                lower_top = min(float(placed_words[idx]["bbox"][1]) for idx in lower)

                if upper_bottom <= lower_top:
                    continue  # no overlap

                denom = (upper_bottom + lower_top)
                # Use harmonic mean for the shared boundary.
                boundary = (2.0 * upper_bottom * lower_top / denom) if denom != 0.0 else 0.0
                b_int = int(round(boundary))

                # Clamp the upper line bottoms to the boundary.
                for idx in upper:
                    bb = placed_words[idx]["bbox"]
                    if bb[3] > b_int:
                        bb[3] = b_int
                        # Ensure non-negative height.
                        if bb[3] < bb[1]:
                            bb[3] = bb[1]

                # Clamp the lower line tops to the boundary.
                for idx in lower:
                    bb = placed_words[idx]["bbox"]
                    if bb[1] < b_int:
                        bb[1] = b_int
                        # Ensure non-negative height.
                        if bb[1] > bb[3]:
                            bb[1] = bb[3]

        _fix_vertical_overlaps(out_words, line_word_indices)

        return out_words

    for b in out["blocks"]:
        if not isinstance(b, dict):
            raise ValueError("Каждый элемент blocks должен быть объектом (dict)")

        if "content" not in b or not isinstance(b["content"], str):
            raise ValueError(f"У блока {b.get('id')} нет корректного content (ожидаю строку)")

        if "bbox_size" not in b or not isinstance(b["bbox_size"], (list, tuple)) or len(b["bbox_size"]) != 2:
            raise ValueError(f"У блока {b.get('id')} нет корректного bbox_size=[w,h]")

        w0, h0 = int(b["bbox_size"][0]), int(b["bbox_size"][1])
        
        # 1) FULL-WIDTH блок (шире колонки)
        if w0 > col_w:
            if h0 > max_col_h:
                raise ValueError(
                    f"Блок {b.get('id')} слишком высокий для страницы: h={h0} > max_h={max_col_h}. "
                    f"Нужно уменьшить bbox_size или реализовать разбиение блока."
                )

            y1 = max(y_col[0], y_col[1])
            if not fits(y1, h0):
                raise ValueError(
                    f"Блок {b.get('id')} не помещается на страницу: не хватает места по высоте для full-width блока."
                )

            x1 = margin
            y2 = y1 + h0
            x2 = x1 + w0

            bb = deepcopy(b)
            bb["bbox_size"] = [w0, h0]
            bb["bbox"] = [x1, y1, x2, y2]
            bb["font"] = style_map[bb["type"]]["font_name"]

            words_laid = layout_words(
                bb,
                padding_pt=style_map["padding_pt"],
                type_of_content=bb["type"],
                block_bbox=bb["bbox"],
            )
            if words_laid is not None:
                bb["words"] = words_laid

            laid_blocks.append(bb)

            # full-width блок "срезает" обе колонки — продолжаем ниже него
            next_y = y2 + v_gap
            y_col[0] = next_y
            y_col[1] = next_y
            # col_idx оставляем как есть (чтобы заполнение продолжалось привычно)
            
        # 2) Обычный (колоночный) блок
        else:
            if h0 > max_col_h:
                raise ValueError(
                    f"Блок {b.get('id')} слишком высокий для колонки: h={h0} > max_col_h={max_col_h}. "
                    f"Нужно уменьшить bbox_size или реализовать разбиение блока."
                )

            # пробуем текущую колонку
            y_try = y_col[col_idx]
            if not fits(y_try, h0):
                # не влезает — пробуем другую колонку, начиная с её текущего y (наивысшая свободная строка)
                other = 1 - col_idx
                y_try_other = y_col[other]
                if fits(y_try_other, h0):
                    col_idx = other
                    y_try = y_try_other
                else:
                    raise ValueError(
                        f"Блок {b.get('id')} не помещается на страницу: обе колонки заполнены."
                    )

            x1 = col_x[col_idx]
            y1 = y_try
            x2 = x1 + w0
            y2 = y1 + h0

            bb = deepcopy(b)
            bb["bbox_size"] = [w0, h0]
            bb["bbox"] = [x1, y1, x2, y2]
            bb["font"] = style_map[bb["type"]]["font_name"]

            words_laid = layout_words(
                bb,
                padding_pt=style_map["padding_pt"],
                type_of_content=bb["type"],
                block_bbox=bb["bbox"],
            )
            if words_laid is not None:
                bb["words"] = words_laid

            laid_blocks.append(bb)

            y_col[col_idx] = y2 + v_gap

    out["blocks"] = laid_blocks

    # Clip page
    max_x2, max_y2 = 0, 0
    for b in out["blocks"]:
        bbox = b["bbox"]
        x2 = int(round(float(bbox[2])))
        y2 = int(round(float(bbox[3])))
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)

    out["page"] = {
        "width": int(max_x2 + margin),
        "height": int(max_y2 + margin),
        "dpi": int(dpi)
    }
    
    return out