from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple


def generate_layout(
    data: Dict[str, Any],
    *,
    page_w: int = 2480,     # A4 @300dpi
    page_h: int = 3508,
    dpi: int = 300,
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

    if "blocks" not in data or not isinstance(data["blocks"], list):
        raise ValueError("Ожидаю ключ 'blocks' со списком блоков")

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
        laid_blocks.append(bb)

        y_col[col_idx] = y2 + v_gap

    out["blocks"] = laid_blocks
    return out