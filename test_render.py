# import json
# from pathlib import Path
# from typing import Any, Tuple

# import cv2


# def draw_bboxes_from_json(
#     image_path: str,
#     json_path: str,
#     out_path: str | None = None,
#     *,
#     color: Tuple[int, int, int] = (0, 255, 0),
#     thickness: int = 2,
#     draw_id: bool = False,
#     font_scale: float = 0.6,
# ) -> str:
#     """
#     Load image and JSON with blocks containing 'bbox' = [x1, y1, x2, y2],
#     draw rectangles (and optional block id), and save a copy.

#     Args:
#         image_path: Path to input image.
#         json_path: Path to JSON file with {"blocks":[...]}.
#         out_path: Where to save. If None, saves рядом с исходным как <name>_bboxes.<ext>
#         color: BGR color for rectangles (OpenCV uses BGR).
#         thickness: Rectangle line thickness.
#         draw_id: Whether to draw block id near the bbox.
#         font_scale: Text scale for id.

#     Returns:
#         Path to saved image.
#     """
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise FileNotFoundError(f"Failed to read image: {image_path}")

#     with open(json_path, "r", encoding="utf-8") as f:
#         data: Any = json.load(f)

#     blocks = data.get("blocks", [])
#     if not isinstance(blocks, list):
#         raise ValueError("JSON must contain key 'blocks' as a list")

#     h, w = img.shape[:2]

#     for b in blocks:
#         if not isinstance(b, dict):
#             continue
#         bbox = b.get("bbox", None)
#         if not (isinstance(bbox, list) and len(bbox) == 4):
#             continue

#         x1, y1, x2, y2 = bbox
#         # robust cast
#         try:
#             x1, y1, x2, y2 = map(int, map(round, map(float, (x1, y1, x2, y2))))
#         except Exception:
#             continue

#         # clamp
#         x1 = max(0, min(x1, w - 1))
#         x2 = max(0, min(x2, w - 1))
#         y1 = max(0, min(y1, h - 1))
#         y2 = max(0, min(y2, h - 1))
#         if x2 <= x1 or y2 <= y1:
#             continue

#         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

#         if draw_id:
#             bid = str(b.get("id", ""))
#             if bid:
#                 # background box for readability
#                 (tw, th), baseline = cv2.getTextSize(bid, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
#                 tx, ty = x1, max(0, y1 - 5)
#                 bg_x2 = min(w - 1, tx + tw + 6)
#                 bg_y1 = max(0, ty - th - baseline - 6)
#                 cv2.rectangle(img, (tx, bg_y1), (bg_x2, ty + 2), (0, 0, 0), -1)
#                 cv2.putText(img, bid, (tx + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

#     in_p = Path(image_path)
#     if out_path is None:
#         out_path = str(in_p.with_name(in_p.stem + "_bboxes" + in_p.suffix))

#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     ok = cv2.imwrite(out_path, img)
#     if not ok:
#         raise RuntimeError(f"Failed to write image: {out_path}")

#     return out_path

# if __name__ == "__main__":

#     out = draw_bboxes_from_json(
#         "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/20260125T064316Z_72bde19c5bcc4a0380de8d9f9302a460/doc.png",
#         "/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out/20260125T064316Z_72bde19c5bcc4a0380de8d9f9302a460/ans.json",
#         out_path="/home/jovyan/people/Glebov/synt_gen_2/test_render.png",
#     )
#     print("saved:", out)



from __future__ import annotations

from pathlib import Path
from typing import Iterable

ROOT = Path("/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out")

def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.2f} {units[i]}"

total_bytes = 0
n_files = 0

for f in iter_files(ROOT):
    try:
        total_bytes += f.stat().st_size
        n_files += 1
    except FileNotFoundError:
        # файл мог быть удалён/переименован во время обхода
        continue

avg = (total_bytes / n_files) if n_files else 0.0

print(f"Dir: {ROOT}")
print(f"Files: {n_files}")
print(f"Total: {fmt_bytes(total_bytes)} ({total_bytes} bytes)")
print(f"Avg:   {fmt_bytes(avg)} ({avg:.2f} bytes)")