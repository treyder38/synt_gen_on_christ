import re
from typing import Any, Dict, List


def split_tokens_preserve_newlines(text: str) -> List[str]:
    """Split into tokens, preserving explicit line breaks as '\n' tokens.

    - Collapses Windows newlines to '\n'
    - Returns a list of tokens where '\n' appears as its own token
    - Other whitespace is treated as a separator
    """
    if not isinstance(text, str) or not text:
        return []

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep '\n' as separate tokens; everything else split by whitespace
    # Example: "a b\n c" -> ["a","b","\n","c"]
    return re.findall(r"\n|[^\s]+", text)


def split_to_blocks(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return:
    {
      "blocks": [
        {"id":"b1","type":"title|header|paragraph","content":"..."},
        ...
      ]
    }
    Title is always taken from the FIRST paragraph (first block separated by blank lines).
    """

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def strip_md_heading(s: str) -> str:
        return re.sub(r"^\s*#{1,6}\s*", "", s).strip()

    def is_header(line: str) -> bool:
        s = strip_md_heading(norm(line))
        if not s:
            return False
        # avoid list items
        if re.match(r"^([-*•]|\d+[.)])\s+", s):
            return False
        words = s.split()
        if s.endswith(":") and len(words) <= 20:
            return True
        if len(words) <= 12 and not re.search(r"[.!?…]\s*$", s):
            return True
        return False

    # normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return {"blocks": []}

    blocks: List[Dict[str, Any]] = []
    bid = 1

    def make_words(s: str) -> List[Dict[str, str]]:
        return [{"content": tok} for tok in split_tokens_preserve_newlines(s)]

    # split text into paragraph-ish chunks by blank lines
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]

    # Title is always the first paragraph
    title_chunk = raw_blocks[0]
    title_text = strip_md_heading(norm(title_chunk))
    blocks.append({"id": f"b{bid}", "type": "title", "content": title_text, "words": make_words(title_text)})
    bid += 1

    # Process remaining chunks
    for chunk in raw_blocks[1:]:
        chunk_lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not chunk_lines:
            continue

        # peel off consecutive headers at the top of the chunk
        i = 0
        while i < len(chunk_lines) and is_header(chunk_lines[i]):
            header_text = strip_md_heading(norm(chunk_lines[i]))
            blocks.append({"id": f"b{bid}", "type": "header", "content": header_text, "words": make_words(header_text)})
            bid += 1
            i += 1

        rest = "\n".join(chunk_lines[i:]).strip()
        if rest:
            blocks.append({"id": f"b{bid}", "type": "paragraph", "content": rest, "words": make_words(rest)})
            bid += 1

    return {"blocks": blocks}
