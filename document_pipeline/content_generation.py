import os
import re
from openai import OpenAI


GENERATE_DOCUMENT_DATA_PROMPT = """You are an expert in content creation and have broad knowledge about various topics.
My persona is: "{persona}"
I need materials about "{topic}" that can be used to generate a realistic document.

Here are the requirements:
1. The materials must be directly related to the topic and customized according to the given persona.
2. The materials must be realistic and grounded in real-world context. Use real-world entities, names, places, dates, and organizations where appropriate.
   Do NOT use placeholder names such as xxA, xxB, John Doe, or template markers like [Name], [Date], etc.
3. The materials should cover different relevant aspects of the topic to make the document informative.
4. All materials must be written in Russian, even if the persona is non-Russian.
5. Document MUST contain from 300 to 400 words.
6. Structure requirements:
- Start with a clear TITLE on the first line (1 sentence, no bullet/numbering).
- Do NOT use markdown headings with # symbols. Use plain text headers without any leading symbols.
- Do NOT use asterisks for emphasis or formatting. Use plain text only.
- If there are mulitple paragraphs, include a short header before each of them.
- If lists are helpful for clarity, include bullet list OR numbered list. Do NOT force lists into every document.
- Any list must be contextually appropriate (e.g., steps, criteria, pros/cons, checklist). If not appropriate, use normal paragraphs instead.
- Include concrete numbers where appropriate. Numbers must be plausible and consistent with the topic.
- Keep the tone professional and realistic; avoid generic fluff."""


def _contains_cjk(text: str) -> bool:
    """Detect CJK (Chinese/Japanese/Korean) ideographs in text."""
    if not text:
        return False
    # Covers common CJK ranges (BMP + supplementary planes)
    cjk_re = re.compile(
        r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF\U0002A700-\U0002B73F\U0002B740-\U0002B81F\U0002B820-\U0002CEAF]"
    )
    return cjk_re.search(text) is not None


def generate_text(persona: str, topic: str, model: str, site_url: str = "http://localhost",
    site_title: str = "persona-topic-generator") -> str:
    
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = GENERATE_DOCUMENT_DATA_PROMPT.format(persona=persona, topic=topic)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY the document body as plain text in Russian. Do not add any extra commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=3000,
    )

    text = (completion.choices[0].message.content or "").strip()

    if _contains_cjk(text):
        raise ValueError("Generated text contains CJK (e.g., Chinese) characters; aborting.")

    return text