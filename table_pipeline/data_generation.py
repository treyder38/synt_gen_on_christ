import os
import json
from typing import Any, Dict
from openai import OpenAI
from functools import lru_cache


GENERATE_DOCUMENT_DATA_JSON_PROMPT = """You are an expert in data analysis and have broad knowledge about various topics.
My persona is: "{persona}"
I need some data about "{topic}", which can be used to generate a table. 

Here are the requirements:
1. The data structure must be suitable for the table.
2. The contents are related to the topic and customized according to my persona.
3. The data should be realistic, and the contents should be named using Russian real-world entities. Do not use placeholder names like xxA, xxB, etc.
4. The data should be diverse and contain multiple data points to ensure the table is informative.
5. Do not provide too much data. Just necessary data points to satisfy the topic and table. Also, do not generate long text inside cells: keep values concise (short phrases), avoid long sentences/paragraphs.
6. All data must be in Russian, even if the persona is non-Russian."""


@lru_cache(maxsize=32)
def _get_openai_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )


def generate_data(persona: str, topic: str, model: str, base_url: str) -> str:
    
    client = _get_openai_client(base_url)

    prompt = GENERATE_DOCUMENT_DATA_JSON_PROMPT.format(persona=persona, topic=topic)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY a single valid JSON object with only 'data' field. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    text = (completion.choices[0].message.content or "").strip()
    obj: Dict[str, Any] = json.loads(text)
    return obj["data"]