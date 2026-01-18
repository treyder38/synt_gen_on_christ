import os
import json
from typing import Any, Dict
from openai import OpenAI


GENERATE_DOCUMENT_DATA_JSON_PROMPT = """You are an expert in data analysis and have broad knowledge about various topics.
My persona is: "{persona}"
I need some data about "{topic}", which can be used to generate a {figure_type}. 

Here are the requirements:
1. The data structure must be suitable for the {figure_type}.
2. The contents are related to the topic and customized according to my persona.
3. The data should be realistic, and the contents should be named using real-world entities. Do not use placeholder names like xxA, xxB, etc.
4. The data should be diverse and contain multiple data points to ensure the chart is informative.
5. Do not provide too much data. Just necessary data points to satisfy the topic and figure type.
6. All data must be in Russian, even if the persona is non-Russian."""


def generate_data(persona: str, topic: str, model: str, figure_type: str) -> str:
    
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = GENERATE_DOCUMENT_DATA_JSON_PROMPT.format(persona=persona, topic=topic, figure_type=figure_type)

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