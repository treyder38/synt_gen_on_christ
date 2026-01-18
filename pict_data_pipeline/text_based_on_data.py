import os
import json
from typing import Any, Dict
from openai import OpenAI


GENERATE_DOCUMENT_TEXT_JSON_PROMPT = """You are an expert writer.
You are given:
- a persona: "{persona}"
- structured data describing the topic: {topic} (referred to as "data")

Here is the data in JSON format:
<data>
{data}
</data>

Your task is to generate a realistic, coherent, single-page document text based strictly on the provided data and adapted to the given persona. 
Do not invent facts that contradict the data; use the data as the factual backbone and expand it into natural language where appropriate.

Here are the requirements:
1. The text must be directly based on the provided data and customized according to the given persona.
2. The text must be realistic and grounded in real-world context. Use real-world entities, names, places, dates, and organizations when they are present in the data.
   Do NOT introduce placeholder names such as xxA, xxB, John Doe, or template markers like [Name], [Date], etc.
3. The text should cover different relevant aspects implied by the data to make the document informative, but concise.
4. The output must be written entirely in Russian, even if the persona is non-Russian."""


def generate_text(persona: str, topic: str, model: str, data: str) -> str:
    
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = GENERATE_DOCUMENT_TEXT_JSON_PROMPT.format(persona=persona, topic=topic, data=data)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY a single valid JSON object with only the 'main_text' field. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    text = (completion.choices[0].message.content or "").strip()
    obj: Dict[str, Any] = json.loads(text)
    return obj["main_text"]