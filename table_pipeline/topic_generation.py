import os
from openai import OpenAI
from functools import lru_cache

GENERATE_DOCUMENT_TOPIC_PROMPT = """You are an expert in data analysis and have a broad knowledge of different topics.
My persona is: "{persona}"
I want you to generate a topics that I will be interested in or I may see during my daily life given my persona.

Here are the requirements:
1. The topic is a high-level summary of tabular data, e.g., "Sales data of a Russian grocery stores in 2021 Q1".
2. The topic should be diverse to help me generate varied tables.
3. The topic is conditioned on the table type.
4. The topics must be in Russian, even if the persona is non-Russian."""


@lru_cache(maxsize=32)
def _get_openai_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )


def generate_topic(persona: str, model: str, base_url: str) -> str:

    client = _get_openai_client(base_url)

    prompt = GENERATE_DOCUMENT_TOPIC_PROMPT.format(persona=persona)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY the topic string in Russian. No quotes, no extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=200,
    )

    content = completion.choices[0].message.content
    return content.replace("\n", "").replace("\r", "").strip()