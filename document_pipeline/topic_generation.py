import os
from openai import OpenAI


GENERATE_DOCUMENT_TOPIC_PROMPT = """You are an expert in document generation and have a broad knowledge of different topics.
My persona is: "{persona}"

I want you to generate a single topic that I will be interested in or that I may encounter in my daily life given my persona.

Here are the requirements:
1. The topic should be a high-level summary of a document’s contents with some realistic details (e.g., purpose, context, or key elements).
2. The topic should be relevant and realistic for the given persona.
3. The topic must be written in Russian, even if the persona is non-Russian."""


def generate_topic(persona: str, model: str) -> str:

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = GENERATE_DOCUMENT_TOPIC_PROMPT.format(persona=persona)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return ONLY the topic string in Russian. No quotes, no extra text, no newlines.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=120,
    )

    content = completion.choices[0].message.content
    return content.replace("\n", "").replace("\r", "").strip()