import os
from openai import OpenAI


GENERATE_DOCUMENT_TOPIC_PROMPT = """You are an expert in data analysis and have a broad knowledge of different topics.
My persona is: "{persona}"
I want you to generate the topic for {figure_type} that I will be interested in or I may see during my daily life given my persona.

Here are the requirements:
1. The topic is a high-level summary of statistical distribution with some details, e.g., "population growth in the US with a breakdown of the age groups."
2. The topic should be diverse to help me generate varied figures.
3. The topic is conditioned on the figure type. Please ensure the topic you provided can be best visualized in "{figure_type}".
4. The topic must be in Russian, even if the persona is non-Russian."""

def generate_topic(persona: str, model: str, figure_type: str) -> str:

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = GENERATE_DOCUMENT_TOPIC_PROMPT.format(persona=persona, figure_type=figure_type)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY the topic string in English. No quotes, no extra text, no newlines."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=120,
    )

    content = completion.choices[0].message.content
    return content.replace("\n", "").replace("\r", "").strip()