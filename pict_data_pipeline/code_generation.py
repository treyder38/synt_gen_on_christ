import os
import re
import io
import base64
from openai import OpenAI
from functools import lru_cache
from e2b_code_interpreter import Sandbox

_B64_RE = re.compile(r"^BYTES_B64:([A-Za-z0-9+/=]+)\s*$")

GENERATE_CHART_CODE_MATPLOTLIB_PROMPT = """You are an expert Python data analyst who writes clean, executable `matplotlib` code.

My persona is: "{persona}"
I have data about {topic} to visualize as a {figure_type}.

I will provide the data as a **JSON file**.  
Your code must **extract all data from this JSON and load it into a single variable named `df` of type `pd.DataFrame`**.  
**All extracted data must be explicitly assigned to `df` in the code.**  
This `df` variable will then be used throughout the rest of the code.

Here is the JSON data:
<data>
{data}
</data>

Your task:
- Output **only executable Python code**.
- The code must generate the requested chart using `matplotlib`.

Requirements:
1. Data extraction:
   - Parse the provided JSON data.
   - Create a pandas DataFrame named **`df`**.
   - Ensure **all extracted data is fully contained in `df`**.
   - The creation of `df` must be clearly visible in the code.

2. Plot function:
   - Define a function named `generate_plot()`.
   - The function must use the already-created `df`.
   - Do not pass any arguments to the function.
   - The function must create the plot and **must not call itself**.

3. Plotting rules:
   - Use `matplotlib` only.
   - Select appropriate figure size, fonts, markers, and axis scales for clarity.
   - Avoid overlapping text and ensure readability.
   - All text shown on the chart MUST be in Russian (Cyrillic). This includes the title, axis labels, legend labels, tick labels, annotations, and any other on-figure text. 

4. Output handling:
   - Save the figure to a `BytesIO` buffer.
   - Return the plot as a `BytesIO` object.
   - Do **not** close the `BytesIO` buffer.
   - Use `bbox_inches='tight'` or `plt.tight_layout()`.
   - Do not display the plot (`plt.show()` is forbidden).

5. Code-only output:
   - Import all required libraries.
   - Do not include explanations, comments outside the code, or any extra text.
   - Do not include example usage.
   - The response must consist of a **single Python code block only**, starting with ```python and ending with ```.
"""


@lru_cache(maxsize=32)
def _get_openai_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )


def generate_code(persona: str, topic: str, model: str, data: str, figure_type: str, base_url: str) -> str:
    
    client = _get_openai_client(base_url)

    prompt = GENERATE_CHART_CODE_MATPLOTLIB_PROMPT.format(persona=persona, topic=topic, data=data, figure_type=figure_type)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY a single Python code block (```python ... ```). No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=3000,
    )

    text = completion.choices[0].message.content
    return text


def extract_b64(line: str) -> str:
    s = line.strip()

    if (len(s) >= 2) and (s[0] == s[-1]) and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    m = _B64_RE.match(s)
    if not m:
        raise ValueError("Строка не соответствует формату BYTES_B64:<base64>")

    return m.group(1)


def extract_python_code(maybe_fenced: str) -> str:
    """Extract python code from a fenced markdown block if present."""
    s = (maybe_fenced or "").strip()
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else s)


def save_generated_image(
    code_from_model: str,
    timeout_s: float = 120.0,
    sbx: Sandbox | None = None,
) -> io.BytesIO:
    """Executes model-generated code in an E2B sandbox and returns decoded image bytes.

    Returns:
        io.BytesIO: buffer positioned at start (seek(0)) containing the decoded bytes.
    """

    # Ensure we always emit BYTES_B64 to stdout in a single execution, even if the model forgot.
    code_to_run = f"""
import io
import base64
{extract_python_code(code_from_model)}
buf = generate_plot()
print("BYTES_B64:" + base64.b64encode(buf.getvalue()).decode("ascii"))
"""

    if sbx is None:
        with Sandbox.create() as sbx:
            exec_res = sbx.run_code(code_to_run, timeout=timeout_s)
    else:
        exec_res = sbx.run_code(code_to_run, timeout=timeout_s)

    if getattr(exec_res, "error", None):
        raise RuntimeError(f"E2B execution error: {exec_res.error}")
        
    b64 = extract_b64(exec_res.logs.stdout[0])
    raw = base64.b64decode(b64, validate=False)

    out_buf = io.BytesIO(raw)
    out_buf.seek(0)
    return out_buf
