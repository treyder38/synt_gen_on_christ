import os
import re
import base64
from pathlib import Path
from openai import OpenAI
from e2b_code_interpreter import Sandbox


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
   - Return the plot as a `PIL.Image` object.
   - Do **not** close the `BytesIO` buffer.
   - Use `bbox_inches='tight'` or `plt.tight_layout()`.
   - Do not display the plot (`plt.show()` is forbidden).

5. Code-only output:
   - Import all required libraries.
   - Do not include explanations, comments outside the code, or any extra text.
   - Do not include example usage.
   - The response must consist of a **single Python code block only**, starting with ```python and ending with ```.

Output **only** the code block."""


def generate_code(persona: str, topic: str, model: str, data: str, figure_type: str) -> str:
    
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

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

    text = (completion.choices[0].message.content or "").strip()
    return text


def _extract_python_code(maybe_fenced: str) -> str:
    """Extract python code from a fenced markdown block if present."""
    s = (maybe_fenced or "").strip()
    # Match ```python ... ``` or ``` ... ```
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else s)


def save_generated_image(
    code_from_model: str,
    out_path: str,
    *,
    remote_path: str = "/home/user/out.png",
    image_format: str = "PNG",
    timeout_s: float = 120.0,
) -> str:
    """Executes model-generated code in an E2B Code Interpreter sandbox and saves the image locally.

    Requirements:
    - `pip install e2b-code-interpreter`
    - `E2B_API_KEY` env var set

    The generated code is expected to define `generate_plot()` that returns a PIL.Image.
    """
    code = _extract_python_code(code_from_model)

    with Sandbox.create() as sbx:
        # 1) define everything (df + generate_plot)
        exec1 = sbx.run_code(code, timeout=timeout_s)
        if getattr(exec1, "error", None):
            raise RuntimeError(f"E2B execution error while defining code: {exec1.error}")

        # 2) call generate_plot and save to a known path
        save_snippet = f"""
from pathlib import Path
img = generate_plot()
Path('{remote_path}').parent.mkdir(parents=True, exist_ok=True)
img.save('{remote_path}', format='{image_format}')
print('SAVED:{remote_path}')
""".strip()

        exec2 = sbx.run_code(save_snippet, timeout=timeout_s)
        if getattr(exec2, "error", None):
            raise RuntimeError(f"E2B execution error while saving image: {exec2.error}")

        # 3) download file bytes from sandbox (IMPORTANT: avoid UTF-8 round-tripping binary)
        files_api = getattr(sbx, "files", None)
        if files_api is None:
            raise RuntimeError("E2B sandbox does not expose a files API (sbx.files)")

        content = None

        # Prefer bytes mode if available (newer SDKs)
        if hasattr(files_api, "read"):
            try:
                content = files_api.read(remote_path, format="bytes")
            except TypeError:
                # Older SDK: no format kwarg
                content = files_api.read(remote_path)

            # Some SDKs may support stream reads
            if not isinstance(content, (bytes, bytearray)):
                try:
                    stream = files_api.read(remote_path, format="stream")
                    content = b"".join(stream)
                except TypeError:
                    pass

        if content is None and hasattr(files_api, "read_file"):
            content = files_api.read_file(remote_path)

        if content is None and hasattr(files_api, "download"):
            content = files_api.download(remote_path)

        if content is None:
            raise RuntimeError("Could not find a supported method to read a file from the E2B sandbox")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Normalize content into raw bytes.
    # Depending on SDK version, `content` can be bytes/bytearray, base64 string, or a small wrapper.
    if isinstance(content, dict):
        content = content.get("content", content.get("data", content))

    if hasattr(content, "content"):
        content = getattr(content, "content")
    if hasattr(content, "data") and not isinstance(content, (bytes, bytearray)):
        content = getattr(content, "data")

    if isinstance(content, bytearray):
        content = bytes(content)

    if isinstance(content, str):
        s = content.strip()
        # Heuristic: treat as base64 if it matches base64 charset and length.
        looks_b64 = (
            len(s) > 32
            and len(s) % 4 == 0
            and re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", s) is not None
        )
        if s.startswith("iVBOR") or s.startswith("/9j/") or looks_b64:
            try:
                content = base64.b64decode(s, validate=False)
            except Exception:
                content = s.encode("utf-8")
        else:
            content = s.encode("utf-8")

    if not isinstance(content, (bytes, bytearray)):
        raise RuntimeError(f"E2B file read returned unsupported type: {type(content)}")

    out.write_bytes(content)

    # Validate output to catch corruption early
    suf = out.suffix.lower()
    if suf == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(
            "Saved .png does not look like a PNG file (wrong magic bytes). "
            "This usually means the sandbox file download returned text/base64 instead of raw bytes."
        )
    if suf in (".jpg", ".jpeg") and not content.startswith(b"\xff\xd8\xff"):
        raise RuntimeError(
            "Saved .jpg does not look like a JPEG file (wrong magic bytes). "
            "This usually means the sandbox file download returned text/base64 instead of raw bytes."
        )

    return str(out.resolve())
