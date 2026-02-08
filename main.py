import json
import os
import random
import logging
import tarfile
import time
import shutil
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional
from argparse import ArgumentParser
from tqdm import tqdm
import boto3
import uuid
import sys
from botocore.config import Config
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


logger = logging.getLogger(__name__)
# Silence very noisy dependency loggers
for _name in ("openai", "httpx", "httpcore", "urllib3", "e2b", "e2b.sandbox", "e2b.api", "e2b.client", "e2b.http", "e2b_core"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def register_fonts(fonts_dir: str) -> None:
    """Register every font file in `fonts_dir`.

    Registers all readable .ttf/.otf files found directly inside `fonts_dir`.
    The font is registered under its filename (including extension), e.g. "Caveat-Bold.ttf".
    """
    try:
        files = os.listdir(fonts_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to list fonts_dir={fonts_dir}: {e}")

    font_files: list[str] = []
    for f in files:
        lf = f.lower()
        if not (lf.endswith(".ttf") or lf.endswith(".otf")):
            continue
        path = os.path.join(fonts_dir, f)
        if os.path.isfile(path) and os.access(path, os.R_OK):
            font_files.append(f)

    if not font_files:
        raise RuntimeError(
            f"No readable .ttf/.otf fonts found in {fonts_dir}. "
            "Check that the directory exists and that the font files are present and readable."
        )

    for font_name in sorted(set(font_files)):
        path = os.path.join(fonts_dir, font_name)
        try:
            pdfmetrics.registerFont(TTFont(font_name, path))
        except Exception as e:
            logger.warning("Failed to register font '%s' from '%s': %s", font_name, path, e)


def sample_random_fonts_for_style_map(style_map: dict[str, Any], fonts_dir: str, *, seed: Optional[int] = None) -> None:
    """Pick a random font for each per-block style (e.g., title/header/paragraph).

    Expects fonts as .ttf files inside `fonts_dir`. Updates style_map in-place.
    """
    if seed is not None:
        random.seed(seed)

    try:
        files = os.listdir(fonts_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to list fonts_dir={fonts_dir}: {e}")

    ttf_paths: list[str] = []
    for f in files:
        if not f.lower().endswith(".ttf"):
            continue
        p = os.path.join(fonts_dir, f)
        if os.path.isfile(p) and os.access(p, os.R_OK):
            ttf_paths.append(f)

    font_names = sorted(set(ttf_paths))
    if not font_names:
        raise RuntimeError(
            f"No readable .ttf fonts found in {fonts_dir}. "
            f"Check that the directory exists and that the .ttf files are present and readable."
        )

    block_keys = [k for k, v in style_map.items() if isinstance(v, dict) and "font_name" in v]
    picks = random.sample(font_names, k=min(len(block_keys), len(font_names)))

    for i, key in enumerate(block_keys):
        style_map[key]["font_name"] = picks[i % len(picks)]


def sample_persona(path: str, seed: Optional[int] = None) -> str:
    """
    Samples a persona string from a .jsonl file.
    Assumes every line is: {"persona": "..."}.
    """
    if seed is not None:
        random.seed(seed)

    personas: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj: Any = json.loads(line)
            personas.append(obj["persona"].strip())

    return random.choice(personas)


def get_dir_size_bytes(path: str | os.PathLike) -> int:
    """Recursively compute directory size in bytes."""
    total = 0
    p = Path(path)
    for root, _, files in os.walk(p):
        for fn in files:
            fp = Path(root) / fn
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                # A file might disappear between walk and stat; ignore.
                continue
    return total


def make_tar_gz(archive_path: str, src_dirs: list[str]) -> str:
    """Create a .tar.gz archive that contains each directory under its basename."""
    ap = Path(archive_path)
    ap.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ap, "w:gz") as tar:
        for d in src_dirs:
            dp = Path(d)
            if not dp.exists():
                continue
            tar.add(str(dp), arcname=dp.name)
    return str(ap)


@lru_cache(maxsize=16)
def get_s3_client(profile: Optional[str], region: Optional[str], endpoint_url: Optional[str]):
    """Create (once) and reuse an S3 client for the given (profile, region, endpoint_url)."""
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.Session(region_name=region)

    # Some S3-compatible providers (OBS/MinIO/SeaweedFS/GCS S3-interop, etc.) may store
    # aws-chunked + trailing checksum data *as-is*, making objects unreadable.
    # Force boto/botocore to only calculate checksums when explicitly required.
    cfg = Config(
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        s3={
            "addressing_style": "virtual",
            # Disables SigV4 streaming (avoids Content-Encoding: aws-chunked in many cases).
            "payload_signing_enabled": False,
        },
    )

    # Extra safety: allow env vars to override behavior consistently across boto3 usage.
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
    return session.client("s3", endpoint_url=endpoint_url, config=cfg)


def upload_file_to_s3(
    local_path: str,
    bucket: str,
    key: str,
    *,
    profile: Optional[str] = None,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    max_put_bytes: int = 4_900_000_000,
) -> None:
    """Upload a local file to S3 using a single PUT (put_object).

    Variant A:
      - Avoid multipart uploads entirely (OBS can store chunked streams incorrectly).
      - Enforce object size < ~5GB.
    """
    client = get_s3_client(profile, region, endpoint_url)

    lp = Path(local_path)
    size = lp.stat().st_size
    if size > max_put_bytes:
        raise RuntimeError(
            f"Archive is too large for single PUT: {size} bytes (> {max_put_bytes}). "
            f"Reduce --batch_gb (e.g., 4.5) so archives stay under 5GB."
        )

    with open(lp, "rb") as f:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ContentLength=size,
        )


def safe_rmtree(path: str) -> None:
    """Remove directory tree if it exists."""
    p = Path(path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


@dataclass
class GenTask:
    idx: int
    pipeline: str
    seed: int


def build_style_map(rng: random.Random) -> dict[str, Any]:
    """Build a randomized style_map for one sample."""
    return {
        "dpi": 300,
        "padding_pt": 1.0,

        "margin": int(rng.randint(80, 180)),
        "gutter": int(rng.randint(20, 70)),
        "v_gap": int(rng.randint(12, 37)),

        "title": {
            "font_size": float(rng.randint(13, 19)),
            "leading": float(rng.randint(13, 15)),
            "font_name": "Caveat-Bold.ttf",
        },
        "header": {
            "font_size": float(rng.randint(13, 15)),
            "leading": float(rng.randint(8, 13)),
            "font_name": "Caveat-SemiBold.ttf",
        },
        "paragraph": {
            "font_size": float(rng.randint(11, 13)),
            "leading": float(rng.randint(8, 12)),
            "font_name": "Caveat-Regular.ttf",
        },
    }


def _worker_loop(
    gpu_id: int,
    task_q: "mp.Queue[Optional[GenTask]]",
    result_q: "mp.Queue[dict[str, Any]]",
    fonts_dir: str,
    personas_path: str,
    vllm_host: str,
    vllm_base_port: int,
    samples_dir: str,
) -> None:
    """One worker pinned to a single GPU via CUDA_VISIBLE_DEVICES."""
    # IMPORTANT: must be set before importing torch/diffusers/etc.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    vllm_port = int(vllm_base_port) + int(gpu_id)
    vllm_base_url = f"http://{vllm_host}:{vllm_port}"
    samples_root = Path(samples_dir)

    # Import pipelines only inside the worker after CUDA pinning.
    from pict_data_pipeline.complete_pipe_pic import pic_pipeline
    from document_pipeline.complete_pipe_doc import doc_pipeline
    from table_pipeline.complete_pipe_table import table_pipeline

    from e2b_code_interpreter import Sandbox
    from e2b.exceptions import RateLimitException
    sbx = None
    sbx_created_at = 0.0

    # E2B tuning: keep retries bounded to avoid multi-minute stalls.
    SBX_CREATE_RETRIES = 3
    SBX_CREATE_BACKOFF_START_S = 2.0
    SBX_CREATE_BACKOFF_MAX_S = 20.0

    PIC_RETRY_SLEEP_START_S = 2.0
    PIC_RETRY_SLEEP_MAX_S = 15.0

    def _close_sbx() -> None:
        nonlocal sbx
        if sbx is None:
            return

        import threading

        _local_sbx = sbx
        sbx = None

        def _do_close() -> None:
            _local_sbx.kill()

        t = threading.Thread(target=_do_close, daemon=True)
        t.start()
        t.join(timeout=5.0)
        if t.is_alive():
            logger.warning("[gpu=%s] E2B: sandbox close/kill timed out; continuing", gpu_id)

    def _ensure_sbx() -> None:
        """Ensure a live sandbox exists and periodically recreate it.

        For Hobby, we must set sandbox lifetime explicitly; default can be ~5 min.
        We set lifetime to 1 hour and proactively recreate ~55 min.
        Also, Sandbox.create() can hit 429 if too many concurrent sandboxes exist; use backoff.
        """
        nonlocal sbx, sbx_created_at
        now = time.time()

        # Recreate after ~55 minutes to stay below 1-hour session limits.
        if sbx is not None and (now - sbx_created_at) <= 55 * 60:
            return

        _close_sbx()

        logger.info("[gpu=%s] E2B: creating sandbox...", gpu_id)
        t_create0 = time.monotonic()
        backoff = SBX_CREATE_BACKOFF_START_S
        last_err: Exception | None = None
        for attempt in range(SBX_CREATE_RETRIES):
            try:
                # IMPORTANT: set sandbox lifetime at create time (otherwise it may expire quickly).
                t0 = time.monotonic()
                sbx = Sandbox.create()
                dt = time.monotonic() - t0
                logger.info("[gpu=%s] E2B: sandbox created in %.2fs", gpu_id, dt)
                sbx.set_timeout(60 * 60)
                sbx_created_at = time.time()
                return
            except RateLimitException as e:
                last_err = e
                logger.warning("[gpu=%s] E2B: sandbox create failed (attempt %d/%d): %s", gpu_id, attempt + 1, SBX_CREATE_RETRIES, e)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, SBX_CREATE_BACKOFF_MAX_S)
            except Exception as e:
                last_err = e
                logger.warning("[gpu=%s] E2B: sandbox create failed (attempt %d/%d): %s", gpu_id, attempt + 1, SBX_CREATE_RETRIES, e)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, SBX_CREATE_BACKOFF_MAX_S)

        logger.error("[gpu=%s] E2B: failed to create sandbox after %.2fs", gpu_id, time.monotonic() - t_create0)
        raise RuntimeError(f"Failed to create E2B sandbox after retries: {last_err}")

    # Each process must register fonts in its own ReportLab registry.
    register_fonts(fonts_dir)

    while True:
        try:
            task = task_q.get(timeout=1.0)
        except Empty:
            continue

        if task is None:
            _close_sbx()
            return

        out_dir: Optional[str] = None
        try:
            rng = random.Random(task.seed)
            style_map = build_style_map(rng)

            sample_random_fonts_for_style_map(style_map, fonts_dir, seed=task.seed)
            sampled_persona = sample_persona(personas_path, seed=task.seed)

            if task.pipeline == "doc":
                out_dir = doc_pipeline(sampled_persona, style_map, out_path = samples_root, base_url=f"{vllm_base_url}/v1")

            elif task.pipeline == "pic":
                t_sbx0 = time.monotonic()
                _ensure_sbx()
                logger.info("[gpu=%s idx=%s] E2B: ensure sandbox OK in %.2fs", gpu_id, task.idx, time.monotonic() - t_sbx0)
                _figure_types = [
                    # Core 2D plots
                    "line plot",
                    "multi-line plot",
                    "step plot",
                    "stem plot",
                    "scatter plot",
                    "bubble chart",
                    "bar chart",
                    "grouped bar chart",
                    "stacked bar chart",
                    "horizontal bar chart",
                    "stacked horizontal bar chart",
                    "histogram",
                    "stacked histogram",
                    "density plot",
                    "area chart",
                    "stacked area chart",
                    "pie chart",
                    "donut chart",
                    "box plot",
                    "violin plot",
                    "strip plot",
                    "swarm plot",
                    "error bar plot",
                    "bar chart with error bars",
                    "scatter plot with error bars",

                    # Heatmaps / grids
                    "heatmap",
                    "annotated heatmap",
                    "correlation heatmap",

                    # Contours / fields
                    "contour plot",
                    "filled contour plot",
                    "quiver plot",
                    "streamplot",
                    "hexbin plot",

                    # Time-series / ranges
                    "time series line plot",
                    "time series area chart",
                    "fill between plot",
                ]
                figure_type = rng.choice(_figure_types)
                try:
                    t_pic0 = time.monotonic()
                    logger.info("[gpu=%s idx=%s] pic_pipeline: start (type=%s)", gpu_id, task.idx, figure_type)
                    out_dir = pic_pipeline(
                        sampled_persona,
                        figure_type,
                        style_map,
                        out_path = samples_root,
                        base_url=f"{vllm_base_url}/v1",
                        sbx=sbx,
                    )
                    logger.info("[gpu=%s idx=%s] pic_pipeline: done in %.2fs", gpu_id, task.idx, time.monotonic() - t_pic0)
                except Exception as e:
                    # Keep retries bounded to avoid multi-minute stalls.
                    msg = str(e)
                    low = msg.lower()

                    # Generated-code errors: recreating the sandbox will not help.
                    if "executionerror" in low:
                        raise

                    is_rate_limit = ("rate limit" in low) or ("ratelimit" in low)
                    is_not_found = ("not found" in low) or ("notfound" in low)
                    is_timeoutish = ("timeout" in low) or ("timed out" in low)

                    logger.warning(
                        "[gpu=%s idx=%s] pic_pipeline: error after %.2fs: %s",
                        gpu_id,
                        task.idx,
                        time.monotonic() - t_pic0 if 't_pic0' in locals() else float('nan'),
                        msg,
                    )

                    # 1) Rate limit: wait a bit and retry ONCE without recreating sandbox.
                    if is_rate_limit:
                        sleep_s = min(PIC_RETRY_SLEEP_START_S * 2.0, PIC_RETRY_SLEEP_MAX_S)
                        logger.warning("[gpu=%s idx=%s] E2B: rate limit; sleeping %.1fs then retry once", gpu_id, task.idx, sleep_s)
                        time.sleep(sleep_s)
                        t_pic1 = time.monotonic()
                        out_dir = pic_pipeline(
                            sampled_persona,
                            figure_type,
                            style_map,
                            out_path = samples_root,
                            base_url=f"{vllm_base_url}/v1",
                            sbx=sbx,
                        )
                        logger.info("[gpu=%s idx=%s] pic_pipeline: retry done in %.2fs", gpu_id, task.idx, time.monotonic() - t_pic1)
                        
                    # 2) Sandbox likely dead (404/not found): recreate and retry ONCE.
                    elif is_not_found:
                        logger.warning("[gpu=%s idx=%s] E2B: sandbox likely dead; recreating and retry once", gpu_id, task.idx)
                        _close_sbx()
                        t_sbx1 = time.monotonic()
                        _ensure_sbx()
                        logger.info("[gpu=%s idx=%s] E2B: recreated sandbox in %.2fs", gpu_id, task.idx, time.monotonic() - t_sbx1)
                        t_pic1 = time.monotonic()
                        out_dir = pic_pipeline(
                            sampled_persona,
                            figure_type,
                            style_map,
                            out_path = samples_root,
                            base_url=f"{vllm_base_url}/v1",
                            sbx=sbx,
                        )
                        logger.info("[gpu=%s idx=%s] pic_pipeline: retry after recreate done in %.2fs", gpu_id, task.idx, time.monotonic() - t_pic1)

                    # 3) Timeout/connection issues: retry once without recreation; if still failing, fail fast.
                    elif is_timeoutish:
                        logger.warning("[gpu=%s idx=%s] E2B: timeout/connection; retry once without recreate", gpu_id, task.idx)
                        t_pic1 = time.monotonic()
                        out_dir = pic_pipeline(
                            sampled_persona,
                            figure_type,
                            style_map,
                            out_path = samples_root,
                            base_url=f"{vllm_base_url}/v1",
                            sbx=sbx,
                        )
                        logger.info("[gpu=%s idx=%s] pic_pipeline: retry done", gpu_id, task.idx, time.monotonic() - t_pic1)

                    else:
                        raise

            elif task.pipeline == "table":
                t_sbx0 = time.monotonic()
                _ensure_sbx()
                logger.info("[gpu=%s idx=%s] E2B: ensure sandbox OK in %.2fs", gpu_id, task.idx, time.monotonic() - t_sbx0)
                try:
                    t_pic0 = time.monotonic()
                    logger.info("[gpu=%s idx=%s] table_pipeline: start", gpu_id, task.idx)
                    out_dir = table_pipeline(
                        sampled_persona,
                        style_map,
                        out_path = samples_root,
                        base_url=f"{vllm_base_url}/v1",
                        sbx=sbx,
                    )
                    logger.info("[gpu=%s idx=%s] table_pipeline: done in %.2fs", gpu_id, task.idx, time.monotonic() - t_pic0)
                except Exception as e:
                    # Keep retries bounded to avoid multi-minute stalls.
                    msg = str(e)
                    low = msg.lower()

                    if "executionerror" in low:
                        raise

                    is_rate_limit = ("rate limit" in low) or ("ratelimit" in low)
                    is_not_found = ("not found" in low) or ("notfound" in low)
                    is_timeoutish = ("timeout" in low) or ("timed out" in low)

                    logger.warning(
                        "[gpu=%s idx=%s] pic_pipeline: error after %.2fs: %s",
                        gpu_id,
                        task.idx,
                        time.monotonic() - t_pic0 if 't_pic0' in locals() else float('nan'),
                        msg,
                    )

                    if is_rate_limit:
                        sleep_s = min(PIC_RETRY_SLEEP_START_S * 2.0, PIC_RETRY_SLEEP_MAX_S)
                        logger.warning("[gpu=%s idx=%s] E2B: rate limit; sleeping %.1fs then retry once", gpu_id, task.idx, sleep_s)
                        time.sleep(sleep_s)
                        t_pic1 = time.monotonic()
                        out_dir = table_pipeline(
                            sampled_persona,
                            style_map,
                            out_path = samples_root,
                            base_url=f"{vllm_base_url}/v1",
                            sbx=sbx,
                        )
                        logger.info("[gpu=%s idx=%s] table_pipeline: retry done in %.2fs", gpu_id, task.idx, time.monotonic() - t_pic1)
                        
                    elif is_not_found:
                        logger.warning("[gpu=%s idx=%s] E2B: sandbox likely dead; recreating and retry once", gpu_id, task.idx)
                        _close_sbx()
                        t_sbx1 = time.monotonic()
                        _ensure_sbx()
                        logger.info("[gpu=%s idx=%s] E2B: recreated sandbox in %.2fs", gpu_id, task.idx, time.monotonic() - t_sbx1)
                        t_pic1 = time.monotonic()
                        out_dir = table_pipeline(
                            sampled_persona,
                            style_map,
                            out_path = samples_root,
                            base_url=f"{vllm_base_url}/v1",
                            sbx=sbx,
                        )
                        logger.info("[gpu=%s idx=%s] table_pipeline: retry after recreate done in %.2fs", gpu_id, task.idx, time.monotonic() - t_pic1)

                    elif is_timeoutish:
                        logger.warning("[gpu=%s idx=%s] E2B: timeout/connection; retry once without recreate", gpu_id, task.idx)
                        t_pic1 = time.monotonic()
                        out_dir = table_pipeline(
                            sampled_persona,
                            style_map,
                            out_path = samples_root,
                            base_url=f"{vllm_base_url}/v1",
                            sbx=sbx,
                        )
                        logger.info("[gpu=%s idx=%s] table_pipeline: retry done in %.2fs", gpu_id, task.idx, time.monotonic() - t_pic1)

                    else:
                        raise
            

            sz = get_dir_size_bytes(out_dir)
            result_q.put({
                "ok": True,
                "idx": task.idx,
                "vllm_base_url": vllm_base_url,
                "vllm_port": vllm_port,
                "out_dir": out_dir,
                "size_bytes": sz,
                "style_fonts": {
                    "title": style_map["title"]["font_name"],
                    "header": style_map["header"]["font_name"],
                    "paragraph": style_map["paragraph"]["font_name"],
                },
            })
        except Exception as e:
            # Clean partial outputs if any.
            if out_dir:
                safe_rmtree(out_dir)
            result_q.put({
                "ok": False,
                "idx": task.idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


def main() -> None:

    fonts_dir = "/home/jovyan/people/ulitin/gigavision_data/ocrsynth/py/imagegen/hw_diffuz/synt_pipe_with_giga/ruhw_fonts"
    personas_path = "/home/jovyan/people/Glebov/synt_gen_2/utils/persona.jsonl"

    parser = ArgumentParser()
    parser.add_argument(
        "--pipeline",
        type=str,
        default="doc",
        choices=["doc", "pic", "table"],
        help="The type of content to generate",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Root output directory where generated sample folders will be placed and where archives will be created.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="How many samples (documents) to generate.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=7,
        help="How many GPUs/workers to use for parallel generation.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=29328,
        help="Base seed used to derive per-sample seeds (base_seed + idx).",
    )
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="127.0.0.1",
        help="Host where per-GPU vLLM instances are listening.",
    )
    parser.add_argument(
        "--vllm_base_port",
        type=int,
        default=8000,
        help="Base port for per-GPU vLLM instances. Worker on GPU i uses base_port + i.",
    )
    parser.add_argument(
        "--batch_gb",
        type=float,
        default=4.5,
        help="Approx target size (GB) per local batch before archiving/upload.",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket to upload archives to. If not set, archives are not uploaded.",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="synt_docs",
        help="S3 key prefix (folder) for uploaded archives.",
    )
    parser.add_argument(
        "--aws_profile",
        type=str,
        default=None,
        help="Optional AWS profile name to use for S3 upload.",
    )
    parser.add_argument(
        "--aws_region",
        type=str,
        default=None,
        help="Optional AWS region for the S3 client.",
    )
    parser.add_argument(
        "--s3_endpoint",
        type=str,
        default=None,
        help="Optional S3 endpoint URL for S3-compatible storages.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    samples_root = out_root / "samples"
    archives_root = out_root / "archives"
    samples_root.mkdir(parents=True, exist_ok=True)
    archives_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "S3 upload: enabled=%s bucket=%r prefix=%r endpoint=%r profile=%r region=%r out_root=%r",
        bool(args.s3_bucket),
        args.s3_bucket,
        args.s3_prefix,
        args.s3_endpoint,
        args.aws_profile,
        args.aws_region,
        args.out_root,
    )

    # Batching state
    target_bytes = int(args.batch_gb * 1024 * 1024 * 1024)
    batch_dirs: list[str] = []
    batch_bytes = 0
    batch_idx = 0

    def flush_batch() -> None:
        nonlocal batch_dirs, batch_bytes, batch_idx
        if not batch_dirs:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_name = f"batch_{batch_idx:05d}_{ts}_{uuid.uuid4().hex}.tar.gz"
        archive_path = str(archives_root / archive_name)

        logger.info(
            "Archiving %d sample dirs (~%.2f GB) -> %s",
            len(batch_dirs),
            batch_bytes / (1024**3),
            archive_path,
        )
        make_tar_gz(archive_path, batch_dirs)

        # Log actual archive size (can differ from folder size).
        archive_size = Path(archive_path).stat().st_size
        logger.info(
            "Created archive %s (%.2f GB)",
            archive_path,
            (archive_size / (1024**3)) if archive_size >= 0 else float('nan'),
        )

        prefix = args.s3_prefix.strip("/")
        s3_key = f"{prefix}/{archive_name}" if prefix else archive_name
        if not args.s3_bucket:
            # Upload disabled: keep archive locally, but remove sample dirs.
            for d in batch_dirs:
                safe_rmtree(d)
            logger.info("Upload disabled. Kept local archive: %s", archive_path)
        else:
            logger.info("Uploading to s3://%s/%s", args.s3_bucket, s3_key)
            try:
                upload_file_to_s3(
                    archive_path,
                    args.s3_bucket,
                    s3_key,
                    profile=args.aws_profile,
                    region=args.aws_region,
                    endpoint_url=args.s3_endpoint,
                    max_put_bytes=4_900_000_000,
                )
                # After successful upload, delete local archive
                os.remove(archive_path)
                logger.info("Upload OK. Removed local archive: %s", archive_path)

                # Only after a successful upload remove sample dirs.
                for d in batch_dirs:
                    safe_rmtree(d)
                logger.info("Batch uploaded and local sample dirs removed.")
            except Exception as e:
                logger.error(
                    "S3 upload FAILED for %s -> s3://%s/%s. Error: %s",
                    archive_path,
                    args.s3_bucket,
                    s3_key,
                    e,
                )
                logger.error("Traceback:\n%s", traceback.format_exc())
                for d in batch_dirs:
                    safe_rmtree(d)
                logger.info("Local sample dirs removed after the error.")
                return

        batch_dirs = []
        batch_bytes = 0
        batch_idx += 1


    # Generate samples in parallel (multi-GPU)
    n_total = int(args.n_samples)
    n_workers = args.num_gpus

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()

    workers: list[mp.Process] = []
    for gpu_id in range(n_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(gpu_id, task_q, result_q, fonts_dir, personas_path, args.vllm_host, args.vllm_base_port, str(samples_root)),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Enqueue tasks
    for i in range(n_total):
        seed = int(args.base_seed) + i
        task_q.put(GenTask(idx=i, pipeline=args.pipeline, seed=seed))

    # Tell workers to stop
    for _ in range(n_workers):
        task_q.put(None)

    # Collect results
    done = 0
    pbar = tqdm(total=n_total, desc="Generating", unit="doc", dynamic_ncols=True, disable=False, file=sys.stdout)
    try:
        while done < n_total:
            try:
                res = result_q.get(timeout=60)
            except Empty:
                alive = [p.is_alive() for p in workers]
                logger.warning("No results for 60s. Workers alive=%s", alive)
                if not any(alive):
                    raise RuntimeError("All workers have exited but generation is incomplete")
                continue
                
            done += 1
            pbar.update(1)

            if not res.get("ok", False):
                logger.error(
                    "Failed to generate sample %d/%d (pic=%s). Error: %s\n%s",
                    res.get("idx", -1) + 1,
                    n_total,
                    args.pipeline,
                    res.get("error"),
                    res.get("traceback"),
                )
                continue

            out_dir = str(res["out_dir"])
            sz = int(res["size_bytes"])
            fonts = res.get("style_fonts", {})

            sz = get_dir_size_bytes(out_dir)

            # logger.info(
            #     "Generated %d/%d: %s (%.2f MB). Fonts: title=%s header=%s paragraph=%s vLLM=%s",
            #     res.get("idx", 0) + 1,
            #     n_total,
            #     out_dir,
            #     sz / (1024**2),
            #     fonts.get("title"),
            #     fonts.get("header"),
            #     fonts.get("paragraph"),
            #     res.get("vllm_base_url"),
            # )

            batch_dirs.append(out_dir)
            batch_bytes += sz

            # logger.info("Current batch: %.2f GB (target %.2f GB).", batch_bytes / (1024**3), target_bytes / (1024**3))

            if batch_bytes >= target_bytes:
                flush_batch()
    finally:
        pbar.close()
        # Ensure workers exit
        for p in workers:
            if p.is_alive():
                p.join(timeout=1.0)

    # Flush remaining
    flush_batch()


if __name__ == "__main__":
    main()