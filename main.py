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
from botocore.config import Config
import traceback


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from pict_data_pipeline.complete_pipe_pic import pic_pipeline 
from document_pipeline.complete_pipe_doc import doc_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

logger = logging.getLogger(__name__)


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


def main() -> None:

    style_map = {
        "dpi": 300,
        "padding_pt": 4.0,
        "height_safety_factor": 1.0,

        "margin": int(random.randint(80, 180)),
        "gutter": int(random.randint(20, 70)),
        "v_gap": int(random.randint(12, 48)),
        "scale_to_column": True,

        "title": {"font_size": float(random.randint(13, 19)), "leading": float(random.randint(13, 15)), "font_name": "Caveat-Bold.ttf"},
        "header": {"font_size": float(random.randint(13, 15)), "leading": float(random.randint(8, 13)), "font_name": "Caveat-SemiBold.ttf"},
        "paragraph": {"font_size": float(random.randint(9, 13)), "leading": float(random.randint(8, 12)), "font_name": "Caveat-Regular.ttf"},
    }

    fonts_dir = "/home/jovyan/people/Glebov/synt_gen_2/ruhw_fonts"
    register_fonts(fonts_dir)

    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default=None,
        help="The types of visualizations to generate. If omitted, generates documents.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="How many samples (documents) to generate.",
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
        help="Optional S3 endpoint URL for S3-compatible storages (e.g., https://obs.ru-moscow-1.hc.sbercloud.ru).",
    )
    args = parser.parse_args()

    personas_path = "/home/jovyan/people/Glebov/synt_gen_2/utils/persona.jsonl"

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
        archive_name = f"batch_{batch_idx:05d}_{ts}.tar.gz"
        archive_path = str(Path("/home/jovyan/people/Glebov/synt_gen_2/document_pipeline/out") / archive_name)

        logger.info(
            "Archiving %d sample dirs (~%.2f GB) -> %s",
            len(batch_dirs),
            batch_bytes / (1024**3),
            archive_path,
        )
        make_tar_gz(archive_path, batch_dirs)

        prefix = args.s3_prefix.strip("/")
        s3_key = f"{prefix}/{archive_name}" if prefix else archive_name
        logger.info("Uploading to s3://%s/%s", args.s3_bucket, s3_key)
        upload_file_to_s3(
            archive_path,
            args.s3_bucket,
            s3_key,
            profile=args.aws_profile,
            region=args.aws_region,
            endpoint_url=args.s3_endpoint,
            max_put_bytes=4_900_000_000,
        )

        # After successful upload, delete local archive and local sample dirs
        os.remove(archive_path)
        for d in batch_dirs:
            safe_rmtree(d)
        logger.info("Batch uploaded and local data removed (dirs + archive).")

        batch_dirs = []
        batch_bytes = 0
        batch_idx += 1

    # Generate samples
    n_total = int(args.n_samples)
    for i in tqdm(range(n_total), total=n_total, desc="Generating", unit="doc"):
        sampled_persona = sample_persona(personas_path)
        #logger.info("Sampled persona: %s", sampled_persona)
        sample_random_fonts_for_style_map(style_map, fonts_dir)

        logger.info(
            "Random fonts chosen: title=%s, header=%s, paragraph=%s",
            style_map["title"]["font_name"],
            style_map["header"]["font_name"],
            style_map["paragraph"]["font_name"],
        )

        try:
            if args.type is None:
                out_dir = doc_pipeline(sampled_persona, style_map)
            else:
                out_dir = pic_pipeline(sampled_persona, args.type, style_map)
        except Exception as e:
            # Log error and continue generating next samples.
            logger.exception("Failed to generate sample %d/%d (type=%s). Error: %s", i + 1, args.n_samples, args.type, e)
            safe_rmtree(out_dir)
            continue

        sz = get_dir_size_bytes(out_dir)
        batch_dirs.append(out_dir)
        batch_bytes += sz

        logger.info(
            "Generated %d/%d: %s (%.2f MB). Current batch: %.2f GB.",
            i + 1,
            args.n_samples,
            out_dir,
            sz / (1024**2),
            batch_bytes / (1024**3),
        )

        if batch_bytes >= target_bytes:
            flush_batch()
        
    # Flush remaining
    flush_batch()


if __name__ == "__main__":
    main()