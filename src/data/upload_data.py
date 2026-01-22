import os
import random
import mimetypes
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from minio import Minio
from urllib3.exceptions import ProtocolError
from http.client import RemoteDisconnected

from vector_store.utils import ensure_bucket

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

MAX_RETRIES = 5
BACKOFF_BASE_SEC = 0.3


def get_content_type(path: Path) -> str:
    ct, _ = mimetypes.guess_type(str(path))
    return ct or "application/octet-stream"


def list_class_folders(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")
    folders = [p for p in root.iterdir() if p.is_dir()]
    folders.sort()
    return folders


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def build_object_key(prefix: str, class_name: str, filename: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{class_name}/{filename}" if prefix else f"{class_name}/{filename}"


def build_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


_thread_local = threading.local()

def get_minio_client_threadlocal() -> Minio:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    return _thread_local.client


def upload_one_file(
    img_path: Path,
    bucket: str,
    object_key: str,
) -> Tuple[str, bool, str]:
    s3_uri = build_s3_uri(bucket, object_key)
    content_type = get_content_type(img_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_minio_client_threadlocal()
            client.fput_object(
                bucket_name=bucket,
                object_name=object_key,
                file_path=str(img_path),
                content_type=content_type,
            )
            return s3_uri, True, ""
        except (RemoteDisconnected, ProtocolError, ConnectionResetError) as e:
            if attempt == MAX_RETRIES:
                return s3_uri, False, f"{type(e).__name__}: {e}"
            time.sleep(BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
        except Exception as e:
            return s3_uri, False, f"{type(e).__name__}: {e}"

    return s3_uri, False, "Unknown error"


def upload_sop_subset_parallel(
    root_dir: str,
    bucket: str,
    prefix: str = "sop-base",
    per_folder: int = 1000,
    seed: int = 42,
    max_workers: int = 16,
    dry_run: bool = False,
) -> None:
    random.seed(seed)

    admin_client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    ensure_bucket(admin_client, bucket)

    tasks: List[Tuple[Path, str]] = []

    for folder in list_class_folders(root_dir):
        class_name = folder.name
        imgs = list_images(folder)
        if not imgs:
            continue

        k = min(len(imgs), per_folder)
        chosen = random.sample(imgs, k)

        for img_path in chosen:
            object_key = build_object_key(prefix, class_name, img_path.name)
            tasks.append((img_path, object_key))

    print(f"Prepared {len(tasks)} files. Uploading with {max_workers} workers...")

    if dry_run:
        for img_path, key in tasks[:20]:
            print("DRY_RUN:", build_s3_uri(bucket, key), "->", img_path)
        print("Dry run only. No upload performed.")
        return

    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(upload_one_file, img_path, bucket, key)
            for (img_path, key) in tasks
        ]
        for fut in as_completed(futures):
            s3_uri, success, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {s3_uri}: {err}")

    print("==== DONE ====")
    print(f"OK={ok}, FAIL={fail}, TOTAL={len(tasks)}")


if __name__ == "__main__":
    upload_sop_subset_parallel(
        root_dir="data/sop/Stanford_Online_Products",
        bucket="visual-search",
        prefix="sop-base",
        per_folder=1000,
        seed=42,
        max_workers=16,
        dry_run=False,
    )
