import random
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from minio import Minio

from vector_store.utils import get_minio_client, ensure_bucket, upload_image_bytes

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def list_class_folders(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    folders = [p for p in root.iterdir() if p.is_dir()]
    folders.sort()
    return folders

def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def build_object_key(prefix: str, class_name: str, filename: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{class_name}/{filename}" if prefix else f"{class_name}/{filename}"


def build_s3_uri(bucket: str, prefix: str, class_name: str, filename: str) -> str:
    prefix = prefix.strip("/")
    if prefix:
        key = f"{prefix}/{class_name}/{filename}"
    else:
        key = f"{class_name}/{filename}"
    return f"s3://{bucket}/{key}"


def upload_one(img_path: Path, bucket: str, prefix: str, class_name: str) -> Tuple[str, bool, str]:
    s3_uri = build_s3_uri(bucket, prefix, class_name, img_path.name)
    try:
        data = img_path.read_bytes()
        content_type = "image/jpeg"
        upload_image_bytes(s3_uri, data=data, content_type=content_type)
        return s3_uri, True, ""
    except Exception as e:
        return s3_uri, False, str(e)


def upload_sop_subset_parallel(
        root_dir: str,
        bucket: str,
        prefix: str = "sop-base",
        per_folder: int = 1000,
        seed: int = 42,
        max_workers: int = 16,
        dry_run: bool = False,
):
    random.seed(seed)
    client = get_minio_client()
    ensure_bucket(client, bucket)

    tasks: List[Tuple[str, Path]] = []

    folders = list_class_folders(root_dir)
    for folder in folders:
        class_name = folder.name
        imgs = list_images(folder)
        if not imgs:
            continue

        k = min(len(imgs), per_folder)
        chosen = random.sample(imgs, k)

        for img_path in chosen:
            tasks.append((img_path, bucket, prefix, class_name))

    print(f"Prepared {len(tasks)} files. Uploading with {max_workers} workers...")

    ok = 0
    fail = 0

    if dry_run:
        for img_path, bucket, prefix, class_name in tasks[:20]:
            print("DRY_RUN:", build_s3_uri(bucket, prefix, class_name, img_path.name))
        print("Dry run only. No upload performed.")
        return
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(upload_one, img_path, bucket, prefix, class_name)
            for (img_path, bucket, prefix, class_name) in tasks
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

if __name__ == '__main__':
    upload_sop_subset_parallel(
        root_dir="data/sop/Stanford_Online_Products",
        bucket="visual-search",
        prefix="sop-base",
        per_folder=1000,
        seed=42,
        max_workers=16,
        dry_run=False,
    )