import logging
from minio import Minio
from io import BytesIO
from typing import Tuple, List

from vector_store.config import Config

logger = logging.getLogger("minio_io")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri (must start with s3://): {s3_uri}")
    s3_uri = s3_uri[5:]  # strip s3://
    if "/" not in s3_uri:
        raise ValueError(f"Invalid s3 uri (missing key): {s3_uri}")
    bucket, key = s3_uri.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def parse_category_from_uri(s3_uri: str) -> str:
    _, key = parse_s3_uri(s3_uri)
    parts = key.split("/")
    # parts[0]=prefix, parts[1]=category
    return parts[1] if len(parts) >= 2 else "unknown"


def get_minio_client(cfg: Config) -> Minio:
    return Minio(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        secure=cfg.minio_secure,
    )

def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info(f"Bucket {bucket} created")


def upload_image_bytes(client: Minio, s3_uri: str, data: bytes, content_type: str = "image/jpeg") -> None:
    bucket, key = parse_s3_uri(s3_uri)
    bio = BytesIO(data)
    client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=bio,
        length=len(data),
        content_type=content_type,
    )

    logger.info("Uploaded image bytes to bucket %s", bucket)


def download_object_types(client: Minio, s3_uri: str) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)

    response = None
    try:
        response = client.get_object(bucket, key)
        data = response.read()
        return data
    finally:
        if response is not None:
            response.close()
            response.release_conn()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_key(key: str) -> bool:
    key_lower = key.lower()
    return any(key_lower.endswith(ext) for ext in IMAGE_EXTS)


def list_image_objects(client: Minio, cfg: Config) -> List[str]:
    keys = []
    for obj in client.list_objects(cfg.bucket, prefix=cfg.prefix.strip("/") + "/", recursive=cfg.recursive):
        if obj.is_dir:
            continue
        if is_image_key(obj.object_name):
            keys.append(obj.object_name)
        if cfg.max_objects and len(keys) >= cfg.max_objects:
            break
    return keys