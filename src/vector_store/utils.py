import logging
from minio import Minio
from io import BytesIO
from typing import Tuple

logger = logging.getLogger("minio_io")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    s3_uri = s3_uri.replace("s3://", "")
    if "/" not in s3_uri:
        raise ValueError(f"Invalid s3 uri (missing key): {s3_uri}")
    bucket, key = s3_uri.split("/", 1)
    return bucket, key


def get_minio_client() -> Minio:
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info(f"Bucket {bucket} created")


def upload_image_bytes(s3_uri: str, data: bytes, content_type: str = "image/jpeg") -> None:
    bucket, key = parse_s3_uri(s3_uri)
    client = get_minio_client()
    ensure_bucket(client, bucket)
    bio = BytesIO(data)
    client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=bio,
        length=len(data),
        content_type=content_type,
    )

    logger.info("Uploaded image bytes to bucket %s", bucket)


def download_object_types(s3_uri: str) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)
    client = get_minio_client()

    response = None
    try:
        response = client.get_object(bucket, key)
        data = response.read()
        return data
    finally:
        if response is not None:
            response.close()
            response.release_conn()
