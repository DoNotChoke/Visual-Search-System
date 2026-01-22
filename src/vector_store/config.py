from dataclasses import dataclass

@dataclass
class Config:
    feast_repo: str
    model_path: str

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool

    bucket: str
    prefix: str
    recursive: bool = True

    batch_size: int = 64
    max_objects: int = 0
    num_workers_download: int = 1
    sleep_on_error_sec: float = 0.2

    feature_view_name: str = "image_embeddings"

    # FeatureView
    entity_key_col: str = "img_path"
    vector_col: str = "vector"
    s3_uri_col: str = "s3_uri"
    category_col: str = "category"
    ts_col: str = "event_timestamp"

