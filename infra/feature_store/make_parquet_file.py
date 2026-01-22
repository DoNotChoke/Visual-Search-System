import random
from pathlib import Path
import pandas as pd
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def build_s3_uri(bucket: str, prefix: str, class_name: str, filename: str) -> str:
    prefix = prefix.strip("/")
    key = f"{prefix}/{class_name}/{filename}" if prefix else f"{class_name}/{filename}"
    return f"s3://{bucket}/{key}"

def make_parquet_from_folders(
    root_dir: str,
    out_path: str = "data/image_embeddings.parquet",
    bucket: str = "visual-search",
    prefix: str = "sop-base",
    per_folder: int = 100,
    embedding_dim: int = 512,
    seed: int = 42,
):
    random.seed(seed)
    root = Path(root_dir)
    rows = []

    for folder in sorted([p for p in root.iterdir() if p.is_dir()]):
        class_name = folder.name
        imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not imgs:
            continue

        chosen = random.sample(imgs, k=min(per_folder, len(imgs)))
        for img_path in chosen:
            s3_uri = build_s3_uri(bucket, prefix, class_name, img_path.name)
            vec = np.random.rand(embedding_dim).astype("float32").tolist()

            rows.append({
                "img_path": s3_uri,  # entity key (đơn giản: dùng s3 uri)
                "event_timestamp": pd.Timestamp.now(tz="UTC"),
                "vector": vec,
                "s3_uri": s3_uri,
                "category": class_name,
            })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

# Example:
# make_parquet_from_folders(r"D:\datasets\SOP\images")
