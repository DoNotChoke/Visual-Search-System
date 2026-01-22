import time

import pandas as pd
from feast import FeatureStore

from model.embedding_model import embed, get_model

from PIL import Image

from vector_store.config import Config
from vector_store.utils import download_object_types, get_minio_client, list_image_objects, parse_category_from_uri

from io import BytesIO

from typing import List, Tuple

client = None

def embedding_and_store(model, store: FeatureStore, cfg: Config):
    global client
    if client is None:
        client = get_minio_client(cfg)
    keys = list_image_objects(client, cfg)
    print(f"Found {len(keys)} image objects under s3://{cfg.bucket}/{cfg.prefix}/")

    ok = 0
    fail = 0

    batch_imgs: List[Image.Image] = []
    batch_meta: List[Tuple[str, str, str]] = []
    for i, key in enumerate(keys, start = 1):
        s3_uri = f"s3://{cfg.bucket}/{key}"
        category = parse_category_from_uri(s3_uri)

        try:
            data = download_object_types(client, s3_uri)
            img = Image.open(BytesIO(data)).convert("RGB")

            batch_imgs.append(img)
            batch_meta.append((s3_uri, key, category))
        except Exception as e:
            fail += 1
            print(f"[DOWNLOAD FAIL] {s3_uri}: {type(e).__name__}: {e}")
            time.sleep(cfg.sleep_on_error_sec)
            continue

        if len(batch_imgs) >= cfg.batch_size:
            try:
                vectors = embed(model, batch_imgs)
                now = pd.Timestamp.now(tz="UTC")

                df = pd.DataFrame({
                    cfg.entity_key_col: [m[0] for m in batch_meta],
                    cfg.ts_col: [now] * len(batch_meta),
                    cfg.vector_col: vectors,
                    cfg.s3_uri_col: [m[0] for m in batch_meta],
                    cfg.category_col: [m[2] for m in batch_meta],
                })

                store.write_to_online_store(feature_view_name=cfg.feature_view_name, df=df)
                ok += len(batch_imgs)
                print(f"[OK] Upserted batch: {len(batch_imgs)} (total_ok={ok}, total_fail={fail})")

            except Exception as e:
                fail += len(batch_imgs)
                print(f"[WRITE FAIL] batch_size={len(batch_imgs)}: {type(e).__name__}: {e}")
                time.sleep(cfg.sleep_on_error_sec)
            finally:
                batch_imgs.clear()
                batch_meta.clear()
    if batch_imgs:
        try:
            vectors = embed(model, batch_imgs)
            now = pd.Timestamp.now(tz="UTC")

            df = pd.DataFrame({
                cfg.entity_key_col: [m[0] for m in batch_meta],
                cfg.ts_col: [now] * len(batch_meta),
                cfg.vector_col: vectors,
                cfg.s3_uri_col: [m[0] for m in batch_meta],
                cfg.category_col: [m[2] for m in batch_meta],
            })
            store.write_to_online_store(feature_view_name=cfg.feature_view_name, df=df)
            ok += len(batch_imgs)
            print(f"[OK] Upserted final batch: {len(batch_imgs)} (total_ok={ok}, total_fail={fail})")
        except Exception as e:
            fail += len(batch_imgs)
            print(f"[WRITE FAIL] final batch_size={len(batch_imgs)}: {type(e).__name__}: {e}")

    print("==== DONE ====")
    print(f"OK={ok}, FAIL={fail}, TOTAL={len(keys)}")


if __name__ == '__main__':
    config = Config(
        feast_repo="../../infra/plugins/config",
        model_path="resnet152_sop_embedding_best.pt",

        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,

        bucket="visual-search",
        prefix="sop-base"
    )
    client = get_minio_client(config)
    store = FeatureStore(config.feast_repo)
    model = get_model(config.model_path)
    print(embedding_and_store(model, store, config))