import pandas as pd
import torch
from feast import FeatureStore

from model.embedding_model import ResNetEmbedding

from PIL import Image

from model.transform import transform
from vector_store.utils import download_object_types

from io import BytesIO


def get_model(ckpt_path: str):
    model = ResNetEmbedding(pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def parse_category_from_uri(s3_uri: str) -> str:
    p = s3_uri.replace("s3://", "").split("/", 2)
    rest = p[2] if len(p) >= 3 else ""
    parts = rest.split("/")
    return parts[1] if len(parts) >= 2 else "unknown"


def embed_and_store(input_uri: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store = FeatureStore("../../infra/feature_store")

    model = get_model("resnet152_sop_embedding_best.pt").to(device)
    model.eval()

    img_bytes = download_object_types(input_uri)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x)

    vector = emb.detach().cpu().flatten().to(torch.float32).tolist()

    category = parse_category_from_uri(input_uri)

    df = pd.DataFrame(
        {
            "img_path": [input_uri],
            "event_timestamp": [pd.Timestamp.now(tz="UTC")],
            "vector": [vector],
            "s3_uri": [input_uri],
            "category": [category],
        }
    )

    store.write_to_online_store(feature_view_name="image_embeddings", df=df)
    return {
        "img_path": input_uri,
        "s3_uri": input_uri,
        "category": category,
        "dim": len(vector),
    }