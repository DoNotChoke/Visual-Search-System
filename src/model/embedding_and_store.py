import pandas as pd
import torch

from src.model.model import ResNetEmbedding

from PIL import Image

from src.model.transform import transform
from src.vector_store.utils import download_object_types

from io import BytesIO


def get_model(ckpt_path: str):
    model = ResNetEmbedding(pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def embed_and_store( input_uri: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from feast import FeatureStore
    store = FeatureStore("../../infra/plugins/config")
    model = get_model("resnet152_sop_embedding.pt")
    img_bytes = download_object_types(input_uri)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)

    vector = emb.detach().cpu().flatten().to(torch.float32).tolist()

    df = pd.DataFrame({
        "img_path": [input_uri],
        "event_timestamp": [pd.Timestamp.now(tz="UTC")],
        "embedding": [vector],
    })
    store.write_to_online_store(feature_view_name="image_embeddings", df=df)