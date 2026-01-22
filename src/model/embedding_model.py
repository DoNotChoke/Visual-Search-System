import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL.Image import Image

from typing import List

from model.transform import transform


class ResNetEmbedding(nn.Module):
    def __init__(self, emb_dim=512, pretrained=True, backbone_name="resnet152"):
        super(ResNetEmbedding, self).__init__()
        if backbone_name == "resnet50":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = torchvision.models.resnet50(weights=weights)
        elif backbone_name == "resnet101":
            weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = torchvision.models.resnet101(weights=weights)
        elif backbone_name == "resnet152":
            weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = torchvision.models.resnet152(weights=weights)
        else:
            raise NotImplementedError

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


def get_model(ckpt_path: str):
    model = ResNetEmbedding(pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def embed(model, imgs: List[Image]) -> List[List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = []
    for img in imgs:
        x = transform(img).unsqueeze(0)
        xs.append(x)
    x = torch.cat(xs, dim=0).to(device)

    with torch.no_grad():
        emb = model(x)

    emb = emb.detach().cpu().flatten(1)
    emb = emb.to(torch.float32)

    return [row.tolist() for row in emb]

