import torch.nn as nn
import torch.nn.functional as F
import torchvision


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

