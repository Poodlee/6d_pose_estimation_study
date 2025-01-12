import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()
        model = models.resnet50(pretrained=True)

        self.features = nn.Sequential(
            *list(model.children())[:-2] # layer 4까지 포함
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(-2,-1)) # Global Average Pooling (B, 2048, H, W -> B, 2048)
        return x
