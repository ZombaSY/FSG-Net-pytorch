import timm
import torch.nn as nn
import torch


class BackboneLoader(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, **kwargs)

    def forward(self, x):
        out = []

        x = self.backbone.stem(x)
        for stage in self.backbone.stages:
            x = stage(x)
            out.append(x)

        return out
