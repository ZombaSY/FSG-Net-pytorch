import timm
import torch.nn as nn
import torch


class BackboneLoader(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, **kwargs)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):

        return self.backbone.forward_features(x)
