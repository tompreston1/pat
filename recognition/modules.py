# modules.py
import torch
import torch.nn as nn
import timm

class ConvNeXtBinary(nn.Module):
    def __init__(self, model_name='convnext_tiny', pretrained=True, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
