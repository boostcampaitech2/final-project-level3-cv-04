import torch
import torch.nn as nn

class X3D_xs(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_name = "x3d_xs"
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model=self.model_name, pretrained=True)
        self.model.blocks[5].proj = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)