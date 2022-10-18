import torch
from torch import nn

import torchvision.models as models
class resnet_pre(nn.Module):
    def __init__(self, num_classes=10):
        super(resnet_pre, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
