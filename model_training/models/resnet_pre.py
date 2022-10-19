import torch
from torch import nn

import torchvision.models as models
class resnet_pre(nn.Module):
    def __init__(self, num_classes=10,pretrained=True):
        super(resnet_pre, self).__init__()
        if pretrained:
            print("ResNet-34: using pretrained model")
        else:
            print("ResNet-34: not using pretrained model")
        self.resnet = models.resnet34(pretrained=pretrained)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
