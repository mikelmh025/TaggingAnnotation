import torch
from torch import nn

import torchvision.models as models

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class resnet_pre(nn.Module):
    def __init__(self, num_classes=10,pretrained=True,resnet_option='50'):
        super(resnet_pre, self).__init__()
        if pretrained:
            print("ResNet: using pretrained model")
        else:
            print("ResNet: not using pretrained model")
        if resnet_option == '18':
            self.resnet = models.resnet18(pretrained=pretrained)
            self.resnet.fc = nn.Linear(512, num_classes) # Not sure maybe 256
        elif resnet_option == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
            self.resnet.fc = nn.Linear(512, num_classes)
        elif resnet_option == '50':
            self.resnet = models.resnet50(pretrained=pretrained)
            self.resnet.fc = nn.Linear(2048, num_classes)
        

    def forward(self, x):
        x = self.resnet(x)
        return x
