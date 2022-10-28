import json
from PIL import Image
import torch
from torch import nn

from torchvision import transforms
# Load ViT
from pytorch_pretrained_vit import ViT


import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class vit_pre(nn.Module):
    def __init__(self, num_classes=10,pretrained=True,option=None):
        super(vit_pre, self).__init__()
        if pretrained:
            print("Vit: using pretrained model")
        else:
            print("Vit: not using pretrained model")

        self.model = ViT('B_16_imagenet1k', pretrained=True)

        #change model output layer
        self.model.fc = torch.nn.Linear(768, num_classes)

        

    def forward(self, x):
        x = self.model(x)
        return x
