import json
from PIL import Image
import torch
from torchvision import transforms

# Load ViT
from pytorch_pretrained_vit import ViT
num_classes = 26
model = ViT('B_16_imagenet1k', pretrained=True)

#change model output layer
model.fc = torch.nn.Linear(768, num_classes)

model.eval()

# Load image
# NOTE: Assumes an image `img.jpg` exists in the current directory
img = transforms.Compose([
    transforms.Resize((384, 384)), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])(Image.open('/media/otter/navi_hitl/FairFace2.0/test/10439.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 384, 384])

# Classify
with torch.no_grad():
    outputs = model(img)
print(outputs.shape)  # (1, 1000)

