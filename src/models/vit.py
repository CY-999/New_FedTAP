
import torch
import torch.nn as nn
from .base import BaseModel

class ViT_B16_CIFAR10(BaseModel):

    def __init__(self, num_classes=10):
        super().__init__(num_classes)

        from torchvision.models import vit_b_16
        self.model = vit_b_16(num_classes=num_classes)

        self.model.conv_proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)

    def forward(self, x):
        return self.model(x)

class ViT_B16_TinyImageNet(BaseModel):

    def __init__(self, num_classes=200):
        super().__init__(num_classes)
        from torchvision.models import vit_b_16
        self.model = vit_b_16(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
