import torch
import torch.nn as nn
from backbone.resnet_blocks import BasicBlockGCSE  

class GCCNN(nn.Module):
    def __init__(self, num_classes=10, use_gc=True, use_se=True, se_reduction=16):
        super().__init__()
        self.layer1 = BasicBlockGCSE(3, 64, use_gc=use_gc, use_se=use_se, reduction=se_reduction)
        self.layer2 = BasicBlockGCSE(64, 128, stride=2, use_gc=use_gc, use_se=use_se, reduction=se_reduction)
        self.layer3 = BasicBlockGCSE(128, 256, stride=2, use_gc=use_gc, use_se=use_se, reduction=se_reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
