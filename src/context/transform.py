import torch.nn as nn
from layers.normalization import get_batchnorm
from layers.activation import get_activation

class GCTransform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = get_batchnorm(channels)
        self.relu = get_activation("relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
