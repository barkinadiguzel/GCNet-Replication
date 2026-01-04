import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  

    def forward(self, x):
        attn = self.conv(x)            
        attn = attn.view(x.size(0), -1) 
        attn = F.softmax(attn, dim=1)
        attn = attn.view(x.size(0), 1, x.size(2), x.size(3))
        return attn
