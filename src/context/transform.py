import torch.nn as nn

class GCTransform(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.ln = nn.LayerNorm([hidden, 1, 1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)      
        x = self.ln(x)         
        x = self.relu(x)
        x = self.conv2(x)      
        return x
