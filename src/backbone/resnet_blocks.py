import torch
import torch.nn as nn
from layers.conv_layer import Conv2dWrapper
from layers.normalization import get_batchnorm
from layers.activation import get_activation
from context.attention_pool import SpatialAttention
from context.global_context import aggregate_context
from context.transform import GCTransform 

class BasicBlockGCSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 use_gc=False, use_se=False, se_reduction=16, gc_reduction=16):
        super().__init__()
        self.conv1 = Conv2dWrapper(in_channels, out_channels, 3, stride, 1)
        self.bn1 = get_batchnorm(out_channels)
        self.relu = get_activation("relu")
        self.conv2 = Conv2dWrapper(out_channels, out_channels, 3, 1, 1)
        self.bn2 = get_batchnorm(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2dWrapper(in_channels, out_channels, 1, stride, 0),
                get_batchnorm(out_channels)
            )

        self.use_gc = use_gc
        if self.use_gc:
            self.attn = SpatialAttention(out_channels)         
            self.gc_transform = GCTransform(out_channels, gc_reduction)


        self.use_se = use_se
        if self.use_se:
            hidden = max(1, out_channels // se_reduction)
            self.se_fc1 = nn.Conv2d(out_channels, hidden, kernel_size=1)
            self.se_relu = nn.ReLU(inplace=True)
            self.se_fc2 = nn.Conv2d(hidden, out_channels, kernel_size=1)
            self.se_sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_gc:
            attn_map = self.attn(out)                    
            context = aggregate_context(out, attn_map)   
            out = out + self.gc_transform(context)        

        if self.use_se:
            se = out.mean(dim=(2,3), keepdim=True)      
            se = self.se_fc1(se)
            se = self.se_relu(se)
            se = self.se_fc2(se)
            se = self.se_sigmoid(se)
            out = out * se                               

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
