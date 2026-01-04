import torch.nn as nn

def get_batchnorm(channels):
    return nn.BatchNorm2d(channels)
