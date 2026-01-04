import torch

def aggregate_context(x, attn):
    B,C,H,W = x.size()
    context = (x * attn).view(B,C,-1).sum(-1).view(B,C,1,1)
    return context
